"""Use a separable structure and cell-response kernel on the locked Tahoe endpoint."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import MIN_GROUPS, WELL, eligibility
from src.p123.p1_score import ENDPOINTS, StudyBook
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, LINE
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.mechanism import bit_tanimoto
from src.r13_optimization.objectives import center
from src.r19_plan.hcc_fixed_panel_refit import (GAINS, MODES, cohort, observed, projection, provenance,
                                               selection_curve)
from src.r19_plan.hcc_scaffold import CASES, CI, DOSE, LINE_ID, MINIMUM, TARGET, scaffold_draw
from src.r19_plan.hcc_scaffold_refit import require_hash, ridge


NEW = ("Identity", "Shared", "Self_only", "Aux_shuffle")
ARMS = ("B0", "State", *NEW)


def lineage(args):
    parts, panel, genes, blocks, stamp = provenance(args)
    pin = args.reports / "r19_plan/data/hcc_fixed44/freeze.json"
    freeze = require_hash(args.reference / "freeze.json", sha256(pin))
    previous = read_json(freeze)
    fields = ("visible_sha256", "score_pin_sha256", "panel_sha256", "genes_sha256", "feature_sha256")
    if any(previous[k] != stamp[k] for k in fields):
        raise ValueError("The prior State and ECFP forecasts use a different data contract")
    lines = next(iter(panel.values()))["lines"]
    if any(item["lines"] != lines for item in panel.values()):
        raise ValueError("The shared model needs one common panel across plates")
    return parts, panel, genes, blocks, previous, lines, dict(stamp, code_sha256=sha256(__file__), prior_freeze_sha256=sha256(pin))


def fit_responses(args, parts, fold, role, cohort_projection, rows, lines):
    file = f"f{fold}-{role}.npy"
    path = require_hash(args.previous / "visible" / file, parts["visible"]["files"][file])
    delta = np.load(path, mmap_mode="r", allow_pickle=False)
    indices = []
    for well in rows[WELL]:
        active = np.asarray(cohort_projection.wells[well])
        order = pd.Index(cohort_projection.original.iloc[active][LINE]).get_indexer(lines)
        if (order < 0).any():
            raise ValueError("A fit well lacks the common 44-cell panel")
        indices.append(active[order])
    values = delta[np.stack(indices)].astype(np.float64)
    if delta.shape != (len(cohort_projection.original), 978) or not np.isfinite(values).all():
        raise ValueError("A fit response differs from the registered gene axis")
    return values - values.mean(axis=1, keepdims=True)


def subset(rows, values, allowed):
    scope = eligibility(rows.assign(position=np.arange(len(rows))).loc[allowed], MINIMUM, DOSE).reset_index(drop=True)
    if scope[GROUP].nunique() < 20:
        raise ValueError("An inner or outer task lacks 20 independent fitting parents")
    labels = values[scope.pop("position").to_numpy(dtype=int)]
    centered = center(Contrast(scope, MIN_GROUPS), labels.reshape(len(labels), -1)).reshape(labels.shape)
    return scope, centered


def per_parent_profiles(rows, labels):
    leader = rows.sort_values([GROUP, ENTITY]).drop_duplicates(GROUP).set_index(GROUP)[ENTITY]
    keep = rows[ENTITY].eq(rows[GROUP].map(leader)).to_numpy()
    indices = rows.loc[keep].groupby(GROUP, sort=True).indices
    entities = leader.loc[list(indices)]
    chosen = labels[keep]
    profile = np.stack([chosen[x].mean(axis=0) for x in indices.values()])
    return entities, profile


def cell_gram(y):
    flat = y.transpose(1, 0, 2).reshape(y.shape[1], -1)
    norms = np.linalg.norm(flat, axis=1)
    if not np.all(norms > 1e-12):
        raise ValueError("An input cell profile has no fitting contrast")
    unit = flat / norms[:, None]
    return 0.5 * (unit @ unit.T + np.eye(len(unit)))


def kronecker_predict(kernel, cross, cells, targets, target_cell):
    compounds, contexts, genes = targets.shape
    d, u = np.linalg.eigh(kernel)
    c, v = np.linalg.eigh(cells)
    if min(d.min(), c.min()) < -1e-9:
        raise ValueError("A separable model kernel is not positive semidefinite")
    encoded = (u.T @ targets.reshape(compounds, -1)).reshape(compounds, contexts, genes)
    encoded = np.einsum("ijg,jk->ikg", encoded, v, optimize=True)
    encoded /= (1 + np.maximum(d, 0)[:, None] * np.maximum(c, 0)[None, :])[:, :, None]
    target = np.einsum("ikg,k->ig", encoded, v[target_cell] * c)
    return (cross @ u) @ target


def reshuffle(labels, target_cell):
    rng = np.random.default_rng(4401)
    result = labels.copy()
    for cell in range(labels.shape[1]):
        if cell == target_cell:
            continue
        order = rng.permutation(len(labels))
        result[order, cell] = labels[np.roll(order, 1), cell]
    return result


def model_arms(train, labels, query, features, mapping, target):
    reps, y = per_parent_profiles(train, labels)
    fi, qi = [mapping.get_indexer(keys) for keys in (reps, query[ENTITY])]
    if (fi < 0).any() or (qi < 0).any():
        raise ValueError("A fit or target drug has no original ECFP4 vector")
    chemistry = bit_tanimoto(features[fi], features[fi])
    cross = bit_tanimoto(features[qi], features[fi])
    cell = cell_gram(y)
    one = np.zeros_like(y)
    one[:, target] = y[:, target]
    result = {"Identity": ridge(features[fi], features[qi], y[:, target], bit_tanimoto)}
    for name, variants in (("Shared", y), ("Self_only", one), ("Aux_shuffle", reshuffle(y, target))):
        result[name] = kronecker_predict(chemistry, cross, cell, variants, target)
    return result


def prior_arrays(args, previous, mode, fold, role):
    key = f"{mode}/f{fold}-{role}"
    arrays = {}
    for lane, name in (("cpu", "ECFP"), ("state", "State")):
        path = args.reference / "forecasts" / key / lane
        record = read_json(require_hash(path / "receipt.json", previous["cases"][f"{key}/{lane}"]))
        with np.load(require_hash(path / "forecasts.npz", record["forecast_sha256"]), allow_pickle=False) as f:
            arrays[name] = f[name]
    return arrays


def predict(args, ctx):
    parts, _, _, blocks, previous, lines, stamp = ctx
    target = lines.index(LINE_ID)
    fold, role, mode = args.fold, args.role, args.mode
    visible, held = (projection(args, parts, fold, p) for p in ("fit", "held"))
    rows, _, _ = cohort(visible)
    query, _, _ = cohort(held)
    y = fit_responses(args, parts, fold, role, visible, rows, lines)
    shared_scaffold = rows[GROUP].map(blocks).isin(query[GROUP].map(blocks)).to_numpy()
    allowed = ~shared_scaffold if mode == "purged" else np.ones(len(rows), dtype=bool)
    if set(rows.loc[allowed, GROUP]) & set(query[GROUP]) or (mode == "purged" and shared_scaffold[allowed].any()):
        raise ValueError("A protected held chemical or scaffold reached any fitting context")
    with np.load(require_hash(args.features, stamp["feature_sha256"]), allow_pickle=False) as src:
        axis, values = pd.Index(src["entity"]), src["E"].astype(np.float64)
    if not axis.is_unique:
        raise ValueError("The original fingerprint axis is ambiguous")
    losses = {name: [] for name in NEW}
    for split in (0, 1):
        train, fit = subset(rows, y, allowed & rows.inner_fold.ne(split).to_numpy())
        select, labels = subset(rows, y, allowed & rows.inner_fold.eq(split).to_numpy())
        if set(train[GROUP]) & set(select[GROUP]):
            raise ValueError("An inside-selection chemical overlaps a training context")
        forecasts = model_arms(train, fit, select, values, axis, target)
        for name, loss in selection_curve(select, labels[:, target], forecasts).items():
            losses[name].append(loss)
    train, fit = subset(rows, y, allowed)
    raw = model_arms(train, fit, query, values, axis, target)
    selected = {name: float(GAINS[min(range(len(GAINS)), key=lambda j: (round(np.mean(loss, axis=0)[j], 10), j))])
                for name, loss in losses.items()}
    preds = {name: selected[name] * pred for name, pred in raw.items()}
    prior = prior_arrays(args, previous, mode, fold, role)["ECFP"]
    replay = float(np.max(np.abs(preds["Identity"] - prior)))
    if replay > 1e-10:
        raise ValueError(f"Old single-context model replay differs by {replay:g}")
    output = args.run / "forecasts" / mode / f"f{fold}-{role}"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent, prefix=".multitask-") as tmp:
        stage = Path(tmp)
        file = stage / "forecasts.npz"
        np.savez_compressed(file, **preds)
        write_json(stage / "receipt.json", dict(stamp, fold=fold, role=role, mode=mode, forecast_sha256=sha256(file),
                                               gains=selected, source_parents=train[GROUP].nunique(), identity_max_diff=replay))
        stage.rename(output)


def collect(args, ctx):
    parts, _, genes, _, prior, _, stamp = ctx
    cases, hashes, selections = {}, {}, []
    for mode in MODES:
        for fold, role in CASES:
            key = f"{mode}/f{fold}-{role}"
            projected = projection(args, parts, fold, "held")
            query, _, _ = cohort(projected)
            old = prior_arrays(args, prior, mode, fold, role)
            root = args.run / "forecasts" / key
            receipt = read_json(root / "receipt.json")
            if (any(receipt[k] != v for k, v in stamp.items()) or receipt["identity_max_diff"] > 1e-10
                    or (receipt["mode"], receipt["fold"], receipt["role"]) != (mode, fold, role)):
                raise ValueError("The shared-learner forecast has a different case or provenance")
            with np.load(require_hash(root / "forecasts.npz", receipt["forecast_sha256"]), allow_pickle=False) as f:
                if set(f.files) != set(NEW):
                    raise ValueError("The response-sharing model inventory changed")
                new = {arm: f[arm] for arm in NEW}
            shape = len(query), len(genes)
            values = dict(B0=np.zeros(shape), State=old["State"], **new)
            if any(v.shape != shape or not np.isfinite(v).all() for v in values.values()) or np.max(np.abs(values["Identity"]-old["ECFP"])) > 1e-10:
                raise ValueError("An invalid forecast axis or unmatched single-context baseline was found")
            hashes[key] = sha256(root / "receipt.json")
            selections.extend(dict(mode=mode, fold=fold, role=role, arm=name, gain=choice,
                fit_parents=receipt["source_parents"], identity_max_diff=receipt["identity_max_diff"]) for name, choice in receipt["gains"].items())
            cases[mode, fold, role] = (projected, query, values)
    return cases, selections, dict(stamp, cases=hashes)


def summarize(books):
    point, effects = [], []
    for (mode, unit), book in books.items():
        rows, draws = book.results()
        summaries = rows.loc[rows.stratum.eq(TARGET)].assign(fit_scheme=mode, unit=unit)
        if not summaries.parent_groups.eq(203).all() or not summaries.rows_roles.eq(550).all():
            raise ValueError("The response-sharing score support differs from the fixed-panel reference")
        summaries[["sse", "sst"]] *= 978 / 2000
        point.append(summaries)
        p = summaries.set_index("arm")
        for baseline in ("B0", "Identity", "State", "Self_only", "Aux_shuffle"):
            for metric in ENDPOINTS:
                candidate, ref = p.at["Shared", metric], p.at[baseline, metric]
                low, high = np.quantile(draws[TARGET, "Shared", metric] - draws[TARGET, baseline, metric], CI)
                improvement = 100*(ref-candidate)/ref if metric=="contrast_rmse" else 100*(candidate-ref)
                effects.append(dict(mode=mode, unit=unit, arm="Shared", reference=baseline, metric=metric, difference=candidate-ref,
                                    low99=low, high99=high, improvement_percent_or_pp=improvement))
    return pd.concat(point, ignore_index=True), pd.DataFrame(effects)


def finish(args, ctx):
    cases, selections, stamp = collect(args, ctx)
    freeze = args.run / "freeze.json"
    if args.phase == "freeze":
        if freeze.exists():
            raise FileExistsError(freeze)
        write_json(freeze, stamp)
        return
    if read_json(require_hash(freeze, args.digest)) != stamp:
        raise ValueError("Response-sharing scoring requires the independently frozen forecasts")
    parts, panel, _, blocks, *_ = ctx
    books = {(mode, unit): StudyBook(sorted(blocks.index), DOSE, MINIMUM, "fixed44 shared")
             for mode in MODES for unit in ("parent", "scaffold")}
    for mode in MODES:
        book = books[mode, "scaffold"]
        book.draw = scaffold_draw(book.index, blocks)
    for (mode, fold, role), (projected, query, values) in cases.items():
        other = "B" if role == "A" else "A"
        y = observed(args, parts, panel, fold, other, "score", projected, query)
        for unit in ("parent", "scaffold"):
            books[mode, unit].add(query, y, values)
    point, effects = summarize(books)
    out = args.run / "score"
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=args.run, prefix=".multitask-score-") as tmp:
        stage = Path(tmp)
        point.to_csv(stage / "points.tsv", sep="\t", index=False)
        effects.to_csv(stage / "paired.tsv", sep="\t", index=False)
        pd.DataFrame(selections).to_csv(stage / "settings.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(stamp, freeze_sha256=args.digest,
                   files={file.name: sha256(file) for file in stage.iterdir()}))
        stage.rename(out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("predict", "freeze", "score"))
    for field in ("reports", "protocol", "previous", "panel-lock", "reference", "run", "features"):
        cli.add_argument("--" + field, type=Path)
    cli.add_argument("--fold", type=int, choices=range(3))
    cli.add_argument("--role", choices="AB")
    cli.add_argument("--mode", choices=MODES)
    cli.add_argument("--digest")
    args = cli.parse_args()
    ctx = lineage(args)
    (predict if args.phase == "predict" else finish)(args, ctx)
