"""Fit matched phenotype ridge candidates on the frozen SNU-423 design."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from src.goal_tracks.t1_data import eligibility
from src.p123.p1_score import ENDPOINTS, StudyBook
from src.p4.common import WellProjection
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE, table
from src.r13_optimization.mechanism import bit_tanimoto
from src.r19_plan.hcc_phenotype_support import profiles
from src.r19_plan.hcc_scaffold import CASES, CI, DOSE, LINE_ID, MINIMUM, TARGET, scaffold_draw
from src.r19_plan.hcc_scaffold_refit import cut_training, projected_training, require_hash, training_profiles
from src.r19_plan.i_kernel import masked_gram
from src.r19_plan.i_kernel_arms import fit_optional_profiles, shuffle_views
from src.r19_plan.r18_preflight import checked_tables


VIEWS = dict(I2=("I2",), I4=("I4",), combined=("I2", "I4"))
REFERENCES = ("B0", "scaffold_ecfp", "scaffold_chemb")
ARMS = (*REFERENCES, *(f"{name}_{mode}" for name in VIEWS for mode in ("profile", "mask", "shuffle")))


def inputs(args):
    data = args.reports / "data"
    recorded = data / "hcc_scaffold_refit"
    source = require_hash(args.previous / "freeze.json", sha256(recorded / "freeze.json"))
    old = read_json(source)
    visible = args.previous / "visible"
    visible_pin = require_hash(visible / "receipt.json", old["fit_input_sha256"])
    feature_pin = args.features
    require_hash(feature_pin, old["features_sha256"])
    blocks = table(require_hash(args.reports.parent / "r12_multiaxis/group_scaffolds.tsv", old["scaffold_sha256"]))
    proof = checked_tables(args.support, data / "hcc_phenotype_support/audit/receipt.json")
    if proof["expression_read"] or proof["visible_sha256"] != sha256(visible_pin):
        raise ValueError("Phenotype support differs from the pinned expression-free audit")
    available = table(args.support / "summary.tsv").set_index("view", verify_integrity=True)
    if not available.loc[["I2", "I4", "both"], "gate"].eq(True).all():
        raise ValueError("An assay view did not meet the registered support gate")
    stamp = dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__), source_sha256=sha256(source),
                 support_sha256=sha256(args.support / "receipt.json"), features_sha256=old["features_sha256"])
    return old, visible, read_json(visible_pin), blocks.set_index(GROUP, verify_integrity=True).scaffold_component, stamp


def data_slice(args, context, fold, role):
    old, visible, meta, _, _ = context
    key = f"f{fold}-{role}"
    held_name = f"f{fold}-held.tsv"
    held = table(require_hash(visible / held_name, meta["files"][held_name]))
    if held.quarantined.any() or held.plate_id.eq("plate13").any():
        raise ValueError("Reserved Tahoe metadata reached prediction or scoring")
    projection = WellProjection(held)
    line = projection.rows[LINE].eq(LINE_ID).to_numpy()
    query = projection.rows.loc[line].reset_index(drop=True)
    folder = args.previous / key
    case = read_json(require_hash(folder / "receipt.json", old["cases"][key]))
    if not query[KEY + [GROUP]].equals(table(require_hash(folder / "rows.tsv", case["files"]["rows.tsv"]))):
        raise ValueError("The original SNU-423 axis differs from the current rows")
    packed = require_hash(folder / "forecasts.npz", case["files"]["forecasts.npz"])
    with np.load(packed, allow_pickle=False) as original:
        refs = {name: original[name] for name in REFERENCES}
    return projection, line, query, refs


def ridge(gram, prototypes):
    n = len(prototypes)
    fit = gram[:n, :n].copy()
    fit.flat[::n + 1] += 1
    return gram[n:, :n] @ cho_solve(cho_factor(fit), prototypes)


def check_fit_support(args, fold, mapped, fitting, active):
    support = table(args.support / "fold_support.tsv")
    summary = support.loc[support.fold.eq(fold)].set_index("view", verify_integrity=True)
    masks = {view: present for view, (_, present) in mapped.items()}
    for key, mask in {**masks, "both": masks["I2"] & masks["I4"]}.items():
        if (int(mask[fitting].sum()) != summary.at[key, "fit_support"]
                or int(mask[active].sum()) != summary.at[key, "query_support"]):
            raise ValueError("The measured fold support changed from its pre-fit audit")


def predict(args, context):
    _, visible, meta, blocks, stamp = context
    fold, role = args.fold, args.role
    _, _, query, previous = data_slice(args, context, fold, role)
    names = (f"f{fold}-fit.tsv", f"f{fold}-{role}.npy")
    fit = table(require_hash(visible / names[0], meta["files"][names[0]]))
    if fit.quarantined.any() or fit.plate_id.eq("plate13").any() or set(fit[GROUP]) & set(query[GROUP]):
        raise ValueError("A held or protected assay row reached fitting")
    delta = np.load(require_hash(visible / names[1], meta["files"][names[1]]), allow_pickle=False)
    if delta.shape != (len(fit), meta["genes"]) or not np.isfinite(delta).all():
        raise ValueError("Invalid pre-registered fit array")
    masks, _ = cut_training(fit, query, blocks, fold)
    selected = masks["scaffold"]
    fit_rows, labels = projected_training(fit.loc[selected].reset_index(drop=True), delta[selected])
    entities, prototypes = training_profiles(fit_rows, labels)
    if set(entities.index.map(blocks)) & set(query[GROUP].map(blocks)):
        raise ValueError("Evaluation scaffold leaked into the reference prototypes")
    parent_axis = table(args.support / "parent_axis.tsv").set_index(GROUP, verify_integrity=True).parent
    fitting, active = parent_axis.index.isin(entities.index), parent_axis.index.isin(query[GROUP])
    raw_profiles = {}
    reservations = read_json(args.support / "receipt.json")["view_receipts"]
    for name in ("I2", "I4"):
        root = args.support / name
        require_hash(root / "receipt.json", reservations[name])
        checked_tables(root, args.reports / f"data/hcc_phenotype_support/audit/{name}/receipt.json")
        raw_profiles[name] = profiles(root, parent_axis)
    mapped = fit_optional_profiles(raw_profiles, fitting)
    check_fit_support(args, fold, mapped, fitting, active)
    groups = entities.index.append(pd.Index(sorted(set(query[GROUP]))))
    chosen = parent_axis.index.get_indexer(groups)
    expanded = groups.get_indexer(entities.index.append(pd.Index(query[GROUP])))
    if (chosen < 0).any() or (expanded < 0).any():
        raise ValueError("An active parent has no registered feature coordinate")
    compact = {name: (vec[chosen], present[chosen]) for name, (vec, present) in mapped.items()}
    shuffled = shuffle_views(compact, len(entities), seed=1501)
    with np.load(args.features, allow_pickle=False) as assets:
        entity_axis = pd.Index(assets["entity"])
        index = entity_axis.get_indexer(entities.tolist() + query[ENTITY].tolist())
        if not entity_axis.is_unique or (index < 0).any():
            raise ValueError("Unresolved original structure axis")
        fp = assets["E"][index].astype(np.float64)
    structure = bit_tanimoto(fp, fp)
    ref = ridge(structure, prototypes)
    diff = np.max(np.abs(ref - previous["scaffold_ecfp"]))
    if diff > 1e-10:
        raise ValueError(f"Structural reference replay differs by {diff:g}")
    forecasts = previous.copy()
    flags = {name: compact[name][1][expanded[len(entities):]] for name in compact}
    for name, views in VIEWS.items():
        available = np.logical_or.reduce([flags[view] for view in views])
        for mode, source in (("profile", compact), ("mask", compact), ("shuffle", shuffled)):
            blocks_for_gram = [(np.ones((len(expanded), 1)) if mode == "mask" else source[v][0][expanded],
                                source[v][1][expanded]) for v in views]
            forecast = ridge(masked_gram(structure, blocks_for_gram), prototypes)
            forecast[~available] = previous["scaffold_ecfp"][~available]
            forecasts[f"{name}_{mode}"] = forecast
    dest = args.out / f"f{fold}-{role}"
    if dest.exists():
        raise FileExistsError(dest)
    with TemporaryDirectory(dir=args.out, prefix=".hcc-phenotype-") as tmp:
        stage = Path(tmp)
        np.savez_compressed(stage / "forecasts.npz", **forecasts)
        np.savez_compressed(stage / "flags.npz", **flags)
        write_json(stage / "receipt.json", dict(stamp, fold=fold, role=role, structure_max_abs_diff=float(diff),
                   files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(dest)


def forecast_slice(args, context, fold, role):
    _, _, _, _, stamp = context
    projection, line, query, previous = data_slice(args, context, fold, role)
    root = args.out / f"f{fold}-{role}"
    receipt = read_json(root / "receipt.json")
    if ((receipt["fold"], receipt["role"]) != (fold, role)
            or any(receipt[key] != digest for key, digest in stamp.items())):
        raise ValueError("Unexpected forecast case or provenance")
    with np.load(require_hash(root / "forecasts.npz", receipt["files"]["forecasts.npz"]), allow_pickle=False) as packed:
        arrays = dict(packed)
    with np.load(require_hash(root / "flags.npz", receipt["files"]["flags.npz"]), allow_pickle=False) as packed:
        flags = dict(packed)
    if (set(arrays) != set(ARMS) or set(flags) != {"I2", "I4"}
            or any(values.shape != previous["B0"].shape or not np.isfinite(values).all() for values in arrays.values())
            or any(values.shape != (len(query),) or values.dtype.kind != "b" for values in flags.values())
            or any(not np.array_equal(arrays[name], values) for name, values in previous.items())):
        raise ValueError("Invalid phenotype arms, axes or fallback references")
    for name, views in VIEWS.items():
        absent = ~np.logical_or.reduce([flags[v] for v in views])
        if any(not np.array_equal(arrays[f"{name}_{mode}"][absent], previous["scaffold_ecfp"][absent])
               for mode in ("profile", "mask", "shuffle")):
            raise ValueError("A missing-profile row lacks the exact structure fallback")
    return (projection, line, query, arrays, flags), sha256(root / "receipt.json")


def collect(args, context):
    cases, hashes = {}, {}
    for fold, role in CASES:
        case, hashes[f"f{fold}-{role}"] = forecast_slice(args, context, fold, role)
        cases[fold, role] = case
    for fold in range(3):
        a, b = cases[fold, "A"][4], cases[fold, "B"][4]
        if any(not np.array_equal(a[key], b[key]) for key in a):
            raise ValueError("Physical vehicle roles changed the observed phenotype support")
    return cases, dict(context[-1], cases=hashes)


def score(args, context, cases, manifest):
    freeze = require_hash(args.out / "freeze.json", args.expected_sha256)
    if read_json(freeze) != manifest:
        raise ValueError("The independent phenotype forecast freeze differs")
    target = args.previous / "score"
    registered = args.reports / "data/hcc_scaffold_refit/score/receipt.json"
    source = read_json(require_hash(target / "receipt.json", sha256(registered)))
    blocks = context[3]
    books = {scope: StudyBook(sorted(blocks.index), DOSE, MINIMUM, "SNU-423 external phenotype")
             for scope in ("full", *VIEWS)}
    for book in books.values():
        book.draw = scaffold_draw(book.index, blocks)
    counts = []
    for (fold, role), (projection, line, query, arrays, flags) in cases.items():
        file = f"f{fold}-{'B' if role == 'A' else 'A'}.npy"
        truth = projection.centered(np.load(require_hash(target / file, source["files"][file]), allow_pickle=False))[line]
        scopes = dict(full=np.ones(len(query), dtype=bool), **{
            name: np.logical_and.reduce([flags[v] for v in views]) for name, views in VIEWS.items()})
        for name, mask in scopes.items():
            rows = eligibility(query.assign(position=np.arange(len(query))).loc[mask], 0)
            indices = rows.pop("position").to_numpy(dtype=int)
            if len(indices) == 0:
                raise ValueError("No eligible physical comparison pools remain")
            active = ARMS if name == "full" else (*REFERENCES, *(f"{name}_{mode}" for mode in ("profile", "mask", "shuffle")))
            books[name].add(rows, truth[indices], {arm: arrays[arm][indices] for arm in active})
            counts.append(dict(scope=name, fold=fold, role=role, parents=rows[GROUP].nunique(), wells=len(rows)))
    results, comparisons = [], []
    for scope, book in books.items():
        points, draws = book.results()
        points = points.loc[points.stratum.eq(TARGET)].assign(scope=scope)
        results.append(points)
        lookup = points.set_index("arm")
        for name in (VIEWS if scope == "full" else (scope,)):
            candidate = f"{name}_profile"
            for baseline in (*REFERENCES, f"{name}_mask", f"{name}_shuffle"):
                for metric in ENDPOINTS:
                    difference = lookup.at[candidate, metric] - lookup.at[baseline, metric]
                    lo, hi = np.quantile(draws[TARGET, candidate, metric] - draws[TARGET, baseline, metric], CI)
                    improvement = -100 * difference / lookup.at[baseline, metric] if metric == "contrast_rmse" else 100 * difference
                    comparisons.append(dict(scope=scope, arm=candidate, reference=baseline, metric=metric, difference=difference,
                                            low99=lo, high99=hi, improvement_percent_or_pp=improvement))
    out = args.out / "scored"
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=args.out, prefix=".hcc-phenotype-score-") as tmp:
        stage = Path(tmp)
        points = pd.concat(results, ignore_index=True)
        points[["sse", "sst"]] *= context[2]["genes"] / 2000
        points.to_csv(stage / "points.tsv", sep="\t", index=False)
        pd.DataFrame(comparisons).to_csv(stage / "paired.tsv", sep="\t", index=False)
        pd.DataFrame(counts).to_csv(stage / "scored_support.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(manifest, freeze_sha256=args.expected_sha256,
                   files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("predict", "freeze", "score"))
    for field in ("out", "previous", "reports", "protocol", "features", "support"):
        cli.add_argument("--" + field, type=Path, required=True)
    cli.add_argument("--fold", type=int, choices=range(3))
    cli.add_argument("--role", choices=("A", "B"))
    cli.add_argument("--expected-sha256")
    args = cli.parse_args()
    context = inputs(args)
    if args.action == "predict":
        predict(args, context)
    else:
        cases, manifest = collect(args, context)
        if args.action == "freeze":
            dest = args.out / "freeze.json"
            if dest.exists():
                raise FileExistsError(dest)
            write_json(dest, manifest)
        else:
            score(args, context, cases, manifest)
