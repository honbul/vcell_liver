"""Benchmark the fixed-panel SNU-423 endpoint against a matched State refit."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import HIGH_PLATES, MIN_GROUPS, WELL, eligibility
from src.p123.p1_score import ENDPOINTS, StudyBook
from src.p4.common import WellProjection
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, LINE, group_weights, load_data, table
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.mechanism import bit_tanimoto
from src.r13_optimization.objectives import center, row_products
from src.r19_plan.hcc_dose_sharing import cosine_kernel
from src.r19_plan.hcc_scaffold import CASES, CI, DOSE, LINE_ID, MINIMUM, TARGET, scaffold_draw
from src.r19_plan.hcc_scaffold_refit import require_hash, ridge, training_profiles
from src.r19_plan.hcc_vehicle_panel import cohort, fixed_vectors, old_rows, receipts


MODES = ("purged", "full")
CPU = ("hybrid", "ECFP", "ChemBERTa", "sibling")
STATE = ("State",)
ARMS = ("B0", *CPU, *STATE)
GAINS = np.array([0., .125, .25, .5, .75, 1.])


def provenance(args):
    parts, stamp = receipts(args)
    stamp["code_sha256"] = sha256(__file__)
    root = args.reports / "r19_plan/data"
    pin = root / "hcc_vehicle_panel/lock/receipt.json"
    manifest = read_json(require_hash(args.panel_lock / "receipt.json", sha256(pin)))
    if not manifest["metadata_only"] or manifest["visible_sha256"] != stamp["visible_sha256"]:
        raise ValueError("The metadata-defined panel does not match this archived cohort")
    pools = read_json(require_hash(args.panel_lock / "panel_manifest.json", manifest["files"]["panel_manifest.json"]))
    lines = sorted(set.intersection(*(set(v["lines"]) for fold in pools.values() for v in fold["common_all_wells"].values())))
    if len(lines) != 44 or LINE_ID not in lines:
        raise ValueError("The registered global 44-cell-line panel changed")
    blocks_path = require_hash(args.reports / "r12_multiaxis/group_scaffolds.tsv", parts["visible"]["scaffold_sha256"])
    blocks = table(blocks_path).set_index(GROUP, verify_integrity=True).scaffold_component
    genes_path = require_hash(args.previous / "visible/genes.tsv", parts["visible"]["files"]["genes.tsv"])
    genes = table(genes_path).token_id_tahoe.to_numpy(dtype=int)
    if len(genes) != 978 or len(np.unique(genes)) != 978 or len(blocks) != 203:
        raise ValueError("The registered gene or scaffold axis changed")
    panel = {plate: dict(lines=lines) for plate in HIGH_PLATES}
    return parts, panel, genes, blocks, dict(stamp, panel_sha256=manifest["files"]["panel_manifest.json"],
        panel_receipt_sha256=sha256(pin), genes_sha256=sha256(genes_path), feature_sha256=parts["visible"]["features_sha256"])


def projection(args, part, fold, phase):
    return WellProjection(old_rows(args, part["visible"], f"f{fold}-{phase}.tsv"))


def observed(args, part, panel, fold, role, phase, projected, query):
    file = f"f{fold}-{role}.npy"
    path = args.previous / phase / file
    values = np.load(require_hash(path, part[phase]["files"][file]), allow_pickle=False, mmap_mode="r")
    if values.shape != (len(projected.original), 978):
        raise ValueError("Observed response array and locked metadata disagree")
    return fixed_vectors(projected, query, panel, values)


def eligible_slice(rows, values, mask):
    eligible = eligibility(rows.assign(position=np.arange(len(rows))).loc[mask], MINIMUM, DOSE).reset_index(drop=True)
    idx = eligible.pop("position").to_numpy(dtype=int)
    if eligible[GROUP].nunique() < 20:
        raise ValueError("Fewer than 20 source or inner-select parents remain after pool filtering")
    return eligible, center(Contrast(eligible, MIN_GROUPS), values[idx])


def chemistry(args, stamp, train, labels, query):
    with np.load(require_hash(args.features, stamp["feature_sha256"]), allow_pickle=False) as asset:
        mapping = pd.Index(asset["entity"])
        entities, profiles = training_profiles(train, labels)
        fi, qi = [mapping.get_indexer(ids) for ids in (entities, query[ENTITY])]
        if (fi < 0).any() or (qi < 0).any() or not mapping.is_unique:
            raise ValueError("A target or fit entity lacks the original ECFP/ChemBERTa axis")
        x, c = [asset[key].astype(np.float64) for key in ("E", "C")]
    similarity = bit_tanimoto(x[qi], x[fi])
    maxima = similarity.max(axis=1)
    close = maxima >= .70
    closest = profiles[similarity.argmax(axis=1)]
    ecfp = ridge(x[fi], x[qi], profiles, bit_tanimoto)
    return dict(ECFP=ecfp, ChemBERTa=ridge(c[fi], c[qi], profiles, cosine_kernel),
                sibling=closest * close[:, None], hybrid=np.where(close[:, None], closest, ecfp)), maxima


def state_inputs(args, parts, genes, role):
    pin = require_hash(args.controls / "receipt.json", parts["visible"]["source_counts_sha256"])
    entry = read_json(pin)
    for name in (f"{role}_rows.tsv", f"{role}_counts.npz"):
        require_hash(args.controls / name, entry["files"][name])
    return load_data(args.controls, role, panel=genes, mask=lambda rows: (
        rows.plate_id.isin(HIGH_PLATES) & rows[LINE].eq(LINE_ID) & ~rows.quarantined).to_numpy())


def state_model(args, stamp, control, genes, role, train, labels, query):
    from src.goal_tracks.t2_state import fit_model

    register, counts = control
    control_axis = pd.Index(register[WELL])
    with np.load(require_hash(args.features, stamp["feature_sha256"]), allow_pickle=False) as asset:
        entities = pd.Index(asset["entity"])
        indices = [entities.get_indexer(rows[ENTITY]) for rows in (train, query)]
        if any((index < 0).any() for index in indices):
            raise ValueError("State molecular input lacks a registered parent")
        drug_fit, drug_query = [asset["E"][index].astype(np.float32) for index in indices]

    def well_units(rows):
        requested = rows[f"control_{role}"]
        index = control_axis.get_indexer(requested)
        if (index < 0).any() or not register.iloc[index].plate_id.reset_index(drop=True).equals(rows.plate_id.reset_index(drop=True)):
            raise ValueError("State control exact identity or plate differs")
        keys = list(zip(requested, rows[LINE], strict=True))
        return keys, counts[index].toarray().astype(np.float32)

    fit_units, fit_control = well_units(train)
    query_units, query_control = well_units(query)
    adapter = SimpleNamespace(source=SimpleNamespace(role=role), panel=genes, train_units=fit_units, query_units=query_units,
        weight=group_weights(train), train=SimpleNamespace(treated=(labels+fit_control).astype(np.float32), chemical={"E": drug_fit}),
        query=SimpleNamespace(control=query_control, chemical={"E": drug_query}))
    result, receipt = fit_model(adapter, args.state, (args.cells,))
    # The upstream adapter emits a legacy description of its default 2,000-gene run.
    receipt.update(model="Official State Transition GPT2 fit to the 978-gene fixed-panel SNU-423 endpoint",
                   panel_genes=len(genes))
    return {"State": result}, receipt


def selection_curve(rows, target, estimates):
    metric = Contrast(rows, MIN_GROUPS)
    weight = metric.primary_weight
    result = {}
    for name, pred in estimates.items():
        centered = center(metric, pred)
        errors = [gain * centered - target for gain in GAINS]
        result[name] = np.array([weight @ row_products(err, err) / weight.sum() for err in errors])
    return result


def predict(args, ctx):
    parts, panel, genes, blocks, stamp = ctx
    fold, role, mode = args.fold, args.role, args.mode
    projected, held = (projection(args, parts, fold, phase) for phase in ("fit", "held"))
    query, _, _ = cohort(held)
    rows, _, _ = cohort(projected)
    values = observed(args, parts, panel, fold, role, "visible", projected, rows)
    withheld = set(query[GROUP])
    shared = rows[GROUP].map(blocks).isin(query[GROUP].map(blocks)).to_numpy()
    allowed = ~shared if mode == "purged" else np.ones(len(rows), dtype=bool)
    if (withheld & set(rows[GROUP]) or not set(rows.inner_fold) <= {0, 1}
            or rows.groupby(GROUP).inner_fold.nunique().gt(1).any()):
        raise ValueError("The held parent or inner fit registry changed")
    best, info = {}, {}
    control = state_inputs(args, parts, genes, role) if args.lane == "state" else None

    def generate(fit, truth, target):
        if args.lane == "state":
            return state_model(args, stamp, control, genes, role, fit, truth, target)
        result, maximum = chemistry(args, stamp, fit, truth, target)
        return result, dict(tanimoto_0_7_parents=target.loc[maximum >= .70, GROUP].nunique())

    for split in (0, 1):
        train, labels = eligible_slice(rows, values, allowed & rows.inner_fold.ne(split).to_numpy())
        select, truth = eligible_slice(rows, values, allowed & rows.inner_fold.eq(split).to_numpy())
        if set(train[GROUP]) & set(select[GROUP]) or (mode == "purged" and set(train[GROUP].map(blocks)) & set(query[GROUP].map(blocks))):
            raise ValueError("A parent or outer evaluation scaffold entered an inner source fit")
        preds, _ = generate(train, labels, select)
        info[f"inner{split}"] = dict(fit=train[GROUP].nunique(), select=select[GROUP].nunique())
        for name, curve in selection_curve(select, truth, preds).items():
            best.setdefault(name, []).append(curve)
    outer, labels = eligible_slice(rows, values, allowed)
    preds, model = generate(outer, labels, query)
    info["outer_fit"] = outer[GROUP].nunique()
    info["held_scaffolds_in_fit"] = len(set(outer[GROUP].map(blocks)) & set(query[GROUP].map(blocks)))
    if mode == "purged" and info["held_scaffolds_in_fit"]:
        raise ValueError("A held scaffold entered an outer refit")
    gain = {name: float(GAINS[min(range(len(GAINS)), key=lambda i: (round(np.mean(curves, axis=0)[i], 10), i))])
            for name, curves in best.items()}
    out = args.run / "forecasts" / mode / f"f{fold}-{role}" / args.lane
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=out.parent, prefix=".fixed-panel-refit-") as tmp:
        stage = Path(tmp)
        payload = stage / "forecasts.npz"
        np.savez_compressed(payload, **{name: gain[name] * pred for name, pred in preds.items()})
        write_json(stage / "receipt.json", dict(stamp, fold=fold, role=role, mode=mode, lane=args.lane,
                   forecast_sha256=sha256(payload), model=model, gains=gain, support=info))
        stage.rename(out)


def collect(args, ctx):
    parts, _, genes, _, stamp = ctx
    cases, inventory, settings = {}, {}, []
    for mode in MODES:
        for fold, role in CASES:
            projected = projection(args, parts, fold, "held")
            query, _, _ = cohort(projected)
            key = f"{mode}/f{fold}-{role}"
            base = args.run / "forecasts" / key
            arms = {"B0": np.zeros((len(query), len(genes)))}
            for lane, names in (("cpu", CPU), ("state", STATE)):
                loc = base / lane
                registered = loc / "receipt.json"
                record = read_json(registered)
                if (any(record[k] != v for k, v in stamp.items()) or
                        (record["mode"], record["fold"], record["role"], record["lane"]) != (mode, fold, role, lane)):
                    raise ValueError("The frozen case provenance disagrees")
                file = require_hash(loc / "forecasts.npz", record["forecast_sha256"])
                with np.load(file, allow_pickle=False) as bundle:
                    if set(bundle.files) != set(names):
                        raise ValueError("Forecasted models differ from the prespecified arms")
                    arms.update({a: bundle[a] for a in names})
                inventory[f"{key}/{lane}"] = sha256(registered)
                settings.extend(dict(mode=mode, fold=fold, role=role, arm=name, gain=record["gains"][name],
                    fit_parents=record["support"]["outer_fit"], held_scaffolds_in_fit=record["support"]["held_scaffolds_in_fit"],
                    inner0_fit=record["support"]["inner0"]["fit"], inner1_fit=record["support"]["inner1"]["fit"],
                    close_parents=record["model"].get("tanimoto_0_7_parents")) for name in names)
            if set(arms) != set(ARMS) or any(v.shape != arms["B0"].shape or not np.isfinite(v).all() for v in arms.values()):
                raise ValueError("At least one frozen forecast fails the well-gene contract")
            cases[mode, fold, role] = (projected, query, arms)
    return cases, settings, dict(stamp, cases=inventory)


def report_points(books):
    points, intervals = [], []
    for (mode, unit), book in books.items():
        table, draws = book.results()
        table = table.loc[table.stratum.eq(TARGET)].assign(fit_scheme=mode, unit=unit)
        if not table.parent_groups.eq(203).all() or not table.rows_roles.eq(550).all():
            raise ValueError("The fixed reference lost part of the original 203-parent, 275-well support")
        table[["sse", "sst"]] *= 978 / 2000
        points.append(table)
        p = table.set_index("arm")
        pairs = [("hybrid", ref) for ref in ("B0", *CPU[1:], "State")] + [(a, "B0") for a in (*CPU[1:], "State")]
        for arm, ref in pairs:
            for metric in ENDPOINTS:
                value, baseline = p.at[arm, metric], p.at[ref, metric]
                lo, hi = np.quantile(draws[TARGET, arm, metric] - draws[TARGET, ref, metric], CI)
                improvement = 100*(baseline-value)/baseline if metric=="contrast_rmse" else 100*(value-baseline)
                intervals.append(dict(fit_scheme=mode, unit=unit, arm=arm, reference=ref, metric=metric,
                    value=value, baseline=baseline, difference=value-baseline, low99=lo, high99=hi,
                    improvement_percent_or_pp=improvement))
    return pd.concat(points, ignore_index=True), pd.DataFrame(intervals)


def finish(args, ctx):
    cases, settings, stamp = collect(args, ctx)
    freeze = args.run / "freeze.json"
    if args.phase == "freeze":
        if freeze.exists():
            raise FileExistsError(freeze)
        write_json(freeze, stamp)
        return
    if read_json(require_hash(freeze, args.digest)) != stamp:
        raise ValueError("Separate fixed-panel scoring requires the independent forecast freeze")
    parts, panel, _, blocks, _ = ctx
    books = {(mode, unit): StudyBook(sorted(blocks.index), DOSE, MINIMUM, "fixed44")
             for mode in MODES for unit in ("parent", "scaffold")}
    for mode in MODES:
        book = books[mode, "scaffold"]
        book.draw = scaffold_draw(book.index, blocks)
    for (mode, fold, role), (projected, query, arrays) in cases.items():
        other = "B" if role == "A" else "A"
        actual = observed(args, parts, panel, fold, other, "score", projected, query)
        for unit in ("parent", "scaffold"):
            books[mode, unit].add(query, actual, arrays)
    points, pairs = report_points(books)
    out = args.run / "score"
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=args.run, prefix=".fixed-panel-score-") as tmp:
        stage = Path(tmp)
        points.to_csv(stage / "points.tsv", sep="\t", index=False)
        pairs.to_csv(stage / "paired.tsv", sep="\t", index=False)
        pd.DataFrame(settings).to_csv(stage / "settings.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(stamp, freeze_sha256=args.digest,
                   files={file.name: sha256(file) for file in stage.iterdir()}))
        stage.rename(out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("predict", "freeze", "score"))
    for field in ("reports", "protocol", "previous", "panel-lock", "run", "features", "controls", "state", "cells"):
        cli.add_argument("--" + field, type=Path)
    cli.add_argument("--lane", choices=("cpu", "state"))
    cli.add_argument("--fold", type=int, choices=range(3))
    cli.add_argument("--role", choices="AB")
    cli.add_argument("--mode", choices=MODES)
    cli.add_argument("--digest")
    args = cli.parse_args()
    ctx = provenance(args)
    (predict if args.phase == "predict" else finish)(args, ctx)
