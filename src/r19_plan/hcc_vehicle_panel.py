"""Lock Tahoe vehicle support and diagnose response-role differences on a fixed panel."""

import argparse
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.goal_tracks.t1_data import HIGH_PLATES, MIN_GROUPS, WELL, eligibility
from src.p4.common import WellProjection
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, LINE, group_weights, table
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.objectives import center, row_products
from src.r19_plan.hcc_scaffold import LINE_ID
from src.r19_plan.hcc_scaffold_refit import require_hash


MODES = ("fit_10", "common_all_wells")


def receipts(args):
    pin = args.reports / "r19_plan/data/hcc_scaffold_refit"
    parts = {}
    for name in ("visible", "score"):
        path = args.previous / name / "receipt.json"
        parts[name] = read_json(require_hash(path, sha256(pin / name / "receipt.json")))
    if any(x["genes"] != 978 for x in parts.values()) or parts["visible"]["source_counts_sha256"] != parts["score"]["source_counts_sha256"]:
        raise ValueError("Tahoe cache axis or source differs from the registered study")
    stamp = dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
                 visible_sha256=sha256(pin / "visible/receipt.json"), score_pin_sha256=sha256(pin / "score/receipt.json"))
    return parts, stamp


def old_rows(args, visible, name):
    data = table(require_hash(args.previous / "visible" / name, visible["files"][name]))
    if data.quarantined.any() or not data.plate_id.isin(HIGH_PLATES).all() or data.plate_id.eq("plate13").any():
        raise ValueError("A reserved or out-of-scope Tahoe row entered")
    return data


def control_support(args, source_pin):
    r11 = read_json(args.reports / "r11_chemical_probe/manifest/receipt.json")
    raw = require_hash(args.metadata, r11["inputs"]["metadata/sample_metadata.parquet"]["sha256"])
    metadata = pq.read_table(raw, columns=["sample", "plate", "drug", "drugname_drugconc"]).to_pandas()
    metadata = metadata.loc[metadata.plate.isin(HIGH_PLATES)]
    all_controls = metadata.loc[metadata.drug.eq("DMSO_TF")]
    if not all(literal_eval(value) == [("DMSO_TF", 0., "uM")] for value in all_controls.drugname_drugconc):
        raise ValueError("The exact vehicle label contains an active treatment")
    solvate = metadata.loc[metadata.drug.str.contains("DMSO", case=False, na=False) & ~metadata.drug.eq("DMSO_TF")]
    stamp = read_json(require_hash(args.controls / "receipt.json", source_pin))
    rows = []
    for role in "AB":
        name = f"{role}_rows.tsv"
        data = table(require_hash(args.controls / name, stamp["files"][name]))
        data = data.loc[data.plate_id.isin(HIGH_PLATES)]
        if data.quarantined.any() or not data.control_role.eq(role).all():
            raise ValueError("The physical A/B metadata disagree with the legacy roles")
        rows.append(data[["plate_id", WELL, "control_role"]].drop_duplicates())
    roles = pd.concat(rows, ignore_index=True)
    if (all_controls.duplicated("sample").any() or roles.duplicated(WELL).any()
            or set(zip(all_controls.plate, all_controls['sample'])) != set(zip(roles.plate_id, roles[WELL]))):
        raise ValueError("Raw DMSO catalogue and the archived roles do not cover the same physical wells")
    summary = roles.groupby("plate_id").agg(wells=(WELL, "nunique"), roles=("control_role", "nunique")).reset_index()
    total = solvate.groupby("plate").size()
    return summary.assign(three_distinct_roles=summary.wells.ge(3), nonvehicle_dmso_label=summary.plate_id.map(total).fillna(0).astype(int))


def line_sets(projection):
    original = projection.original
    active = projection.rows.loc[projection.rows[LINE].eq(LINE_ID)]
    result = {}
    for plate, rows in active.groupby("plate_id"):
        result[plate] = {well: set(original.iloc[projection.wells[well]][LINE]) for well in sorted(set(rows[WELL]))}
    return result


def greedy_panel(fit):
    if not fit:
        return []
    panel, compatible = {LINE_ID}, list(fit.values())
    while len(panel) < 10:
        counts = pd.Series([line for lines in compatible for line in lines - panel], dtype=str).value_counts()
        if counts.empty:
            return []
        chosen = min(counts.index, key=lambda line: (-counts[line], line))
        panel.add(chosen)
        compatible = [lines for lines in compatible if chosen in lines]
    return sorted(panel)


def cohort(projection):
    keep = projection.rows[LINE].eq(LINE_ID).to_numpy()
    return projection.rows.loc[keep].reset_index(drop=True), projection.index[keep], keep


def panel_plan(fit, held):
    a, b = line_sets(fit), line_sets(held)
    query, _, _ = cohort(held)
    fit_target, _, _ = cohort(fit)
    panels = {mode: {} for mode in MODES}
    support = []
    for plate, query_wells in b.items():
        fit_wells = a.get(plate, {})
        common = sorted(set.intersection(*fit_wells.values(), *query_wells.values())) if fit_wells else []
        for mode, panel in (("fit_10", greedy_panel(fit_wells)), ("common_all_wells", common)):
            panel_set = set(panel)
            fit_present = {w for w, lines in fit_wells.items() if panel_set <= lines} if len(panel) >= 10 else set()
            held_present = {w for w, lines in query_wells.items() if panel_set <= lines} if len(panel) >= 10 else set()
            fit_parents = fit_target.loc[fit_target[WELL].isin(fit_present), GROUP].nunique()
            score_parents = query.loc[query[WELL].isin(held_present), GROUP].nunique()
            supported = min(fit_parents, score_parents) >= MIN_GROUPS
            panels[mode][plate] = dict(lines=panel, wells=sorted(held_present) if supported else [])
            support.append(dict(scope=mode, plate=plate, fixed_lines=len(panel), fit_total_wells=len(fit_wells),
                                score_total_wells=len(query_wells), fit_parents=fit_parents, score_parents=score_parents,
                                score_eligible_wells=len(held_present) if supported else 0))
    return panels, support


def lock(args):
    parts, stamp = receipts(args)
    measured = control_support(args, parts["visible"]["source_counts_sha256"])
    planned, records = {}, []
    for fold in range(3):
        fit, held = (WellProjection(old_rows(args, parts["visible"], f"f{fold}-{kind}.tsv")) for kind in ("fit", "held"))
        plan, summary = panel_plan(fit, held)
        planned[str(fold)] = plan
        records.extend(dict(fold=fold, **row) for row in summary)
    args.lock.mkdir(exist_ok=False)
    measured.to_csv(args.lock / "controls.tsv", sep="\t", index=False)
    pd.DataFrame(records).to_csv(args.lock / "pool_support.tsv", sep="\t", index=False)
    write_json(args.lock / "panel_manifest.json", planned)
    write_json(args.lock / "receipt.json", dict(stamp, metadata_only=True,
        files={p.name: sha256(p) for p in args.lock.iterdir()}))


def fixed_vectors(projection, selected, panels, delta):
    outcome = []
    for plate, well in selected[["plate_id", WELL]].itertuples(index=False, name=None):
        panel = panels[plate]["lines"]
        idx = np.asarray(projection.wells[well])
        layout = projection.original.iloc[idx][LINE]
        block = layout.isin(panel).to_numpy()
        target = layout.eq(LINE_ID).to_numpy()
        if len(panel) != int(block.sum()) or target.sum() != 1 or len(panel) < 10:
            raise ValueError("The locked panel is unavailable on a scored physical well")
        outcome.append(delta[idx[target][0]] - delta[idx[block]].astype(np.float64).mean(axis=0))
    return np.stack(outcome)


def artifacts(rows, a, b, *, projected, fixed=False):
    metric = Contrast(rows, MIN_GROUPS)
    if len(metric.active) != len(rows) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("A diagnostic row is outside the original scoring support")
    gap = a - b
    target = (a + b) / 2
    if projected:
        gap, target = center(metric, gap), center(metric, target)
    if fixed and np.max(np.abs(gap)) > 1e-6:
        raise ValueError("A fixed-panel vehicle gap is not numerically invariant")
    return rows[[GROUP, ENTITY, LINE, WELL]].copy().assign(gap_energy=row_products(gap, gap),
        target_energy=row_products(target, target), max_abs_gap=np.max(np.abs(gap), axis=1))


def aggregate(rows):
    groups = len(set(rows[GROUP]))
    weights = group_weights(rows)
    moments = rows[[GROUP]].assign(gap=weights*rows.gap_energy, target=weights*rows.target_energy).groupby(GROUP).sum()
    value = moments.to_numpy()
    rms_gap, rms_target = np.sqrt(value.mean(axis=0))
    samples = np.random.default_rng(150104).integers(groups, size=(10000, groups))
    draws = value[samples].mean(axis=1)
    ratio = np.sqrt(draws[:, 0] / draws[:, 1])
    lo, hi = np.quantile(ratio, (.005, .995))
    return dict(parents=groups, wells=rows[WELL].nunique(), rms_gap=rms_gap, rms_target=rms_target,
                relative_rms=rms_gap/rms_target, relative_low99=lo, relative_high99=hi,
                max_abs_gap=rows.max_abs_gap.max())


def diagnose(args):
    parts, stamp = receipts(args)
    pin = args.lock / "receipt.json"
    lock = read_json(require_hash(pin, args.digest))
    if not lock["metadata_only"] or any(stamp[k] != lock[k] for k in stamp):
        raise ValueError("The panel support freeze has unexpected provenance")
    plans = read_json(require_hash(args.lock / "panel_manifest.json", lock["files"]["panel_manifest.json"]))
    rows = []
    for fold in range(3):
        projection = WellProjection(old_rows(args, parts["visible"], f"f{fold}-held.tsv"))
        query, idx, keep = cohort(projection)
        names = [f"f{fold}-{role}.npy" for role in "AB"]
        deltas = [np.load(require_hash(args.previous / "score" / n, parts["score"]["files"][n]), mmap_mode="r", allow_pickle=False)
                  for n in names]
        if any(x.shape != (len(projection.original), 978) for x in deltas):
            raise ValueError("Vehicle roles have different rows or landmarks")
        raw = [x[idx] for x in deltas]
        original = [projection.centered(x)[keep] for x in deltas]
        panes = [("raw", query, raw, False), ("raw_pool_centered", query, raw, True), ("original_P4", query, original, True)]
        for mode in MODES:
            panel = plans[str(fold)][mode]
            wells = {well for item in panel.values() for well in item["wells"]}
            selected = eligibility(query.assign(index=np.arange(len(query))).loc[query[WELL].isin(wells)], 0)
            which = selected.pop("index").to_numpy(dtype=int)
            if len(selected):
                panes.extend((("original_on_" + mode, selected, [x[which] for x in original], True),
                              (mode, selected, [fixed_vectors(projection, selected, panel, x) for x in deltas], True)))
        rows.extend(artifacts(frame, *values, projected=ctr, fixed=name in MODES).assign(fold=fold, scope=name)
                    for name, frame, values, ctr in panes)
    combined = pd.concat(rows, ignore_index=True)
    full = combined.loc[combined.scope.eq("original_P4")]
    if (full[GROUP].nunique(), full[WELL].nunique()) != (203, 275):
        raise ValueError("The original SNU-423 physical cohort changed")
    output = [dict(scope=scope, fold=fold, **aggregate(frame)) for (scope, fold), frame in combined.groupby(["scope", "fold"])]
    for scope, frame in combined.groupby("scope"):
        output.append(dict(scope=scope, fold="all", **aggregate(frame)))
    args.out.mkdir(exist_ok=False)
    pd.DataFrame(output).to_csv(args.out / "role_gap.tsv", sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(stamp, lock_sha256=args.digest, scored_diagnostic=True,
                   files={p.name: sha256(p) for p in args.out.iterdir()}))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("lock", "diagnose"))
    for field in ("reports", "previous", "protocol", "metadata", "controls", "lock", "out"):
        cli.add_argument("--" + field, type=Path)
    cli.add_argument("--digest")
    args = cli.parse_args()
    (lock if args.action == "lock" else diagnose)(args)
