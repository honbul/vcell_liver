"""Test Tahoe endpoint repeatability before opening archived model predictions."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, LINE, table
from src.r13_optimization.objectives import center, row_products

from .ceilings import REPEATS, SEED, macro_weights
from .followup_b import PLATES, repeat_matches, repeat_vectors, score_pairs, summarize
from .followup_data import MIN_GROUPS, PROTOCOL_SHA, array, load_cache
from .rescore import HISTORICAL_ARMS, history, load_members
from .targets import metric_for


WELL = KEY[0]
ENDPOINTS = ("well", "strong", "hallmark", "rank16")
DESIGNS = ((0.5, "different_position"), (5.0, "different_position"), (5.0, "same_position_6_14"))
REGIMES = ("all", "old")


def hallmark_panel(gmt, symbols):
    available = defaultdict(list)
    for i, symbol in enumerate(symbols):
        available[str(symbol).upper()].append(i)
    kept, mapping, coverage = [], [], {}
    for name, genes in gmt.items():
        matched = sorted(genes & available.keys())
        if len(matched) >= 10:
            weights = np.zeros(len(symbols), dtype=np.float32)
            for gene in matched:
                indices = available[gene]
                weights[indices] = 1 / len(matched) / len(indices)
            kept.append(name)
            mapping.append(weights)
            coverage[name] = len(matched)
    if not kept:
        raise ValueError("No Hallmark set reaches 10 symbols in the specified panel")
    return np.stack(mapping, axis=1), coverage


def read_gmt(path):
    sets = {}
    with Path(path).open() as stream:
        for line in stream:
            name, _, *symbols = line.strip().split("\t")
            if name in sets or not name.startswith("HALLMARK_") or not symbols:
                raise ValueError(f"Unexpected Hallmark record: {name}")
            sets[name] = set(symbols)
    if len(sets) != 50:
        raise ValueError("Expected the 50 human Hallmark gene sets")
    return sets


def well_repeats(rows, values, matches):
    """Each physical well pair uses exactly the intersection of its cell lines."""
    links = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, donors in matches:
        for j in donors:
            links[rows.at[i, WELL]][rows.at[j, "plate_id"]][rows.at[j, WELL]].append((i, j))
    target, donor, metadata, pairs = [], [], [], []
    weights = rows.n_cells.to_numpy()
    for well, plates in links.items():
        t, d, used, support = [], [], set(), []
        for by_well in plates.values():
            t_plate, d_plate = [], []
            for other_well, directions in by_well.items():
                source, replicate = np.asarray(directions).T
                t_plate.append(np.average(values[source], axis=0, weights=weights[source]))
                d_plate.append(np.average(values[replicate], axis=0, weights=weights[replicate]))
                used.update(source.tolist())
                support.append(tuple(sorted((well, other_well))))
            t.append(np.mean(t_plate, axis=0))
            d.append(np.mean(d_plate, axis=0))
        target.append(np.mean(t, axis=0))
        donor.append(np.mean(d, axis=0))
        pairs.append(support)
        row = rows.loc[min(used)].to_dict()
        row[LINE] = "shared_well"
        row["n_cells"] = int(weights[sorted(used)].sum())
        metadata.append(row)
    out_rows = pd.DataFrame(metadata, columns=rows.columns)
    n = values.shape[1]
    return out_rows, np.array(target).reshape(-1, n), np.array(donor).reshape(-1, n), pairs


def repeat_scores(folder, sets):
    frame = table(folder / "rows.tsv")
    genes = table(folder / "panel.tsv")
    hall, coverage = hallmark_panel(sets, genes.symbol.to_numpy())
    basis = array(folder, "basis")
    if basis.shape != (len(genes), 16):
        raise ValueError(f"Unexpected rank basis dimensions in {folder}")
    result = {}
    for dose, design in DESIGNS:
        use = np.flatnonzero(frame.dose_uM.eq(dose))
        rows = frame.iloc[use].reset_index(drop=True)
        book = rows[[WELL, "barcode_position"]].drop_duplicates().set_index(WELL, verify_integrity=True)
        same = design == "same_position_6_14"
        matches = repeat_matches(rows, book.barcode_position, same_position=same, plate_pair=PLATES if same else None)
        for correction in REGIMES:
            actual = array(folder, f"target_{correction}")[use]
            for endpoint in ENDPOINTS:
                if endpoint == "well":
                    examples = well_repeats(rows, actual, matches)
                elif endpoint == "strong":
                    strong = [(i, js) for i, js in matches if rows.at[i, "strength"] == "strong"]
                    examples = repeat_vectors(rows, actual, strong)
                else:
                    examples = repeat_vectors(rows, actual @ (hall if endpoint == "hallmark" else basis), matches)
                result[endpoint, correction, dose, design] = score_pairs(*examples)
    return result, coverage


def passes(result):
    interval = result.get("gain_025_reduction_ci95")
    return result["design"] == "different_position" and result["groups"] >= 10 and interval is not None and interval[0] > 1.0


def ceilings(args, sets):
    by_slice = defaultdict(list)
    coverage_by_fold = {}
    for fold in range(3):
        scores, coverage = repeat_scores(args.cache / f"fold{fold}", sets)
        coverage_by_fold[str(fold)] = coverage
        for key, result in scores.items():
            by_slice[key].append(result)
        print(f"completed repeat-based endpoint evaluation for fold {fold}", flush=True)
    summary = []
    for (endpoint, correction, dose, design), slices in sorted(by_slice.items()):
        result = dict(endpoint=endpoint, correction=correction, dose_uM=dose, design=design,
                      **summarize(pd.concat(slices, ignore_index=True), True))
        if design == "different_position":
            result["gate_passed"] = passes(result)
        summary.append(result)
    candidates = sorted({r["endpoint"] for r in summary if r["correction"] == "all" and r["dose_uM"] == 0.5 and r.get("gate_passed")})
    unsupported = sorted({r["endpoint"] for r in summary if r["correction"] == "all" and r["dose_uM"] == 5 and r.get("gate_passed")})
    return dict(protocol_sha256=PROTOCOL_SHA, cache_sha256=sha256(args.cache / "receipt.json"), gmt_sha256=sha256(args.gmt),
                code_sha256=sha256(__file__), hallmark_symbols=coverage_by_fold, results=summary,
                eligible_frozen_regrade=candidates, passed_at_5_without_frozen_predictions=unsupported,
                gate=dict(minimum_groups=10, gain=0.25, lower_ci_exclusive_pct=1.0, primary_correction="all"),
                weighting="group bootstrap conditional on fixed parent pool centers and matched wells")


def well_values(rows, values):
    blocks = list(rows.groupby(WELL, sort=False).indices.values())
    aggregate = np.stack([np.average(values[ix], axis=0, weights=rows.n_cells.to_numpy()[ix]) for ix in blocks])
    kept = rows.iloc[[ix[0] for ix in blocks]].reset_index(drop=True).copy()
    kept[LINE] = "shared_well"
    kept["n_cells"] = [int(rows.n_cells.iloc[ix].sum()) for ix in blocks]
    return kept, aggregate


def recast(rows, values, name, hallmark, basis):
    if name == "well":
        return well_values(rows, values)
    if name == "strong":
        strong = rows.strength.eq("strong").to_numpy()
        return rows.loc[strong].reset_index(drop=True), values[strong]
    return rows, values @ (hallmark if name == "hallmark" else basis)


def fold_regrade(args, sets, names, task, prev):
    folder = args.cache / f"fold{task // 2}"
    frame = table(folder / "rows.tsv")
    q = frame.dose_uM.eq(0.5).to_numpy()
    frame = frame.loc[q].reset_index(drop=True)
    archived_path = args.previous / f"task{task}" / "rows.tsv"
    if sha256(archived_path) != prev["files"][f"task{task}/rows.tsv"]:
        raise ValueError(f"Archived scored cohort changed: task {task}")
    historical = table(archived_path)
    order = pd.MultiIndex.from_frame(historical[KEY]).get_indexer(pd.MultiIndex.from_frame(frame[KEY]))
    if np.any(order < 0) or not np.array_equal(historical[GROUP].to_numpy()[order], frame[GROUP]):
        raise ValueError(f"Historical score cohort mismatch: task {task}")
    known, rejected, orig_panel = history(args, task, historical)
    if rejected or len(known) != sum(map(len, HISTORICAL_ARMS.values())):
        raise ValueError(f"Frozen predictions do not cover all seven declared arms: {rejected}")
    source = table(folder / "panel.tsv")
    overlap = array(folder, "overlap")
    historic_select = pd.Index(orig_panel).get_indexer(source.token.to_numpy()[overlap])
    if np.any(historic_select < 0):
        raise ValueError("Archived predictions are missing a reported common token")
    hallmark, landmark = hallmark_panel(sets, source.symbol.to_numpy()[overlap])
    basis = array(folder, "basis_overlap")
    actual = array(folder, "target_all")[np.ix_(q, overlap)]
    metric = metric_for(frame, MIN_GROUPS, True)
    frozen = {arm: [center(metric, a[np.ix_(order, historic_select)]) for a in load_members(member)]
              for arm, member in known.items()}
    out = {}
    for name in names:
        rows, response = recast(frame, actual, name, hallmark, basis)
        if rows.empty:
            continue
        group_weight = macro_weights(rows)
        raw = pd.DataFrame({GROUP: rows[GROUP], "B0": np.sqrt(row_products(response, response))})
        for arm, members in frozen.items():
            errors = []
            for a in members:
                _, vector = recast(frame, a, name, hallmark, basis)
                err = response - vector
                errors.append(np.sqrt(row_products(err, err)))
            raw[arm] = np.mean(errors, axis=0)
        cols = raw.columns.drop(GROUP)
        groups = raw[cols].multiply(group_weight, axis=0).groupby(raw[GROUP]).sum()
        out[name] = (groups, len(rows), response.shape[1], len(overlap))
    return out, landmark


def regrade(args, sets, names):
    if not names:
        return {"status": "gate_not_met", "evaluated": []}
    prev = read_json(args.previous / "receipt.json")
    aggr = defaultdict(list)
    hall_coverage = {}
    for task in range(6):
        blocks, landmark = fold_regrade(args, sets, names, task, prev)
        for name, data in blocks.items():
            aggr[name].append(data)
        hall_coverage[str(task)] = landmark
        print(f"regraded eligible endpoint(s) in role-fold task {task}", flush=True)
    results = []
    for name, blocks in aggr.items():
        by_group = pd.concat([b[0] for b in blocks]).groupby(level=0).mean()
        point = by_group.mean()
        draw = np.random.default_rng(SEED).integers(len(by_group), size=(REPEATS, len(by_group)))
        replicate = by_group.to_numpy()[draw].mean(axis=1)
        for i, arm in enumerate(by_group.columns):
            delta = replicate[:, i] - replicate[:, 0]
            results.append(dict(endpoint=name, arm=arm, rmse=float(point[arm]), delta_vs_B0=float(point[arm] - point.B0),
                                delta_ci95=np.quantile(delta, [0.025, 0.975]).tolist(), groups=len(by_group),
                                rows_counted_by_role=sum(b[1] for b in blocks), dimensions=[min(b[2] for b in blocks), max(b[2] for b in blocks)],
                                panel_intersection=[min(b[3] for b in blocks), max(b[3] for b in blocks)]))
    return {"status": "completed", "evaluated": results, "hallmark_symbols_overlap": hall_coverage,
            "weighting": "both DMSO roles, fixed parent-pool centers, clustered by compound group, historic seeds averaged as RMSE"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "protocol", "out", "r12", "r13", "previous", "gmt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    load_cache(args.cache, args.protocol)
    sets = read_gmt(args.gmt)
    result = ceilings(args, sets)
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out / "repeat_ceilings.json"
    write_json(output, result)
    names = result["eligible_frozen_regrade"]
    write_json(args.out / "conditional_regrade.json", dict(
        repeat_sha256=sha256(output), protocol_sha256=PROTOCOL_SHA, **regrade(args, sets, names),
    ))


if __name__ == "__main__":
    main()
