"""Compare same-position and different-position Tahoe replicates at 5 µM."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, LINE, checked_table, group_weights, load_data, table
from src.r13_optimization.objectives import center, row_products

from .ceilings import REPEATS, SEED, eligible, macro_weights
from .study import position_predict, positions
from .targets import GAINS, metric_for


WELL = KEY[0]
PLATES = ("plate6", "plate14")
GRID = [f"grid{i}" for i in range(len(GAINS))]


def repeat_matches(rows, labels, same_position=False, plate_pair=None):
    """Find same-group, same-line anchor-to-donor matches across physical plates."""
    matches = []
    plates = rows.plate_id.to_numpy()
    locations = labels.loc[rows[WELL]].to_numpy()
    lookup = defaultdict(list)
    for i, key in enumerate(zip(rows[GROUP], rows[LINE])):
        if plate_pair is None or plates[i] in plate_pair:
            lookup[key].append(i)
    for members in lookup.values():
        for i in members:
            donors = [j for j in members if plates[i] != plates[j] and (locations[i] == locations[j]) == same_position]
            if donors:
                matches.append((i, donors))
    return matches


def repeat_vectors(rows, values, matches):
    """Keep anchor order and average all donor wells by first balancing donor plates."""
    plates = rows.plate_id.to_numpy()
    well_ids = rows[WELL].to_numpy()
    anchors, references, pairs = [], [], []
    for i, donors in matches:
        by_plate = defaultdict(list)
        for j in donors:
            by_plate[plates[j]].append(j)
        anchors.append(i)
        references.append(np.stack([values[indices].mean(axis=0) for indices in by_plate.values()]).mean(axis=0))
        pairs.append([tuple(sorted((well_ids[i], well_ids[j]))) for j in donors])
    return rows.iloc[anchors].reset_index(drop=True), values[anchors], np.asarray(references).reshape(-1, values.shape[1]), pairs


def score_pairs(rows, observed, predicted, pairs=None):
    actual_var, donor_var = row_products(observed, observed), row_products(predicted, predicted)
    agreement = row_products(observed, predicted)
    scores = rows.copy()
    denom = np.sqrt(actual_var * donor_var)
    scores["cosine"] = np.divide(agreement, denom, out=np.zeros(len(rows)), where=denom > 1e-14)
    for name, gain in zip(GRID, GAINS):
        scores[name] = np.sqrt(np.maximum(actual_var + gain**2 * donor_var - 2 * gain * agreement, 0))
    if pairs is not None:
        scores["pairs"] = pairs
    return scores


def group_scores(scores, weighted, columns):
    weights = macro_weights(scores) if weighted else group_weights(scores)
    return scores[columns].multiply(weights, axis=0).groupby(scores[GROUP], sort=True).sum()


def crossfit_gains(scores, weighted):
    selections = {}
    scores = scores.copy()
    scores["selected_gain_rmse"] = np.nan
    for fold in sorted(scores.outer_fold.unique()):
        fit = scores.loc[scores.outer_fold.ne(fold)]
        if not fit.empty:
            opt = int(np.argmin(group_scores(fit, weighted, GRID).mean().to_numpy()))
            selections[str(fold)] = float(GAINS[opt])
            query = scores.outer_fold.eq(fold)
            scores.loc[query, "selected_gain_rmse"] = scores.loc[query, GRID[opt]]
    return scores, selections


def summarize(scores, weighted):
    report = dict(groups=int(scores[GROUP].nunique()), rows=len(scores),
                  well_pairs=len({pair for links in scores.pairs for pair in links}))
    if scores.empty:
        return report
    cols = {"b0": GRID[0], "gain_025": GRID[2], "gain_1": GRID[-1], "cosine": "cosine"}
    if "selected_gain_rmse" in scores and scores.selected_gain_rmse.notna().all():
        cols["gain_crossfit"] = "selected_gain_rmse"
    components = group_scores(scores, weighted, list(cols.values())).rename(columns={v: k for k, v in cols.items()})
    draw = np.random.default_rng(SEED).integers(len(components), size=(REPEATS, len(components)))
    sample = components.to_numpy()[draw].mean(axis=1)
    point = components.mean()
    for k, name in enumerate(components):
        report[name] = float(point[name])
        report[f"{name}_ci95"] = np.quantile(sample[:, k], [0.025, 0.975]).tolist()
        if name.startswith("gain") and point.b0 > 0 and np.all(sample[:, 0] > 0):
            report[f"{name}_reduction_pct"] = float(100 * (1 - point[name] / point.b0))
            report[f"{name}_reduction_ci95"] = np.quantile(100 * (1 - sample[:, k] / sample[:, 0]), [0.025, 0.975]).tolist()
    return report


def cosine_gap(same, different, weighted):
    joint = group_scores(same, weighted, ["cosine"]).join(
        group_scores(different, weighted, ["cosine"]), how="inner", lsuffix="_same", rsuffix="_different")
    if joint.empty:
        return None
    delta = (joint.cosine_same - joint.cosine_different).to_numpy()
    draw = np.random.default_rng(SEED).integers(len(delta), size=(REPEATS, len(delta)))
    return dict(groups=len(delta), same_minus_different=float(delta.mean()),
                ci95=np.quantile(delta[draw].mean(axis=1), [0.025, 0.975]).tolist())


def cohort_design(metadata, book, group_fold):
    picked = metadata.loc[metadata.plate_id.isin(PLATES) & metadata.dose_uM.eq(5)]
    wells = picked[[WELL, "plate_id", GROUP]].drop_duplicates().copy()
    wells["position"] = wells[WELL].map(book.barcode_position)
    left = wells.loc[wells.plate_id.eq(PLATES[0])].set_index("position", verify_integrity=True)
    right = wells.loc[wells.plate_id.eq(PLATES[1])].set_index("position", verify_integrity=True)
    joint = left.join(right, lsuffix="_left", rsuffix="_right", how="outer")
    same = joint[f"{GROUP}_left"].eq(joint[f"{GROUP}_right"])
    in_fold = joint[f"{GROUP}_left"].isin(group_fold)
    return dict(plate_wells={PLATES[0]: len(left), PLATES[1]: len(right)}, matched_pairs=int(same.sum()),
                matched_groups=int(joint.loc[same, f"{GROUP}_left"].nunique()),
                matched_pairs_in_frozen_cohort=int((same & in_fold).sum()),
                unmatched_or_mismatched_positions=[int(x) for x in joint.index[~same]])


def checked_common(args):
    freeze_path = args.common / "position_freeze.json"
    receipt_path = args.common / "receipt.json"
    freeze, receipt = read_json(freeze_path), read_json(receipt_path)
    if (receipt["position_freeze_sha256"] != sha256(freeze_path)
            or freeze["protocol_sha256"] != sha256(args.prior_protocol)
            or freeze["coordinate_sha256"] != sha256(args.coordinates / "receipt.json")):
        raise ValueError("Position artifacts differ from the previous freeze")
    for name, digest in freeze["files"].items():
        if sha256(args.common / name) != digest:
            raise ValueError(f"Frozen common file changed: {name}")
    for fold in range(3):
        row_name = f"task{2 * fold}/rows.tsv"
        if sha256(args.common / row_name) != receipt["files"][row_name]:
            raise ValueError(f"Previous scoring cohort no longer matches: fold {fold}")


def run(args):
    checked_common(args)
    minimum = config(args.config)["minimum_groups"]
    book = positions(args.coordinates)
    source = args.assets / "doses/source"
    group_fold = {}
    panels = []
    for fold in range(3):
        query = table(args.common / f"task{2 * fold}/rows.tsv")
        if set(query[GROUP]) & set(group_fold):
            raise ValueError("A chemical group enters multiple outer folds")
        group_fold.update(dict.fromkeys(query[GROUP], fold))
        path = args.common / f"fold{fold}"
        panels.extend(np.load(path / f"panel{suffix}.npy", allow_pickle=False) for suffix in ("", "_filtered"))
    columns = np.unique(np.concatenate(panels))
    design = cohort_design(checked_table(source, "treated_rows.tsv"), book, group_fold)
    rows, count = load_data(source, "treated", columns,
                            mask=lambda meta: meta.dose_uM.eq(5).to_numpy() & meta[GROUP].isin(group_fold).to_numpy())
    if not np.array_equal(rows.plate_id.to_numpy(), book.loc[rows[WELL], "plate"].to_numpy()):
        raise ValueError("Dose cache and barcode release plate assignments differ")
    rows["outer_fold"] = rows[GROUP].map(group_fold)
    all_results = defaultdict(list)
    attendance = []
    for fold in range(3):
        fold_mask = rows.outer_fold.eq(fold).to_numpy()
        fold_rows = rows.loc[fold_mask].reset_index(drop=True)
        fold_count = count[fold_mask]
        for weighted, name in ((False, "original"), (True, "combined")):
            path = args.common / f"fold{fold}"
            panel = panels[2 * fold + int(weighted)]
            keep_cells = fold_rows.n_cells.ge(200) if weighted else fold_rows.n_cells.notna()
            idx = eligible(fold_rows, minimum, keep_cells)
            query = fold_rows.iloc[idx].reset_index(drop=True)
            observed = fold_count[idx][:, np.searchsorted(columns, panel)].toarray()
            support = np.zeros(len(query), dtype=int)
            if weighted:
                donors = table(path / "donors.tsv")
                reference = np.load(path / "donors_filtered.npy", allow_pickle=False)
                p, support = position_predict(donors, reference, query, book.barcode_position)
                observed -= read_json(path / "receipt.json")["scale_filtered"] * p
            observed = center(metric_for(query, minimum, weighted), observed)
            same = repeat_matches(query, book.barcode_position, same_position=True, plate_pair=PLATES)
            different = repeat_matches(query, book.barcode_position)
            for design_name, matches in (("same_position_6_14", same), ("different_position", different)):
                all_results[name, design_name].append(score_pairs(*repeat_vectors(query, observed, matches)))
                anchors = np.array([i for i, _ in matches], dtype=int)
                attendance.append(dict(target=name, fold=fold, design=design_name, eligible_rows=len(query),
                                       matched_rows=len(matches), matched_groups=int(query.iloc[anchors][GROUP].nunique()),
                                       position_supported=int(np.count_nonzero(support[anchors])) if weighted else None))
        print(f"completed plate comparison fold {fold}", flush=True)
    result = {}
    for weighted, target in ((False, "original"), (True, "combined")):
        same, same_gains = crossfit_gains(pd.concat(all_results[target, "same_position_6_14"], ignore_index=True), weighted)
        diff, diff_gains = crossfit_gains(pd.concat(all_results[target, "different_position"], ignore_index=True), weighted)
        common_groups = set(same[GROUP]) & set(diff[GROUP])
        same_common, diff_common = same.loc[same[GROUP].isin(common_groups)], diff.loc[diff[GROUP].isin(common_groups)]
        shared_lines = pd.MultiIndex.from_frame(same_common[[GROUP, LINE]]).intersection(
            pd.MultiIndex.from_frame(diff_common[[GROUP, LINE]]))
        same_line = same_common.loc[pd.MultiIndex.from_frame(same_common[[GROUP, LINE]]).isin(shared_lines)]
        diff_line = diff_common.loc[pd.MultiIndex.from_frame(diff_common[[GROUP, LINE]]).isin(shared_lines)]
        result[target] = dict(same_position=summarize(same, weighted), different_position=summarize(diff, weighted),
                              crossfit_gains=dict(same_position=same_gains, different_position=diff_gains),
                              common_groups=dict(same_position=summarize(same_common, weighted),
                                                 different_position=summarize(diff_common, weighted),
                                                 cosine_gap=cosine_gap(same_common, diff_common, weighted)),
                              common_group_lines=dict(same_position=summarize(same_line, weighted),
                                                      different_position=summarize(diff_line, weighted),
                                                      cosine_gap=cosine_gap(same_line, diff_line, weighted)))
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "comparison.json", dict(
        protocol_sha256=sha256(args.protocol), prior_protocol_sha256=sha256(args.prior_protocol),
        common_receipt_sha256=sha256(args.common / "receipt.json"),
        source_receipt_sha256=sha256(source / "receipt.json"), code_sha256=sha256(__file__),
        design=design, attendance=attendance, results=result, gain_grid=GAINS.tolist(),
        interpretation="Fixed parent-pool centers; both directions kept together in a whole-parent-group bootstrap.",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "common", "coordinates", "config", "protocol", "prior-protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
