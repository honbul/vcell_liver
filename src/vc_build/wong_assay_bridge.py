"""Measure patient-paired 2D–3D response transfer and donor-disjoint calibration."""

import argparse
import json
from contextlib import closing
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .evaluate import join_observations
from .external import KEY
from .function_data import new_output
from .hcc_3d_fewshot import clustered_sensitivity
from .io import freeze_table, publish_json, sha256, verify_forecast
from .limore_corrected import shared_support
from .response_components import component_scores, interaction_intervals
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .v0plus_wong import WORKBOOK_HASH, auc_mean, read_registration, workbook
from .wong_expanded import ORIGINAL_WONG
from .wong_tissue import read_forecasts
from .yang_hcc_external import require_hash

DIRECT = "measured_2d"
NULL = "measured_2d_drug_mean"
PRIMARY = "fit_shrunk_slope"
METHODS = ("fit_3d_mean", "fit_unit_slope", "fit_global_slope", PRIMARY)
SCOPE = "WONG_PDTCL_TO_PDTO"
REFERENCES = {
    "organoid": ("interaction_only", "drug_ridge"),
    "tissue": ("interaction_only", "drug_ridge", "cellhit"),
}
FORECASTS = {
    "organoid": "9330a821ea14ab99bde652f7ff66e84879d92ef63bd9e97a3fb3b41e38af9965",
    "tissue": "0537484b825b84e43b0d91efe57c11ed114a18831009f7a6b6373664aa075b1c",
}


def measurements(args):
    require_hash(args.original / "receipt.json", ORIGINAL_WONG)
    read_registration(args.original)
    catalog = pd.read_parquet(args.original / "chemistry.parquet")
    previously = pd.read_parquet(args.original / "query.parquet")
    labels = previously[["compound_id", "source_compound_label"]].drop_duplicates().set_index("compound_id")
    catalog = catalog.assign(
        source_compound_label=catalog.compound_id.map(labels.source_compound_label).fillna(catalog.product_name),
        drug_target_source_labels=catalog.mechanism,
    )
    with closing(workbook(args.workbook)) as handle:
        remaining = iter(handle["Supplementary Table 7"].values)
        header, rows = next(remaining), list(remaining)
    pairs = {}
    for name in header:
        if str(name).startswith("PDTCL_"):
            pdo = str(name).replace("PDTCL_", "PDTO_", 1)
            if pdo not in set(previously.cell_id) or header.count(pdo) != 1 or header.count(name) != 1:
                raise ValueError("Unexpected paired 2D/3D human HCC identity")
            pairs[pdo] = name
    if len(pairs) != 8 or len(catalog) != 100 or not catalog.compound_id.is_unique:
        raise ValueError("Expected eight published adult HCC pairs and the 100-drug catalogue")

    row_records = [
        {"cell_id": pdo, "lineage_group": pdo, "compound_id": compound,
         "row_id": pdo + "|" + compound, "measured_2d": auc_mean(rows[location - 2][header.index(line)]),
         "endpoint_value": auc_mean(rows[location - 2][header.index(pdo)])}
        for pdo, line in sorted(pairs.items())
        for compound, location in catalog[["compound_id", "response_row"]].itertuples(index=False, name=None)
    ]
    panel = pd.DataFrame(row_records)
    counts = panel.dropna().groupby("compound_id").size()
    complete = set(counts[counts.eq(len(pairs))].index)
    if len(complete) < 80:
        raise ValueError("Fewer than 80 compounds have eight complete paired measurements")
    panel = panel[panel.compound_id.isin(complete)].sort_values(["cell_id", "compound_id"])
    roster = panel[KEY + ["measured_2d"]]
    truth = panel[KEY + ["endpoint_value"]]
    with new_output(args.out) as out:
        fit_folder = out / "predict"
        fit_folder.mkdir()
        truth.to_parquet(out / "score_only.parquet", index=False)
        roster.to_parquet(fit_folder / "roster.parquet", index=False)
        catalog[catalog.compound_id.isin(complete)].to_parquet(out / "catalogue.parquet", index=False)
        fit_splits, fit_hashes = {}, {}
        for number, donors in enumerate(combinations(sorted(pairs), 3)):
            path = fit_folder / f"fit-{number}.parquet"
            truth[truth.lineage_group.isin(donors)].to_parquet(path, index=False)
            fit_splits[str(number)] = list(donors)
            fit_hashes[path.name] = sha256(path)
        publish_json(fit_folder / "receipt.json", {
            "core": ORIGINAL_WONG, "workbook": WORKBOOK_HASH, "scope": SCOPE, "fit_groups": fit_splits,
            "files": fit_hashes | {"roster.parquet": sha256(fit_folder / "roster.parquet")},
        })
        publish_json(out / "receipt.json", {"core": ORIGINAL_WONG, "workbook": WORKBOOK_HASH, "scope": SCOPE,
            "matched_donors": len(pairs), "tested_drugs": len(catalog), "complete_drugs": len(complete),
            "measured_rows": len(panel), "calibration_folds": len(fit_splits),
            "files": {name: sha256(out / name) for name in
                      ("score_only.parquet", "catalogue.parquet", "predict/receipt.json")}})


def fit_transfer(x, y, held):
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(held).all()):
        raise ValueError("Paired calibration requires finite fit and query observations")
    if x.shape != y.shape or x.shape[1] != held.shape[1]:
        raise ValueError("Calibration inputs do not have the same drug axis")
    center_x, center_y = x.mean(axis=0), y.mean(axis=0)
    fit_x, fit_y = x - center_x, y - center_y
    variability = np.square(fit_x).sum(axis=0)
    covariance = (fit_x * fit_y).sum(axis=0)
    donor_scale = variability.mean() / len(x)
    if donor_scale <= 1e-10:
        raise ValueError("No gain is identifiable from zero 2D variation")
    shared_gain = covariance.sum() / variability.sum()
    pseudo_sum = 8 * donor_scale
    local_gain = (covariance + pseudo_sum * shared_gain) / (variability + pseudo_sum)
    held = held - center_x
    return (np.broadcast_to(center_y, held.shape), center_y + held, center_y + shared_gain * held,
            center_y + local_gain * held), float(shared_gain)


def predict(args):
    path = args.inputs / "receipt.json"
    receipt = json.loads(path.read_text())
    if (receipt["core"], receipt["workbook"], receipt["scope"], len(receipt["fit_groups"])) != (
        ORIGINAL_WONG, WORKBOOK_HASH, SCOPE, 56
    ):
        raise ValueError("Missing locked calibration partitions")
    roster_path = args.inputs / "roster.parquet"
    require_hash(roster_path, receipt["files"][roster_path.name])
    roster = pd.read_parquet(roster_path).sort_values(["cell_id", "compound_id"])
    matrix = roster.pivot(index="lineage_group", columns="compound_id", values=DIRECT)
    anchor = roster.compound_id.map(matrix.mean())
    predictions = [roster[KEY].assign(method=m, predicted_value=v, track="direct", fold=0)
                   for m, v in ((DIRECT, roster.measured_2d), (NULL, anchor))]
    gains = {}
    for number, fit in receipt["fit_groups"].items():
        fit_file = args.inputs / f"fit-{number}.parquet"
        require_hash(fit_file, receipt["files"][fit_file.name])
        fit_labels = pd.read_parquet(fit_file)
        if tuple(sorted(fit_labels.lineage_group.unique())) != tuple(fit) or len(fit) != 3:
            raise ValueError("Holdout patient in calibration labels")
        target = fit_labels.pivot(index="lineage_group", columns="compound_id", values="endpoint_value")
        held = matrix.index.difference(fit)
        array, gains[number] = fit_transfer(matrix.loc[fit].to_numpy(),
                             target.loc[fit, matrix.columns].to_numpy(), matrix.loc[held].to_numpy())
        block = roster[roster.lineage_group.isin(held)][KEY]
        predictions.extend(block.assign(method=m, track="calibration", fold=int(number), predicted_value=v.ravel())
                           for m, v in zip(METHODS, array))
    freeze_table(pd.concat(predictions, ignore_index=True), args.out, provenance={
        "predict_receipt_sha256": sha256(path), "scope": SCOPE, "calibration_global_gains": gains})


def centered(x, axis):
    return x - x.mean(axis=axis, keepdims=True)


def unit_columns(x):
    norm = np.linalg.norm(x, axis=0)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 1e-10)


def interaction(x):
    return centered(centered(x, 0), 1)


def paired_permutation(x, y, *, repeats=2400):
    """One whole patient permutation across every drug, including the correct pairing."""
    if x.shape != y.shape or x.shape[0] != 8:
        raise ValueError("Exact paired inference uses eight donors")
    nx, ny = (unit_columns(centered(rankdata(a, axis=0), 0)) for a in (x, y))
    eligible = np.ptp(y, axis=0) > 1e-10
    spearman = ny[:, eligible] @ nx[:, eligible].T / eligible.sum()
    xx, yy = (interaction(a) for a in (x, y))
    inner = (yy @ xx.T) / (np.linalg.norm(xx) * np.linalg.norm(yy))
    order = np.asarray(list(permutations(range(len(x)))), dtype=np.int8)
    rng = np.random.default_rng(9247)
    pdos = rng.integers(len(x), size=(repeats, len(x), 1))
    drugs = rng.integers(x.shape[1], size=(repeats, 1, x.shape[1]))
    ax, ay = [centered(centered(raw[pdos, drugs], 1), 2) for raw in (x, y)]
    norm = np.linalg.norm(ax, axis=(1, 2)) * np.linalg.norm(ay, axis=(1, 2))
    corr = np.divide((ax * ay).sum(axis=(1, 2)), norm,
                     out=np.full(repeats, np.nan), where=norm > 1e-10)
    out = {}
    for name, metric in (("within_drug_spearman", spearman), ("interaction_pearson", inner)):
        matched = float(np.trace(metric))
        shuffled = metric[np.arange(len(x)), order].sum(axis=1)
        out[name] = {"matched": matched, "permutation_mean": float(shuffled.mean()),
                     "permutation_ci95": np.quantile(shuffled, [.025, .975]).tolist(),
                     "one_sided_exact_p": float(np.mean(shuffled >= matched - 1e-12))}
    out["interaction_pearson"]["paired_bootstrap_ci95"] = np.nanquantile(corr, [.025, .975]).tolist()
    return out


def archived(args):
    tables, stamps, registered = read_forecasts(args)
    if {k: entry["forecast"] for k, entry in stamps.items()} != FORECASTS:
        raise ValueError("Benchmarks need the unchanged external forecasts")
    rows = []
    for view, methods in REFERENCES.items():
        selected = tables[view][tables[view].method.isin((*methods, "drug_mean"))].copy()
        selected.row_id = selected.cell_id + "|" + selected.compound_id
        selected.method = view + "/" + selected.method
        rows.append(selected.assign(track="direct"))
    return pd.concat(rows), set(registered.compound_id), set(registered.loc[~registered.previously_scored, "compound_id"])


def direct_score(frame, catalogue):
    pairs = [(DIRECT, ref) for ref in sorted(set(frame.method) - {DIRECT})]
    result, drugs, _ = axis_metrics(frame, pairs)
    crossed = {ref: crossed_patient_drug_ci(frame, DIRECT, ref) for _, ref in pairs}
    components = component_scores(frame, cell_axis="lineage_group")
    ir = {ref: interaction_intervals(frame, DIRECT, ref) for _, ref in pairs
          if components["scores"][ref]["interaction_pearson"] is not None}
    block = frame[frame.method.eq(DIRECT)]
    matrices = [block.pivot(index="lineage_group", columns="compound_id", values=key).to_numpy()
                for key in ("predicted_value", "endpoint_value")]
    paired = paired_permutation(*matrices)
    strata = {}
    for name, keys in compound_strata(catalogue).items():
        means = drugs[drugs.compound_id.isin(keys)].groupby("method")[["spearman", "pearson"]].mean().dropna()
        if len(means):
            strata[name] = {"drugs": len(keys), "scores": means.to_dict("index")}
    return {"support": len(block), "methods": result, "sensitivity_ci95": crossed, "components": components,
            "paired_permutation": paired, "interaction_ci95": ir, "strata": strata,
            "per_drug": drugs.astype(object).where(pd.notna(drugs), None).to_dict("records")}


def calibration_score(frame):
    pairs = [(PRIMARY, baseline) for baseline in METHODS[:-1]]
    rank, _, _ = axis_metrics(frame, pairs)
    # The sensitivity helper retains one draw per donor and drug across every overlapping split.
    intervals = clustered_sensitivity(frame, pairs)
    check = {row["baseline"]: row["ci95"] for row in intervals}
    rank_ci = {(row["baseline"], row["metric"]): row["ci95"] for row in rank["paired_contrasts"]}
    gate = {ref: all(check[ref][metric][0] > 0 for metric in ("spearman", "pearson"))
            and all(rank_ci[ref, metric][0] > 0 for metric in ("spearman", "ndcg_at_5"))
            for _, ref in pairs}
    return {"resample": "paired Bayesian donor/drug weights across 56 splits", "metrics": rank,
            "sensitivity_ci95": intervals, "gate": gate}


def score(args):
    path = args.inputs / "receipt.json"
    receipt = json.loads(path.read_text())
    for name, digest in receipt["files"].items():
        require_hash(args.inputs / name, digest)
    predicted, seal = verify_forecast(args.forecast)
    if seal["provenance"]["predict_receipt_sha256"] != receipt["files"]["predict/receipt.json"]:
        raise ValueError("Separate scoring requires the registered paired fit bundle")
    outcomes = pd.read_parquet(args.inputs / "score_only.parquet")
    catalogue = pd.read_parquet(args.inputs / "catalogue.parquet")
    reference, overlap, novel = archived(args)
    results = {}
    for name, ids in (("full", set(catalogue.compound_id)), ("rna_supported", overlap), ("rna_novel", novel)):
        experiment = predicted[predicted.track.eq("direct") & predicted.compound_id.isin(ids)]
        target = outcomes[outcomes.compound_id.isin(ids)]
        observed = experiment
        if name != "full":
            selected = reference[reference.row_id.isin(target.row_id)]
            shared_support(selected, target, reference.method.unique())
            observed = pd.concat([experiment, selected], ignore_index=True)
        results[name] = direct_score(join_observations(observed, target), catalogue[catalogue.compound_id.isin(ids)])
    exact = results["full"]["paired_permutation"]
    direct_ci = results["full"]["sensitivity_ci95"][NULL]
    direct_gate = (all(value[0] > 0 for value in direct_ci.values())
                   and exact["interaction_pearson"]["paired_bootstrap_ci95"][0] > 0
                   and all(item["one_sided_exact_p"] < .05 for item in exact.values()))
    calibrated = join_observations(predicted[predicted.track.eq("calibration")], outcomes)
    checks = calibration_score(calibrated)
    gains = np.array(list(seal["provenance"]["calibration_global_gains"].values()))
    publish_json(args.out, {"scope": SCOPE, "input_sha256": sha256(path), "workbook_sha256": WORKBOOK_HASH,
        "forecast_sha256": seal["sha256"], "archived_comparison_sha256": FORECASTS, "direct": results,
        "direct_gate": bool(direct_gate), "calibration": checks, "calibration_gate": all(checks["gate"].values()),
        "global_slope_min_max": [float(gains.min()), float(gains.max())]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action, fields in {
        "prepare": ("original", "workbook"), "predict": ("inputs",), "score": ("inputs", "forecast", "pdo", "tissue"),
    }.items():
        sub = commands.add_parser(action)
        for field in (*fields, "out"):
            sub.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    {"prepare": measurements, "predict": predict, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
