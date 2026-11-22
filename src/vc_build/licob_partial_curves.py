"""Borrow measured partial concentration windows without filling missing labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import join_observations
from .expanded_panel_cv import RECEIPTS, expanded_targets
from .external import KEY, SOURCE, _validate
from .function_data import new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_cold import PENALTY, checked
from .licob_curves import DOSES, DOSE_REGISTRATION, ENDPOINTS, FRACTIONS, endpoints, read_sources
from .licob_dose_sources import COLD, SCREENS, extract_curves, registration
from .licob_endpoints import drug_gram, paired_donors, read_outcomes, summarize
from .limore_corrected import OBSERVATIONS, registered, shared_support
from .pdo_transfer_cv import device_cuda
from .response_components import centered_kernels, orthogonal_parts
from .structured_kernel import cell_kernels, solve_residuals, training_grid
from .v0plus_cv import molecular_inputs
from .v0plus_licob import ASSET_HASHES
from .v0plus_neural import load_auxiliary
from .v0plus_score import compound_strata
from .yang_hcc_external import require_hash
from .zhu_dose_balance import robust_labels

ARCHIVE = "2278e41c2e87a7dc86265c907b3a8cb60a4672ad9fd12a140853095a42cabb5f"
WIDE_REF = "4cc7e3afa71fee91c0ca9a12ce06002d7a1016f9d4951b1833237830e3d7a2a1"
LIMORE_REF = "fe7f9e656801d63fd1263ec3e94093cad86b049130aa038d4f2e8fdd059b02bb"
FULL, PART, UNSMOOTHED, CONTROL, MEAN = (
    "complete_window", "partial_window", "partial_no_dose_sharing", "shuffled_partial", "drug_mean",
)
ARMS = (FULL, PART, UNSMOOTHED, CONTROL, MEAN)
LIMORE_METHODS = ("corrected/structured", "corrected/drug_ridge", "corrected/cellhit",
                  "corrected/codeae_inductive", "corrected/velodrome_inductive", "corrected/drug_mean")
LICOB_METHODS = ("curve_joint", "direct_endpoint")
LICOB_WIDE_METHODS = ("interaction_transfer", "chemical_mean")


def partial_values(dose, viability, *, minimum, second, maximum):
    if len(dose) < 5:
        return None
    dose = np.log10(dose)
    low, high = np.log10([minimum, maximum])
    grid = low + FRACTIONS * (high - low)
    mask = (grid >= dose[0] - 1e-5) & (grid <= dose[-1] + 1e-5)
    if mask.sum() < 3:
        return None
    value = np.full(len(grid), np.nan)
    value[mask] = np.interp(grid[mask], dose, np.clip(viability, 0, 1))
    return tuple(value)


def extract(args):
    rows, digest, qc = extract_curves(args, partial_values, DOSES, require_prism_window=False)
    if digest != DOSE_REGISTRATION:
        raise ValueError("Unexpected target concentration design")
    rows = rows.astype({col: "float64" for col in DOSES})
    with new_output(args.out) as directory:
        rows.to_parquet(directory / "partial.parquet", index=False)
        publish_json(directory / "receipt.json", {
            "registration": digest, "source": args.assay, "qc": qc,
            "sha256": sha256(directory / "partial.parquet"), "rows": len(rows),
            "compounds": rows.compound_id.nunique(), "observed_doses": rows[DOSES].count().to_dict(),
        })


def load_partial(args, complete):
    tables, digests, counts, coverage = [], {}, {}, {}
    for source in SCREENS:
        directory = args.partial / source
        path = directory / "receipt.json"
        stamp = json.loads(path.read_text())
        require_hash(directory / "partial.parquet", stamp["sha256"])
        rows = pd.read_parquet(directory / "partial.parquet")
        if (stamp["registration"], stamp["source"], stamp["rows"]) != (DOSE_REGISTRATION, source, len(rows)):
            raise ValueError("Unexpected partial source table")
        n = rows[DOSES].notna().sum(axis=1)
        if not n.ge(3).all() or not rows.source_id.eq(source).all() or not rows.row_id.is_unique:
            raise ValueError("Broken measured-dose mask")
        coverage[source] = {k: stamp[k] for k in ("rows", "compounds", "observed_doses")}
        eligible = rows.compound_id.isin(complete.compound_id) & ~rows.row_id.isin(complete.row_id)
        selected = rows[eligible].copy()
        counts[source] = {"added_rows": len(selected), "drugs": selected.compound_id.nunique(),
                          "point_counts": selected[DOSES].count().to_dict(),
                          "dose_counts": n[eligible].value_counts().sort_index().to_dict()}
        tables.append(selected)
        digests[source] = sha256(path)
    return pd.concat(tables, ignore_index=True), {"receipts": digests, "selected": counts, "coverage": coverage}


def permute_partial(partial):
    rng = np.random.default_rng(22639)
    shuffled = partial.copy()
    shuffled["mask"] = sum(shuffled[col].notna().astype(int) * (1 << k) for k, col in enumerate(DOSES))
    for _, block in shuffled.groupby([*SOURCE, "compound_id", "mask"], sort=True):
        order = rng.permutation(block.index)
        shuffled.loc[block.index, DOSES] = shuffled.loc[order, DOSES].to_numpy()
    return shuffled.drop(columns="mask")


def dose_grid(rows, cells, drugs, device):
    weights, values = [], []
    for endpoint in DOSES:
        panel = rows.dropna(subset=[endpoint])
        scaled, _ = robust_labels(panel, panel[endpoint])
        w, y = training_grid(scaled, cells, drugs, device=device)
        residual = orthogonal_parts(y.cpu().numpy(), w.cpu().numpy())[2]
        weights.append(w)
        values.append(torch.as_tensor(residual, device=device, dtype=w.dtype))
    width = len(drugs) * len(DOSES)
    return tuple(torch.stack(block, dim=-1).reshape(len(cells), width) for block in (weights, values))


def curve_predictions(complete, partial, query, rna, windows, edges, targets, auxiliary):
    rows = pd.concat([complete, partial], ignore_index=True)
    if not rows.row_id.is_unique:
        raise ValueError("Repeated native curves would change the comparison weights")
    _validate(rows, query)
    device = device_cuda()
    cells, held = (pd.Index(sorted(frame.cell_id.unique())) for frame in (rows, query))
    drugs, chemical = drug_gram(complete, targets, device)
    kernels, _ = cell_kernels(rna.loc[cells], rna.loc[held], edges, auxiliary, device=device)
    cell, cross = centered_kernels(*kernels["full"])
    # The complete-only arm should fit its own input transforms and RNA centering.
    baseline_cells = pd.Index(sorted(complete.cell_id.unique()))
    baseline, _ = cell_kernels(rna.loc[baseline_cells], rna.loc[held], edges, auxiliary, device=device)
    base_train, base_cross = centered_kernels(*baseline["full"])
    f = torch.as_tensor(FRACTIONS, dtype=chemical.dtype, device=device)
    dose = torch.exp(-.5 * ((f[:, None] - f[None, :]) / .25).square())
    coupled = torch.kron(chemical.contiguous(), dose.contiguous())
    independent = torch.kron(chemical.contiguous(), torch.eye(len(DOSES), device=device))
    means = complete.groupby("compound_id")[DOSES].mean().loc[drugs].to_numpy()
    q = complete.groupby("compound_id")[DOSES].quantile
    physical = (q(.75) - q(.25)).loc[drugs].to_numpy()
    physical = torch.as_tensor(np.maximum(physical, .01).ravel(), device=device, dtype=chemical.dtype)
    w0, _ = dose_grid(complete, cells, drugs, device)
    support_rows = cells.get_indexer(baseline_cells)
    support = w0[support_rows]

    def fit(data, train_cells, gram, held_gram, drug, observed_rows):
        w, z = dose_grid(data, train_cells, drugs, device)
        coeff, it, error = solve_residuals(gram, drug, w, z, PENALTY, max_steps=360)
        middle = coeff @ drug.T
        fitted = (gram @ middle)[observed_rows]
        rms = torch.sqrt((fitted.square() * support).sum(0) / support.sum(0))
        return held_gram @ middle, rms, {"iterations": it, "residual": error}

    unscaled, baseline_rms, diag = fit(complete, baseline_cells, base_train, base_cross, coupled,
                                       np.arange(len(baseline_cells)))
    target = baseline_rms * physical
    predictions = {FULL: (unscaled * physical).cpu().numpy()}
    details = {FULL: diag}
    shuffled = pd.concat([complete, permute_partial(partial)], ignore_index=True)
    for name, label_rows, kernel in ((PART, rows, coupled), (UNSMOOTHED, rows, independent),
                                     (CONTROL, shuffled, coupled)):
        estimated, rms, details[name] = fit(label_rows, cells, cell, cross, kernel, support_rows)
        scale = target / rms.clamp_min(1e-7)
        predictions[name] = (estimated * scale).cpu().numpy()
        details[name]["matched_rms_error"] = float((rms * scale - target).abs().max())
    predictions[MEAN] = np.zeros_like(predictions[FULL])
    idx = held.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    results = []
    for name, residual in predictions.items():
        curve = residual.reshape(len(held), len(drugs), len(DOSES)) + means
        values = endpoints(curve[idx], windows.set_index("compound_id").loc[query.compound_id])
        results.extend(query[KEY].assign(method=name, track=label, fold=0, predicted_value=predicted)
                       for label, predicted in values.items())
    return pd.concat(results, ignore_index=True), {
        "solvers": details, "complete_cells": len(baseline_cells), "pooled_cells": len(cells),
        "permuted_rows_changed": int((~np.isclose(partial[DOSES], shuffled.iloc[len(complete):][DOSES],
                                                equal_nan=True)).any(axis=1).sum()),
    }


def reference(path, digest, selectors, query):
    forecasts, stamp = verify_forecast(path)
    if stamp["sha256"] != digest:
        raise ValueError("The registered comparison forecast changed")
    return forecasts[forecasts.row_id.isin(query.row_id) & forecasts.method.isin(selectors)].copy()


def predict(args):
    if checked(args.cold, ("rna", "query")) != COLD:
        raise ValueError("Unexpected LICOB input")
    complete, digest, stamps = read_sources(argparse.Namespace(run=args.archive, inputs=args.inputs))
    _, archive = verify_forecast(args.archive / "forecast.parquet")
    if archive["sha256"] != ARCHIVE or archive["provenance"]["extracts"] != stamps:
        raise ValueError("Complete windows must be identical to the signed comparator source inputs")
    if len(complete) != 23482 or complete.compound_id.nunique() != 38:
        raise ValueError("Unexpected frozen complete-window training support")
    partial, coverage = load_partial(args, complete)
    if partial.compound_id.nunique() < 10 or len(partial) < 1000:
        raise ValueError("Insufficient extra observations on the declared same-support comparison")
    query = pd.read_parquet(args.cold / "query.parquet")
    query = query[query.compound_id.isin(complete.compound_id)]
    rna = pd.read_parquet(args.cold / "rna.parquet")
    _, limore, l_rna = registered(args, "corrected")
    limore = limore[limore.compound_id.isin(complete.compound_id)]
    axis = rna.columns.intersection(l_rna.columns)
    rna = pd.concat([rna[axis], l_rna.loc[sorted(limore.cell_id.unique()), axis]], verify_integrity=True)
    edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    targets, _ = expanded_targets(complete[~complete.compound_id.isin(targets)], targets, axis)
    auxiliary, _ = load_auxiliary(args.auxiliary)
    windows = pd.read_parquet(args.inputs / "dose.parquet")
    combined = pd.concat([query[KEY], limore[KEY]], ignore_index=True)
    forecast, audit = curve_predictions(complete, partial, combined, rna, windows, edges, targets, auxiliary)
    previous = reference(args.archive / "forecast.parquet", ARCHIVE, LICOB_METHODS, query)
    wide = reference(args.wide, WIDE_REF, LICOB_WIDE_METHODS, query).assign(track="auc", fold=0)
    licob = pd.concat([forecast[forecast.row_id.isin(query.row_id)], previous, wide], ignore_index=True)
    for outcome in ENDPOINTS:
        methods = (*ARMS, *LICOB_METHODS, *(LICOB_WIDE_METHODS if outcome == "auc" else ()))
        shared_support(licob[licob.track.eq(outcome)], query, methods)
    refs = reference(args.limore_ref, LIMORE_REF, LIMORE_METHODS, limore).assign(track="LIMORE")
    l_arms = forecast[forecast.row_id.isin(limore.row_id) & forecast.track.eq("auc")].assign(track="LIMORE")
    l_joint = pd.concat([l_arms, refs], ignore_index=True)
    shared_support(l_joint, limore, (*ARMS, *LIMORE_METHODS))
    freeze_table(pd.concat([licob, l_joint], ignore_index=True), args.out, provenance={
        "registration": digest, "licob": COLD, "archive": ARCHIVE, "wide": WIDE_REF, "limore": LIMORE_REF,
        "complete_rows": len(complete), "partial_rows": len(partial), "coverage": coverage,
        "compounds": sorted(complete.compound_id.unique()), "fit": audit,
        "licob_organoid_rows": len(query), "limore_rows": len(limore), "limore_compounds": limore.compound_id.nunique(),
    })


def assess(rows, comparisons, strata):
    report = summarize(rows, comparisons, strata)
    rank = {(r["baseline"], r["metric"]): r["ci95"][0] for r in report["metrics"]["paired_contrasts"]}
    report["gates"] = {}
    for a, b in comparisons:
        bounds = report["within_drug_ci95"][f"{a}/{b}"]
        report["gates"][b] = all(bounds[m][0] > 0 for m in ("spearman", "pearson")) and all(
            rank[b, k] > -.005 for k in ("spearman", "ndcg_at_5", "top5_overlap"))
    return report


def score(args):
    rows, seal = verify_forecast(args.forecast)
    proof = seal["provenance"]
    if (proof["registration"], proof["licob"], proof["archive"]) != (DOSE_REGISTRATION, COLD, ARCHIVE):
        raise ValueError("The two outcome files require a verified comparison freeze")
    if registration(args.inputs) != DOSE_REGISTRATION or checked(args.cold, ("query",)) != COLD:
        raise ValueError("A previously registered source or cohort has changed")
    query = pd.read_parquet(args.cold / "query.parquet")
    query = query[query.compound_id.isin(proof["compounds"])]
    dose = pd.read_parquet(args.inputs / "dose.parquet")
    require_hash(args.supplement, ASSET_HASHES["supplement"])
    result = {}
    for outcome, truth in read_outcomes(args.supplement, query, dose).items():
        block = paired_donors(rows[rows.track.eq(outcome)], truth)
        methods = (FULL, UNSMOOTHED, CONTROL, MEAN, *LICOB_METHODS,
                   *(LICOB_WIDE_METHODS if outcome == "auc" else ()))
        result[outcome] = assess(block, [(PART, name) for name in methods], compound_strata(query))
    require_hash(args.labels, OBSERVATIONS)
    receipt = args.original / "receipt.json"
    require_hash(receipt, RECEIPTS["original"])
    require_hash(args.original / "test_inputs/rows.parquet",
                 json.loads(receipt.read_text())["artifacts"]["test_inputs/rows.parquet"]["sha256"])
    limore = pd.read_parquet(args.original / "test_inputs/rows.parquet")
    actual = pd.read_parquet(args.labels)
    population = join_observations(rows[rows.track.eq("LIMORE")], actual)
    result["LIMORE"] = assess(population, [(PART, name) for name in (FULL, UNSMOOTHED, CONTROL, MEAN, *LIMORE_METHODS)],
                              compound_strata(limore))
    result["joint_auc"] = all(all(result[k]["gates"].values()) for k in ("auc", "LIMORE"))
    publish_json(args.out, {"forecast_sha256": seal["sha256"], "labels": [ASSET_HASHES["supplement"], OBSERVATIONS],
                            "results": result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage, fields in {
        "extract": ("original", "ctrp", "gdsc", "prism-treatments"),
        "predict": ("cold", "original", "core", "archive", "partial", "wide", "limore-ref",
                    "priors", "hallmark", "targets", "auxiliary"),
        "score": ("cold", "original", "forecast", "supplement", "labels"),
    }.items():
        command = sub.add_parser(stage)
        for field in ("out", "inputs", *fields):
            command.add_argument("--" + field, type=Path, required=True)
        if stage == "extract":
            command.add_argument("--assay", choices=SCREENS, required=True)
    args = parser.parse_args()
    {"extract": extract, "predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
