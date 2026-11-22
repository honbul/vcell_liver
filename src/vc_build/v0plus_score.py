"""Score one merged public freeze against the separately registered measurement file."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .evaluate import join_observations, score, score_across_cells
from .io import publish_json, read_table, sha256, verify_forecast

CYTOTOXIC = {
    "Camptothecin", "Docetaxel", "Doxorubicin", "Epirubicin", "Etoposide", "Fluorouracil",
    "Irinotecan", "Methotrexate", "Paclitaxel", "Topotecan", "Vinblastine", "Decitabine",
}
HCC_DRUGS = {"Cabozantinib", "Lenvatinib", "Regorafenib", "Sorafenib"}
KINASE = re.compile(
    r"EGFR|HER2|VE.?FGR|PDGFR|FGFR|\bMET\b|\bALK\b|\bRET\b|\bABL\b|\bSRC\b|\bCK2\b|"
    r"\bKIT\b|C-KIT|\bRAF\b|\bMEK\b|\bPI3K\b|\bMTOR\b|\bJAK\b|\bPIM\b|"
    r"\bCDK|\bPLK|\bPDK|\bBTK\b|\bIGF1R\b|\bKDR\b|\bFLT3\b|AURORA|CHK1|DNAPK", re.I
)


def compound_strata(compounds):
    table = compounds[["compound_id", "source_compound_label", "drug_target_source_labels"]].drop_duplicates()
    if not table.compound_id.is_unique:
        raise ValueError("Ambiguous original annotations for stratified scoring")
    names = table.source_compound_label
    mechanisms = table.drug_target_source_labels.fillna("")
    result = {
        "hcc_tki": names.isin(HCC_DRUGS), "nonselective_cytotoxics": names.isin(CYTOTOXIC),
        "kinase_inhibitor": mechanisms.str.contains(KINASE),
        "targeted_or_kinase": mechanisms.str.contains(KINASE) | mechanisms.str.contains("HDAC", case=False),
    }
    for name, label in {
        "egfr": r"EGFR|HER2", "met": r"\bMET\b", "fgfr": r"FGFR",
        "mtor": r"\bMTOR\b", "hdac": r"HDAC",
    }.items():
        result[name] = mechanisms.str.contains(label, case=False, regex=True)
    return {name: sorted(table.loc[matched, "compound_id"]) for name, matched in result.items()}


def axis_metrics(joined, comparisons):
    report, cells = score(joined, endpoint_direction="higher", rank_only=True, comparisons=comparisons)
    drugs, contrasts = score_across_cells(joined, comparisons=comparisons)
    common = drugs.groupby(["track", "method"], sort=True)
    extra = []
    for (track, method), group in common:
        eligible = group.dropna(subset=["spearman", "pearson"])
        extra.append({
            "track": track, "method": method, "n_drugs": len(eligible),
            "within_drug_spearman": float(eligible.spearman.mean()) if len(eligible) else None,
            "within_drug_pearson": float(eligible.pearson.mean()) if len(eligible) else None,
        })
    return report | {"across_cell_scores": extra, "across_cell_drug_only_intervals": contrasts}, drugs, cells


def correlations(a, p):
    present = np.isfinite(a) & np.isfinite(p)
    count = present.sum(axis=1)
    a = np.where(present, a, 0)
    p = np.where(present, p, 0)
    da = np.where(present, a - a.sum(axis=1, keepdims=True) / np.maximum(count[:, None, :], 1), 0)
    dp = np.where(present, p - p.sum(axis=1, keepdims=True) / np.maximum(count[:, None, :], 1), 0)
    sa, sp = (da ** 2).sum(axis=1), (dp ** 2).sum(axis=1)
    denominator = np.sqrt(sa * sp)
    result = np.divide((da * dp).sum(axis=1), denominator, out=np.zeros_like(denominator), where=denominator > 1e-16)
    return np.where((count >= 5) & (sa > 1e-16), result, np.nan)


def crossed_patient_drug_ci(joined, candidate, comparator, *, repeats=2400, seed=11942):
    """Paired resampling of patient blocks and drugs on the original registered panel."""
    matrices = {
        method: (joined[joined.method.eq(method)]
                 .groupby(["lineage_group", "compound_id"])[["endpoint_value", "predicted_value"]].mean()
                 .unstack("compound_id").sort_index(axis=1))
        for method in (candidate, comparator)
    }
    a, b = matrices[candidate].align(matrices[comparator], join="inner", axis=None)
    a_truth, b_truth = a.endpoint_value.to_numpy(), b.endpoint_value.to_numpy()
    a_pred, b_pred = a.predicted_value.to_numpy(), b.predicted_value.to_numpy()
    common = np.isfinite(a_pred) & np.isfinite(b_pred) & np.isfinite(a_truth) & np.isfinite(b_truth)
    if (a_truth[common] != b_truth[common]).any():
        raise ValueError("Matched predictions must share outcome identities")
    a_truth, a_pred, b_pred = [np.where(common, matrix, np.nan) for matrix in (a_truth, a_pred, b_pred)]
    drugs = np.where((common.sum(axis=0) >= 5) & (np.nanstd(a_truth, axis=0) > 1e-8))[0]
    if len(drugs) < 2:
        return {}
    rng = np.random.default_rng(seed)
    draws = {"spearman": [], "pearson": []}
    for start in range(0, repeats, 100):
        n = min(100, repeats - start)
        patients = rng.integers(len(a_truth), size=(n, len(a_truth)))
        sampled = rng.choice(drugs, size=(n, len(drugs)))
        aa, pp, bb = (x[patients] for x in (a_truth, a_pred, b_pred))
        for name, arrays in (
            ("pearson", [aa, pp, bb]), ("spearman", [rankdata(x, axis=1, nan_policy="omit") for x in (aa, pp, bb)])
        ):
            observed, predicted, baseline = arrays
            changes = correlations(observed, predicted) - correlations(observed, baseline)
            sampled_changes = np.take_along_axis(changes, sampled, axis=1)
            draws[name].extend(np.nanmean(sampled_changes, axis=1).tolist())
    return {name: np.nanquantile(draw, [0.025, 0.975]).tolist() for name, draw in draws.items()}


def score_external(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    forecast, freeze = verify_forecast(args.forecast)
    provenance = freeze["provenance"]
    if sha256(args.settings) != provenance["settings_sha256"]:
        raise ValueError("Model settings changed after freeze")
    settings = json.loads(args.settings.read_text())
    query_receipt = args.cohort_bundle / "receipt.json"
    if sha256(query_receipt) != provenance["query_receipt_sha256"]:
        raise ValueError("Cohort receipt changed after forecast")
    roster_path = args.cohort_bundle / "test_inputs/rows.parquet"
    registered = json.loads(query_receipt.read_text())
    if sha256(roster_path) != registered["artifacts"]["test_inputs/rows.parquet"]["sha256"]:
        raise ValueError("Cohort roster changed after forecast")
    outcome_path = args.cohort_bundle / "test_labels/outcomes.parquet"
    if sha256(outcome_path) != provenance["registered_observations_sha256"]:
        raise ValueError("Cohort outcome differs from preregistered hash")
    roster = read_table(roster_path)
    observed = read_table(outcome_path)
    merged = join_observations(forecast, observed)
    if args.direction == "lower":
        merged.endpoint_value *= -1
    baseline = settings["best_traditional"]
    public = settings["best_inductive_public"]
    pairs = [("v0plus_inductive", other) for other in ("drug_mean", baseline, public)]
    pairs += [("v0plus_transductive", "v0plus_inductive")]
    pairs.append((baseline, "drug_mean"))
    full, per_drug, _ = axis_metrics(merged, pairs)
    full["crossed_patient_drug_ci95"] = [
        {"candidate": a, "baseline": b, "intervals": crossed_patient_drug_ci(merged, a, b)}
        for a, b in pairs
    ]
    strata = compound_strata(roster)
    slices = {}
    for name, drugs in strata.items():
        subset = merged[merged.compound_id.isin(drugs)]
        if len(subset):
            slice_summary, _, _ = axis_metrics(subset, pairs)
            if len(drugs) <= 5:
                for method in slice_summary["scores"]:
                    method.pop("macro_ndcg_at_5", None)
                    method.pop("macro_top5_overlap", None)
                slice_summary["paired_contrasts"] = [comparison for comparison in slice_summary["paired_contrasts"]
                                                      if comparison["metric"] == "spearman"]
            slices[name] = {"compounds": len(drugs), "compound_ids": drugs, **slice_summary}
    core = per_drug[per_drug.method.isin(["v0plus_inductive", "v0plus_transductive", baseline, public, "drug_mean"])]
    publish_json(args.out, {
        "forecast_sha256": freeze["sha256"], "label_sha256": provenance["registered_observations_sha256"],
        "cohort": provenance["cohort"], "models": int(merged.cell_id.nunique()),
        "lineages": int(merged.lineage_group.nunique()), "compounds": int(merged.compound_id.nunique()),
        "rows": int(merged.row_id.nunique()), "assay_direction": args.direction,
        "best_source_validated_traditional": baseline, "best_source_validated_inductive_public": public,
        "full": full, "strata": slices,
        "compound_annotations": roster[["compound_id", "source_compound_label", "drug_target_source_labels"]]
        .drop_duplicates().sort_values("compound_id").fillna("").to_dict("records"),
        "per_drug": core.astype(object).where(pd.notna(core), None).to_dict("records"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("forecast", "settings", "cohort-bundle", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--direction", choices=("higher", "lower"), required=True)
    score_external(parser.parse_args())


if __name__ == "__main__":
    main()
