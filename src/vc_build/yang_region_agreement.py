"""Measure agreement between disjoint tumor-region groups of the same HCC donor."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import score_across_cells
from .io import publish_json
from .v0plus_score import crossed_patient_drug_ci
from .yang_hcc_adaptation import BENCHMARK_FORECAST, original_inputs
from .yang_hcc_audit import COHORT
from .yang_hcc_external import outcomes


def two_region_sets(frame):
    samples = frame[["lineage_group", "cell_id"]].drop_duplicates().copy()
    region = samples.cell_id.str.extract(r"C(\d+)$", expand=False)
    samples["region"] = pd.to_numeric(region, errors="raise")
    if samples.region.isna().any() or samples.duplicated(["lineage_group", "region"]).any():
        raise ValueError("Ambiguous within-patient organoid region identity")
    ordered = samples.sort_values(["lineage_group", "region"])
    repeated = ordered.groupby("lineage_group").filter(lambda group: len(group) >= 2)
    repeated["half"] = repeated.groupby("lineage_group").cumcount() % 2
    assignment = repeated[["lineage_group", "cell_id", "half"]]
    joined = frame.merge(assignment, on=["lineage_group", "cell_id"], validate="many_to_one")
    result = joined.groupby(["lineage_group", "compound_id", "half"]).endpoint_value.mean().unstack()
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Every diagnostic donor must have observations in both halves")
    return result, samples, repeated


def analyze(values):
    measures = []
    for reference, target in ((0, 1), (1, 0)):
        group = values.reset_index()[["lineage_group", "compound_id"]].assign(
            track=f"half_{reference}_to_{target}", fold=0, endpoint_value=values[target].to_numpy(),
        )
        group["cell_id"] = group.lineage_group
        group["row_id"] = group.lineage_group + "|" + group.compound_id
        guess = group.assign(method="matched_region_half", predicted_value=values[reference].to_numpy())
        base = group.assign(method="constant", predicted_value=0.)
        combined = pd.concat([guess, base], ignore_index=True)
        per_drug, _ = score_across_cells(combined)
        subset = per_drug.loc[per_drug.method.eq("matched_region_half")]
        measures.append({
            "direction": group.track.iloc[0], "spearman": float(subset.spearman.mean()),
            "pearson": float(subset.pearson.mean()),
            "ci95": crossed_patient_drug_ci(combined, "matched_region_half", "constant"),
            "per_drug": subset.to_dict("records"),
        })
    a = values[0].unstack("compound_id").to_numpy()
    b = values[1].unstack("compound_id").to_numpy()
    observed = np.square(a-b).mean()
    rng = np.random.default_rng(312921)
    permuted = np.square(a[None, :, :] - b[rng.random((4999, len(b))).argsort(axis=1)]).mean(axis=(1, 2))
    return measures, {
        "matched_rmse": float(np.sqrt(observed)),
        "permuted_patient_median_rmse": float(np.sqrt(np.median(permuted))),
        "lower_tail_p": float((1 + (permuted <= observed).sum()) / (1 + len(permuted))),
        "null_draws": len(permuted),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("inputs", "teacher", "workbook", "out"):
        parser.add_argument("--" + option, required=True, type=Path)
    args = parser.parse_args()
    roster = original_inputs(args.inputs, args.teacher)
    frame = outcomes(args.workbook, roster)
    values, samples, repeated = two_region_sets(frame)
    measures, measure_rmse = analyze(values)
    publish_json(args.out, {"cohort": COHORT, "teacher_sha256": BENCHMARK_FORECAST,
                           "patients_with_multiple_regions": repeated.lineage_group.nunique(),
                           "patients_with_only_one_region": samples.lineage_group.nunique()
                           - repeated.lineage_group.nunique(), "regions": len(repeated),
                           "compounds": frame.set_index("compound_id").response_label.to_dict(),
                           "agreement": measures, "distance": measure_rmse})


if __name__ == "__main__":
    main()
