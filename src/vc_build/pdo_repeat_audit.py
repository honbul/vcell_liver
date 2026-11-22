"""Measure native AUC agreement of the exact PDO–drug pairs shared by studies."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import rank_metrics
from .io import publish_json
from .pdo_transfer_cv import inputs, PDO_HASH, STUDIES


def rank_and_linear(values):
    return (float(values[left].corr(values[right], method="spearman")),
            float(values[left].corr(values[right]))) if all(
        np.ptp(values[key].to_numpy()) > 1e-8 for key in (left, right)
    ) else (0., 0.)


left, right = "auc_a", "auc_b"


def one_pair(observations, a, b):
    keys = ["lineage_group", "compound_id"]
    x = observations[observations.source.eq(a)][[*keys, "endpoint_value"]].rename(columns={"endpoint_value": left})
    y = observations[observations.source.eq(b)][[*keys, "endpoint_value"]].rename(columns={"endpoint_value": right})
    common = x.merge(y, on=keys, validate="one_to_one")
    drugs, patients = [], []
    for _, group in common.groupby("compound_id"):
        if len(group) >= 3 and np.ptp(group[left].to_numpy()) > 1e-8:
            drugs.append(rank_and_linear(group))
    for _, case in common.groupby("lineage_group"):
        if len(case) >= 5:
            rank, linear = rank_and_linear(case)
            forward = rank_metrics(-case[right].to_numpy(), -case[left].to_numpy())
            backward = rank_metrics(-case[left].to_numpy(), -case[right].to_numpy())
            patients.append((rank, linear, (forward["ndcg_at_5"] + backward["ndcg_at_5"]) / 2,
                             (forward["top5_overlap"] + backward["top5_overlap"]) / 2))

    return {
        "study_a": a, "study_b": b, "paired_rows": len(common),
        "paired_drugs": int(common.compound_id.nunique()), "paired_models": int(common.lineage_group.nunique()),
        "mean_absolute_raw_auc_difference": float((common[left] - common[right]).abs().mean()) if len(common) else None,
        "within_drug_eligible_at_least_three_donors": len(drugs),
        "within_drug_eligible_at_least_five_donors": int(common.groupby("compound_id").lineage_group.nunique().ge(5).sum()),
        "within_drug_spearman": float(np.mean(drugs, axis=0)[0]) if drugs else None,
        "within_drug_pearson": float(np.mean(drugs, axis=0)[1]) if drugs else None,
        "within_model_eligible_count": len(patients),
        **dict(zip(("within_model_spearman", "within_model_pearson", "within_model_ndcg_at_5", "within_model_precision_at_5"),
                   np.mean(patients, axis=0).tolist() if patients else [None] * 4)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs(args.pdo)
    observations = pd.read_parquet(args.pdo / "observations.parquet",
                                   columns=["source", "lineage_group", "compound_id", "endpoint_value"])
    publish_json(args.out, {"pdo_receipt": PDO_HASH,
                            "native_agreement": [one_pair(observations, a, b) for a, b in combinations(STUDIES, 2)]})


if __name__ == "__main__":
    main()
