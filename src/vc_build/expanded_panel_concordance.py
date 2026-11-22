"""Measure matched native response agreement on the registered supplemental drugs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .expanded_panel_cv import RECEIPTS, registered_inputs
from .external import native_sensitivity
from .io import publish_json


def associations(rows, group, minimum):
    scores = []
    for name, block in rows.groupby(group, sort=True):
        a, b = (block[source].to_numpy() for source in ("CTRPv2", "PRISM20Q2"))
        if len(block) < minimum or min(np.ptp(a), np.ptp(b)) <= 1e-8:
            continue
        first, second = (pd.Series(vector).rank().to_numpy() for vector in (a, b))
        scores.append({"group": name, "support": len(block),
                       "spearman": float(np.corrcoef(first, second)[0, 1]),
                       "pearson": float(np.corrcoef(a, b)[0, 1])})
    table = pd.DataFrame(scores)
    return {
        "groups": len(table), "minimum_support": minimum,
        "macro_spearman": float(table.spearman.mean()) if len(table) else None,
        "macro_pearson": float(table.pearson.mean()) if len(table) else None,
        "support_median": float(table.support.median()) if len(table) else None,
        "positive_spearman": int(table.spearman.gt(0).sum()) if len(table) else 0,
    }, scores


def diagnostics(rows, adult_only):
    if adult_only:
        rows = rows[rows.lineage_role.eq("adult_hcc")]
    rows = rows[rows.source_id.isin(["CTRPv2", "PRISM20Q2"])].copy()
    rows["native"] = native_sensitivity(rows)
    axes = ["cell_id", "lineage_group", "compound_id"]
    matched = rows.pivot(index=axes, columns="source_id", values="native").dropna()
    group_axes = ["lineage_group", "compound_id"]
    lineages = matched.groupby(group_axes).mean().reset_index()
    across, per_drug = associations(lineages, "compound_id", 8 if adult_only else 30)
    within, _ = associations(lineages, "lineage_group", 5)
    return {
        "paired_cell_compound_rows": len(matched), "paired_lineage_compound_rows": len(lineages),
        "paired_drugs": int(lineages.compound_id.nunique()),
        "paired_lineages": int(lineages.lineage_group.nunique()),
        "within_drug": across, "within_lineage": within, "per_drug": per_drug,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "core", "extension", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    _, core, _, extra = registered_inputs(args, include_extension=True)
    output = {
        label: {cohort: diagnostics(rows, adult) for cohort, adult in (("all", False), ("hcc", True))}
        for label, rows in (("corrected_core", core), ("extension", extra))
    }
    publish_json(args.out, {"metrics": output, "input_receipts": RECEIPTS})


if __name__ == "__main__":
    main()
