"""F4: Tahoe EmeraldBay design audit (read-only; metadata columns only, no expression decoded).

Outputs (reports/feedback/F4):
  emeraldbay_liver_condition_summary.tsv   SNU-423 / HepG2-C3A cells per condition x sample (+ growth rate)
  emeraldbay_design_summary.json           lines, conditions, doses, replicates, overlap with Tahoe-100M drugs
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

EB = Path("/mnt/S1/sdata/pubdata/Virtual_Cell/EmeraldBay")
T100 = Path("/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata")
LIVER = {"CVCL_0366": "SNU-423", "CVCL_1098": "HepG2/C3A", "CVCL_0525": "SK-HEP-1"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    parts = []
    for f in sorted(glob.glob(str(EB / "expression_data/train-*.parquet"))):
        t = pq.read_table(f, columns=["drug", "drugname_drugconc", "cell_line", "sample"]).to_pandas()
        parts.append(t.groupby(["cell_line", "drug", "drugname_drugconc", "sample"]).size().rename("n_cells"))
    c = pd.concat(parts).groupby(level=[0, 1, 2, 3]).sum().reset_index()
    ss = pd.read_parquet(EB / "metadata/summary_statistics.parquet")
    c = c.merge(ss.groupby(["cell_line", "condition"]).growth_rate.mean().rename("growth_rate_mean").reset_index(),
                left_on=["cell_line", "drugname_drugconc"], right_on=["cell_line", "condition"], how="left")
    c["n_drugs_in_condition"] = c.drugname_drugconc.map(lambda s: len(ast.literal_eval(s)) if s.startswith("[") else 0)
    liver = c[c.cell_line.isin(LIVER)].copy()
    liver["cell_name"] = liver.cell_line.map(LIVER)
    liver.drop(columns=["condition"]).sort_values(["cell_name", "drug", "drugname_drugconc", "sample"]).to_csv(
        out / "emeraldbay_liver_condition_summary.tsv", sep="\t", index=False, float_format="%.4g")
    t100_drugs = set(pd.read_parquet(T100 / "drug_metadata.parquet").drug.str.strip())
    single = c[c.n_drugs_in_condition == 1]
    eb_drugs = set(single.drug.str.strip()) - {"DMSO_TF"}
    cond_per_sample = c.groupby("sample").drugname_drugconc.nunique()
    rep = c.groupby(["cell_line", "drugname_drugconc"])["sample"].nunique()
    design = {
        "n_cells": int(c.n_cells.sum()), "n_cell_lines": int(c.cell_line.nunique()),
        "n_conditions": int(c.drugname_drugconc.nunique()), "n_samples": int(c["sample"].nunique()),
        "n_single_drug_conditions": int(single.drugname_drugconc.nunique()),
        "n_combo_conditions": int(c[c.n_drugs_in_condition > 1].drugname_drugconc.nunique()),
        "n_single_drugs": len(eb_drugs), "single_drugs_also_in_tahoe100m": sorted(eb_drugs & t100_drugs),
        "samples_per_condition_median": float(rep.median()), "samples_per_condition_distribution":
            {int(k): int(v) for k, v in rep.value_counts().items()},
        "conditions_per_sample_max": int(cond_per_sample.max()),
        "dmso_cells_by_line_median": float(c[c.drug.str.startswith("DMSO")].groupby("cell_line").n_cells.sum().median()),
        "liver": {},
    }
    for cid, name in LIVER.items():
        l = c[c.cell_line == cid]
        if not len(l):
            continue
        trt = l[~l.drug.str.startswith("DMSO")]
        design["liver"][name] = {
            "cells": int(l.n_cells.sum()), "dmso_cells": int(l[l.drug.str.startswith("DMSO")].n_cells.sum()),
            "dmso_samples": int(l[l.drug.str.startswith("DMSO")]["sample"].nunique()),
            "conditions": int(trt.drugname_drugconc.nunique()),
            "cells_per_condition_x_sample": trt.n_cells.describe().round(1).to_dict(),
            "replicate_samples_per_condition": {int(k): int(v) for k, v in
                                                trt.groupby("drugname_drugconc")["sample"].nunique().value_counts().items()},
        }
    (out / "emeraldbay_design_summary.json").write_text(json.dumps(design, indent=1))
    print(json.dumps(design, indent=1)[:5000])


if __name__ == "__main__":
    main()
