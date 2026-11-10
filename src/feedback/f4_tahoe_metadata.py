"""F4: Tahoe-100M design/metadata audit for SNU-423 (CVCL_0366) and HepG2/C3A (CVCL_1098).

Reads (read-only) the official Tahoe-100M metadata tables and the full per-cell
obs_metadata (100.6M rows) row-group by row-group, and writes small summaries:

  <out_dir>/snu423_drug_dose_plate_summary.tsv   one row per SNU-423 sample (drug x dose x plate well)
  <out_dir>/hepg2c3a_drug_dose_plate_summary.tsv same for HepG2/C3A
  <out_dir>/tahoe_cell_line_summary.tsv           cells per Tahoe-100M line (all / pass_filter)
  <out_dir>/tahoe_design_summary.json             design facts (doses, plates, DMSO, replicates)
  <asset_dir>/{snu423,hepg2c3a}_cell_qc.parquet   per-cell QC (barcode, pass_filter, mito, phase ...)

Heavy: run through Slurm (cpu partition).
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TAHOE = Path("/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata")
LINES = {"CVCL_0366": "snu423", "CVCL_1098": "hepg2c3a"}
AGG_COLS = ["cell_line", "sample", "pass_filter"]
NUM_COLS = ["gene_count", "tscp_count", "mread_count", "pcnt_mito"]


def parse_conc(s: str):
    t = ast.literal_eval(s)
    return t[0][0], float(t[0][1]), t[0][2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--asset-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    adir = Path(args.asset_dir)
    out.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    smeta = pd.read_parquet(TAHOE / "sample_metadata.parquet")
    parsed = smeta.drugname_drugconc.map(parse_conc)
    smeta["conc_uM"] = parsed.map(lambda x: x[1])
    smeta["conc_unit"] = parsed.map(lambda x: x[2])
    smeta["drug_clean"] = smeta.drug.str.strip()
    dmeta = pd.read_parquet(TAHOE / "drug_metadata.parquet")
    dmeta["drug_clean"] = dmeta.drug.str.strip()
    clmeta = pd.read_parquet(TAHOE / "cell_line_metadata.parquet")

    pf = pq.ParquetFile(TAHOE / "obs_metadata.parquet")
    cols = AGG_COLS + NUM_COLS + ["plate", "phase", "BARCODE_SUB_LIB_ID", "cell_name", "S_score", "G2M_score"]
    agg = None
    keep = defaultdict(list)
    n_rows = 0
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg, columns=cols)
        df = tb.to_pandas()
        n_rows += len(df)
        for c in ["cell_line", "pass_filter", "phase", "cell_name"]:
            df[c] = df[c].astype(str)
        g = df.groupby(AGG_COLS, observed=True).agg(
            n=("gene_count", "size"),
            gene_sum=("gene_count", "sum"),
            tscp_sum=("tscp_count", "sum"),
            mito_sum=("pcnt_mito", "sum"),
        )
        agg = g if agg is None else agg.add(g, fill_value=0)
        sub = df[df.cell_line.isin(LINES)]
        for cl, part in sub.groupby("cell_line"):
            keep[cl].append(part.drop(columns=["cell_line"]))
        if rg % 20 == 0:
            print(f"rg {rg}/{pf.num_row_groups} rows {n_rows:,} {time.time()-t0:.0f}s", flush=True)
    agg = agg.reset_index()
    print("obs rows", n_rows, flush=True)

    # cell-line level summary for the whole atlas
    cl_tab = agg.pivot_table(index="cell_line", columns="pass_filter", values="n", aggfunc="sum", fill_value=0)
    cl_tab["total"] = cl_tab.sum(axis=1)
    name_map = clmeta.drop_duplicates("Cell_ID_Cellosaur").set_index("Cell_ID_Cellosaur")[["cell_name", "Organ"]]
    cl_tab = cl_tab.join(name_map, how="left").sort_values("total", ascending=False)
    cl_tab.index.name = "cell_line"
    cl_tab.to_csv(out / "tahoe_cell_line_summary.tsv", sep="\t")

    design = {
        "obs_rows": int(n_rows),
        "pass_filter_levels": sorted(agg.pass_filter.unique().tolist()),
        "n_cell_lines_obs": int(agg.cell_line.nunique()),
        "n_samples": int(smeta["sample"].nunique()),
        "n_plates": int(smeta.plate.nunique()),
        "n_drugs_excl_dmso": int(smeta.loc[~smeta.drug.str.startswith("DMSO"), "drug_clean"].nunique()),
        "doses_uM": sorted(smeta.conc_uM.unique().tolist()),
        "dmso_samples_per_plate": smeta[smeta.drug.str.startswith("DMSO")].groupby("plate").size().to_dict(),
        "elapsed_s": None,
    }
    trt = smeta[~smeta.drug.str.startswith("DMSO")]
    dd = trt.groupby(["drug_clean", "conc_uM"]).agg(n_samples=("sample", "size"), n_plates=("plate", "nunique"))
    design["drug_dose_pairs"] = int(len(dd))
    design["drug_dose_by_n_samples"] = {int(k): int(v) for k, v in dd.n_samples.value_counts().items()}
    design["drug_dose_by_n_plates"] = {int(k): int(v) for k, v in dd.n_plates.value_counts().items()}
    design["drugs_by_n_plates"] = {int(k): int(v) for k, v in trt.groupby("drug_clean").plate.nunique().value_counts().items()}
    # are the three doses of a drug on the same plate?
    dp = trt.groupby("drug_clean").apply(lambda x: x.groupby("conc_uM").plate.apply(set).to_dict())
    same_plate = 0
    diff_plate = 0
    for drug, m in dp.items():
        plates = [frozenset(v) for v in m.values()]
        if len(plates) >= 2 and len(set.intersection(*[set(p) for p in plates])) > 0:
            same_plate += 1
        else:
            diff_plate += 1
    design["drugs_with_all_doses_sharing_a_plate"] = same_plate
    design["drugs_with_doses_on_disjoint_plates"] = diff_plate
    design["samples_per_plate"] = smeta.groupby("plate").size().to_dict()
    design["drugs_per_plate"] = trt.groupby("plate").drug_clean.nunique().to_dict()

    for cl, tag in LINES.items():
        cells = pd.concat(keep[cl], ignore_index=True)
        cells.to_parquet(adir / f"{tag}_cell_qc.parquet", index=False)
        a = agg[agg.cell_line == cl]
        tot = a.groupby("sample").n.sum().rename("n_cells_all")
        full = a[a.pass_filter == "full"].groupby("sample").agg(
            n_cells_full=("n", "sum"), gs=("gene_sum", "sum"), ts=("tscp_sum", "sum"), ms=("mito_sum", "sum"))
        tab = smeta.set_index("sample").join(tot, how="left").join(full, how="left")
        tab[["n_cells_all", "n_cells_full"]] = tab[["n_cells_all", "n_cells_full"]].fillna(0).astype(int)
        tab["mean_genes_full"] = tab.gs / tab.n_cells_full
        tab["mean_umi_full"] = tab.ts / tab.n_cells_full
        tab["mean_pct_mito_full"] = tab.ms / tab.n_cells_full
        dmso_med = tab[tab.drug.str.startswith("DMSO")].groupby("plate").n_cells_full.mean()
        tab["plate_dmso_mean_cells"] = tab.plate.map(dmso_med)
        tab["cell_ratio_vs_plate_dmso"] = tab.n_cells_full / tab.plate_dmso_mean_cells
        tab = tab.join(dd, on=["drug_clean", "conc_uM"], how="left")
        tab = tab.reset_index().merge(
            dmeta[["drug_clean", "targets", "moa-broad", "moa-fine", "human-approved"]], on="drug_clean", how="left")
        cols_out = ["sample", "plate", "drug_clean", "conc_uM", "conc_unit", "n_cells_all", "n_cells_full",
                    "plate_dmso_mean_cells", "cell_ratio_vs_plate_dmso", "mean_genes_full", "mean_umi_full",
                    "mean_pct_mito_full", "n_samples", "n_plates", "targets", "moa-broad", "moa-fine", "human-approved"]
        tab = tab[cols_out].rename(columns={"drug_clean": "drug", "n_samples": "drug_dose_n_wells",
                                             "n_plates": "drug_dose_n_plates"})
        tab.loc[tab.drug.str.startswith("DMSO"), ["drug_dose_n_wells", "drug_dose_n_plates"]] = np.nan
        tab.sort_values(["drug", "conc_uM", "plate"]).to_csv(out / f"{tag}_drug_dose_plate_summary.tsv", sep="\t",
                                                             index=False, float_format="%.4g")
        trt_t = tab[~tab.drug.str.startswith("DMSO")]
        dmso_t = tab[tab.drug.str.startswith("DMSO")]
        design[tag] = {
            "cells_obs_all": int(tab.n_cells_all.sum()),
            "cells_obs_pass_full": int(tab.n_cells_full.sum()),
            "pass_filter_counts": a.groupby("pass_filter").n.sum().astype(int).to_dict(),
            "n_drugs": int(trt_t.drug.nunique()),
            "n_drug_dose": int(trt_t.groupby(["drug", "conc_uM"]).ngroups),
            "dmso_cells_full_per_plate": dmso_t.groupby("plate").n_cells_full.sum().astype(int).to_dict(),
            "dmso_cells_full_total": int(dmso_t.n_cells_full.sum()),
            "trt_cells_full_per_well": trt_t.n_cells_full.describe().round(1).to_dict(),
            "wells_lt_50_cells": int((trt_t.n_cells_full < 50).sum()),
            "wells_lt_100_cells": int((trt_t.n_cells_full < 100).sum()),
            "wells_lt_20pct_dmso": int((trt_t.cell_ratio_vs_plate_dmso < 0.2).sum()),
            "phase_counts": cells.phase.value_counts().to_dict(),
        }
    design["elapsed_s"] = round(time.time() - t0, 1)
    (out / "tahoe_design_summary.json").write_text(json.dumps(design, indent=1, default=str))
    print(json.dumps(design, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
