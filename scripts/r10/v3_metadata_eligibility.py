"""R10 v3 P0: metadata-level design summaries for the independent-data candidates.

Reads only small metadata files (hash sample sheets, pData, gene annotations, Tahoe-100M sample/cell-line
metadata) and the fixed v1 atlas decoders (HVG ids). No expression matrix is read.

Run with `python3 -I` from outside the download directory; all paths are arguments.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SINGLE_HASH_QC = dict(qval_max=0.05, ratio_min=10.0, n_umi_min=500)  # descriptive analogue of the v1 W/P rule


def sciplex2(meta: str, out: str) -> dict:
    hs = pd.read_csv(f"{meta}/GSM4150377_sciPlex2_hashSampleSheet.txt.gz", sep="\t", header=None)
    lab = hs[0].str.extract(r"^(\w+?)_([0-9.]+)_(\w)(\w)(\d+)$")
    lab.columns = ["drug", "dose_uM", "hash_plate", "row", "col"]
    p = pd.read_csv(f"{meta}/GSM4150377_sciPlex2_pData.txt.gz", sep=" ")
    q = p[(p.qval < SINGLE_HASH_QC["qval_max"]) & (p.top_to_second_best_ratio >= SINGLE_HASH_QC["ratio_min"])
          & (p["n.umi"] >= SINGLE_HASH_QC["n_umi_min"])].copy()
    ql = q.top_oligo.str.extract(r"^(\w+?)_([0-9.]+)_(\w)(\w)(\d+)$")
    ql.columns = ["drug", "dose_uM", "hash_plate", "row", "col"]
    ql["well"] = q.top_oligo.values
    per_well = ql.groupby(["drug", "dose_uM", "hash_plate", "well"]).size().rename("cells").reset_index()
    sheet = lab.groupby(["drug", "dose_uM", "hash_plate"]).size().rename("n_wells_sheet").reset_index()
    agg = per_well.groupby(["drug", "dose_uM", "hash_plate"]).cells.agg(
        n_wells_with_qc_cells="count", cells_min="min", cells_median="median", cells_sum="sum").reset_index()
    t = sheet.merge(agg, how="left", on=["drug", "dose_uM", "hash_plate"]).fillna(0)
    t["dose_nM"] = (t.dose_uM.astype(float) * 1000).round(1)
    t["wells_ge20_cells"] = t.apply(lambda r: int((per_well[(per_well.drug == r.drug) & (per_well.dose_uM == r.dose_uM)
                                                            & (per_well.hash_plate == r.hash_plate)].cells >= 20).sum()), axis=1)
    t = t.sort_values(["drug", "dose_nM", "hash_plate"])
    t.to_csv(f"{out}/candidate_sciplex2_wells.tsv", sep="\t", index=False)
    e1000 = t[t.dose_nM.isin([100.0, 1000.0, 10000.0])]
    return dict(
        n_cells_pdata=int(len(p)), n_cells_single_hash_qc=int(len(q)),
        n_umi_median_all=float(p["n.umi"].median()), n_umi_median_qc=float(q["n.umi"].median()),
        n_umi_min_in_release=float(p["n.umi"].min()),
        libraries=sorted(p["sample"].unique().tolist()),
        drugs=sorted(lab.drug.unique().tolist()), doses_uM=sorted(lab.dose_uM.astype(float).unique().tolist()),
        hash_plates=sorted(lab.hash_plate.unique().tolist()),
        wells_per_drug_dose=lab.groupby(["drug", "dose_uM"]).size().describe().to_dict(),
        e1000_wells_lt20_cells=int((e1000.n_wells_sheet - e1000.wells_ge20_cells).sum()),
        e1000_wells_total=int(e1000.n_wells_sheet.sum()),
    )


def sciplex4(meta: str, out: str) -> dict:
    hs = pd.read_csv(f"{meta}/GSM4150379_sciPlex4_hashSampleSheet.txt.gz", sep=r"\s+", header=None)
    x = hs[0].str.split("_", expand=True)
    x.columns = ["supp_dose", "supp", "hdac_dose_uM", "hdac", "line", "plate", "well"]
    t = x.groupby(["line", "hdac", "hdac_dose_uM", "plate", "supp", "supp_dose"]).size().rename("n_wells").reset_index()
    t.to_csv(f"{out}/candidate_sciplex4_layout.tsv", sep="\t", index=False)
    p = pd.read_csv(f"{meta}/GSM4150379_sciPlex4_pData.txt.gz", sep=" ")
    nosupp = x[(x.supp_dose == "0") & (x.supp == "DMSO") & (x.hdac != "DMSO")]
    return dict(
        n_cells_pdata=int(len(p)), pdata_columns=p.columns.tolist(),
        n_umi_median_all=float(p["n.umi"].median()), libraries=sorted(p["sample"].unique().tolist()),
        plates=sorted(x.plate.unique().tolist()),
        no_supplement_dose_series_plates=nosupp.groupby(["line", "hdac"]).plate.unique().apply(list).to_dict(),
    )


def tahoe(tdir: str, out: str) -> dict:
    s = pd.read_parquet(f"{tdir}/sample_metadata.parquet")
    s["conc_uM"] = s.drugname_drugconc.astype(str).str.extract(r", ([0-9.]+), ")[0].astype(float)
    ct = pd.crosstab(s.plate, s.conc_uM)
    ct.to_csv(f"{out}/candidate_tahoe_plate_dose.tsv", sep="\t")
    c = pd.read_parquet(f"{tdir}/cell_line_metadata.parquet")
    drug_rows = s[s.drug != "DMSO_TF"]
    return dict(
        n_samples=int(len(s)), n_plates=int(s.plate.nunique()), n_drugs=int(drug_rows.drug.nunique()),
        concentrations_uM=sorted(s.conc_uM.dropna().unique().tolist()),
        samples_per_drug_conc=drug_rows.groupby(["drug", "conc_uM"]).size().value_counts().sort_index().to_dict(),
        plates_with_single_dominant_dose=int((ct.drop(columns=[0.0], errors="ignore").max(axis=1) >= 90).sum()),
        cell_line_names_present={k: bool(c.cell_name.isin(v).any()) for k, v in
                                 {"A549": ["A549"], "MCF7": ["MCF-7", "MCF7"], "K562": ["K-562", "K562"]}.items()},
        note="cell_line_metadata lists names only; per-line cell counts need obs_metadata (2.2 GB), not read",
    )


def genes(meta: str, cache: str) -> dict:
    res = {}
    src = pd.read_csv(f"{cache}/genes_human.tsv", sep="\t")
    for tag, f in [("sciplex2", "GSM4150377_sciPlex2_A549_Transcription_Modulators_gene.annotations.txt.gz"),
                   ("sciplex4", "GSM4150379_sciPlex4_A549_MCF7_HDACi_gene.annotations.txt.gz")]:
        ids = set(pd.read_csv(f"{meta}/{f}", sep="\t", header=None)[0].astype(str))
        r = dict(n_gene_ids=len(ids), n_source_ids=int(len(src)), source_ids_present=int(src.id.isin(ids).sum()))
        for line in ["A549", "MCF7", "K562"]:
            z = np.load(f"{cache}/atlas_{line}/decoder.npz", allow_pickle=True)
            hv = z["hvg_id"].astype(str)
            r[f"hvg_present_{line}"] = f"{int(np.isin(hv, list(ids)).sum())}/{len(hv)}"
        res[tag] = r
    for line in ["A549", "MCF7"]:
        o = pd.read_parquet(f"{cache}/obs_{line}.parquet", columns=["n.umi"])
        res[f"source_{line}_qc_n_umi_median"] = float(o["n.umi"].median())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--tahoe", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = dict(single_hash_qc_rule=SINGLE_HASH_QC, sciplex2=sciplex2(a.meta, a.out), sciplex4=sciplex4(a.meta, a.out),
               tahoe=tahoe(a.tahoe, a.out), genes=genes(a.meta, a.cache))
    with open(f"{a.out}/metadata_checks.json", "w") as fh:
        json.dump(res, fh, indent=1, default=str, sort_keys=True)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
