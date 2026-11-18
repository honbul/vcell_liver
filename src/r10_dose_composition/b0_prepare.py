"""B0: hash QC, cell- and well-level dose/replicate/plate map, and the sparse 24 h count cache.

Inputs are the original GEO files of GSM4150378 only (pData, cell/gene annotation, hashTable
metadata, UMI triplet matrix). Outputs (server, under assets/cache/...):
  cells_all.parquet        one row per pData barcode with QC flags (cell-level barcode map)
  wells.tsv                one row per 24 h culture well (hash combo) incl. wells without cells
  counts_<line>.npz        CSR counts, QC-passed 24 h cells x human genes, row order = obs file
  obs_<line>.parquet       obs of those cells
  genes_human.tsv          human gene ids/names, column order of the CSR matrices
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

PFX = "GSM4150378_sciPlex3_"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    raw, out = cfg["raw_dir"], cfg["cache_dir"]
    os.makedirs(out, exist_ok=True)
    q = cfg["qc"]

    p = pd.read_csv(f"{raw}/{PFX}pData.txt.gz", sep=" ", quotechar='"', low_memory=False)
    ann = pd.read_csv(f"{raw}/{PFX}A549_MCF7_K562_screen_cell.annotations.txt.gz", sep="\t", header=None,
                      names=["cell", "sample"])
    assert len(ann) == len(p) and (ann.cell.values == p.cell.values).all(), "pData / cell annotation order"
    genes = pd.read_csv(f"{raw}/{PFX}A549_MCF7_K562_screen_gene.annotations.txt.gz", sep=" ")
    meta = pd.read_csv(f"{raw}/{PFX}A549_MCF7_K562_hashTable_metadata.txt.gz", sep="\t")
    log("pData", p.shape, "genes", genes.shape, "hash metadata", meta.shape)

    p["col_index"] = np.arange(len(p), dtype=np.int64)          # 0-based column index in the matrix
    p["hash_assigned"] = p.cell_type.notna()
    p["is_24h"] = p.time_point == cfg["time_point_h"]
    rW, rP = p.top_to_second_best_ratio_W.fillna(0), p.top_to_second_best_ratio_P.fillna(0)
    p["hash_pass"] = ((p.qval_W.fillna(1) < q["qval_W_max"]) & (p.qval_P.fillna(1) < q["qval_P_max"])
                      & (rW >= q["top_to_second_best_ratio_W_min"]) & (rP >= q["top_to_second_best_ratio_P_min"])
                      & (p["n.umi"] >= q["n_umi_min"]))
    p["keep"] = p.hash_assigned & p.is_24h & p.hash_pass
    log("assigned", int(p.hash_assigned.sum()), "24h", int((p.hash_assigned & p.is_24h).sum()),
        "keep", int(p.keep.sum()))
    cols = ["cell", "col_index", "n.umi", "hash_umis_W", "qval_W", "top_to_second_best_ratio_W", "hash_umis_P",
            "qval_P", "top_to_second_best_ratio_P", "pcr_plate", "rt_plate", "lig_plate", "culture_plate", "Combo",
            "well_oligo", "plate_oligo", "cell_type", "replicate", "time_point", "catalog_number", "product_name",
            "vehicle", "dose", "pathway_level_1", "hash_assigned", "is_24h", "hash_pass", "keep"]
    p[cols].to_parquet(f"{out}/cells_all.parquet", index=False)

    # ---------------------------------------------------------------- well map (24 h)
    m = meta[meta.time_point == cfg["time_point_h"]].copy()
    a = p[p.hash_assigned & p.is_24h]
    g_all = a.groupby("Combo").size().rename("n_hashed")
    k = a[a.keep]
    g = k.groupby("Combo").agg(n_qc=("cell", "size"), median_umi_qc=("n.umi", "median"))
    cp = a.groupby("Combo").culture_plate.agg(lambda s: ",".join(sorted(s.dropna().unique())))
    w = m.merge(g_all, left_on="Combo", right_index=True, how="left").merge(g, left_on="Combo", right_index=True,
                                                                          how="left")
    w = w.merge(cp.rename("culture_plate_pdata"), left_on="Combo", right_index=True, how="left")
    w[["n_hashed", "n_qc"]] = w[["n_hashed", "n_qc"]].fillna(0).astype(int)
    w["frac_hash_pass"] = (w.n_qc / w.n_hashed.replace(0, np.nan)).round(4)
    names = a.drop_duplicates("catalog_number").set_index("catalog_number")
    w["product_name"] = w.catalog_number.map(names.product_name)
    w["pathway_level_1"] = w.catalog_number.map(names.pathway_level_1)
    w = w.rename(columns={"cell_type": "line", "plate_oligo": "culture_plate"})
    w["drug"] = w.catalog_number
    w = w[["line", "replicate", "time_point", "culture_plate", "culture_plate_pdata", "well_oligo", "Combo", "drug",
           "product_name", "pathway_level_1", "vehicle", "dose", "n_hashed", "n_qc", "frac_hash_pass",
           "median_umi_qc", "SMILES"]].sort_values(["line", "replicate", "culture_plate", "drug", "dose"])
    w.to_csv(f"{out}/wells.tsv", sep="\t", index=False)
    log("wells", len(w), "plate oligo == pData culture_plate:",
        bool((w.culture_plate == w.culture_plate_pdata).where(w.n_hashed > 0, True).all()))

    # ---------------------------------------------------------------- counts
    human = genes.id.str.startswith("ENSG").values
    gene_map = np.full(len(genes) + 1, -1, dtype=np.int64)
    gene_map[np.where(human)[0] + 1] = np.arange(human.sum())
    genes[human].to_csv(f"{out}/genes_human.tsv", sep="\t", index=False)
    kept = p[p.keep].reset_index(drop=True)
    cell_map = np.full(len(p) + 1, -1, dtype=np.int64)
    cell_map[kept.col_index.values + 1] = np.arange(len(kept))
    path = f"{raw}/{PFX}A549_MCF7_K562_screen_UMI.count.matrix.gz"
    proc = subprocess.Popen(["pigz", "-dc", "-p", "4", path], stdout=subprocess.PIPE)
    R, C, D = [], [], []
    n_lines = n_kept = 0
    max_count = 0
    for ch in pd.read_csv(proc.stdout, sep="\t", header=None, dtype=np.int64, chunksize=100_000_000,
                          engine="c"):
        v = ch.values
        assert v[:, 0].min() >= 1 and v[:, 0].max() <= len(genes) and v[:, 1].min() >= 1 and v[:, 1].max() <= len(p)
        n_lines += len(v)
        gi, ci = gene_map[v[:, 0]], cell_map[v[:, 1]]
        sel = (gi >= 0) & (ci >= 0)
        R.append(ci[sel].astype(np.int32)); C.append(gi[sel].astype(np.int32)); D.append(v[sel, 2].astype(np.int32))
        n_kept += int(sel.sum()); max_count = max(max_count, int(v[:, 2].max()))
        log("lines", n_lines, "kept", n_kept)
    assert proc.wait() == 0
    R, C, D = np.concatenate(R), np.concatenate(C), np.concatenate(D)
    X = sp.csr_matrix((D, (R, C)), shape=(len(kept), int(human.sum())), dtype=np.int32)
    del R, C, D
    X.sum_duplicates()
    log("matrix", X.shape, "nnz", X.nnz, "max count", max_count)
    tot = np.asarray(X.sum(1)).ravel()
    kept["n_umi_human"] = tot
    summary = dict(n_matrix_lines=n_lines, n_nnz_kept=int(X.nnz), max_count=max_count, n_cells_pdata=len(p),
                   n_hash_assigned=int(p.hash_assigned.sum()), n_assigned_24h=int((p.hash_assigned & p.is_24h).sum()),
                   n_keep=int(len(kept)), n_genes_total=len(genes), n_genes_human=int(human.sum()), lines={})
    for line, idx in kept.groupby("cell_type").groups.items():
        idx = np.sort(np.asarray(idx))
        sp.save_npz(f"{out}/counts_{line}.npz", X[idx], compressed=False)
        o = kept.loc[idx, ["cell", "col_index", "n.umi", "n_umi_human", "Combo", "culture_plate", "replicate",
                           "catalog_number", "product_name", "vehicle", "dose", "pcr_plate", "rt_plate"]]
        o.rename(columns={"catalog_number": "drug"}).to_parquet(f"{out}/obs_{line}.parquet", index=False)
        summary["lines"][line] = dict(n_cells=int(len(idx)), median_umi_human=float(np.median(tot[idx])))
        log(line, len(idx))
    json.dump(summary, open(f"{out}/b0_summary.json", "w"), indent=1)
    log("done", json.dumps(summary))


if __name__ == "__main__":
    main()
