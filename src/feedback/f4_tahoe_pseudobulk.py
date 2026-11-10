"""F4: per-well pseudobulk sums for one Tahoe-100M cell line (raw-count h5ad, read-only).

One streaming pass over the CSR count matrix accumulates summed raw counts for
  * every sample (= one drug x dose well on one plate, DMSO wells included), and
  * random DMSO "pseudo-wells": for each plate the DMSO cells (both DMSO wells) are
    randomly partitioned into B buckets for several B (size-matched null).
Optional per-cell QC parquet (from f4_tahoe_metadata.py) restricts to pass_filter == "full".

Output (assets): <prefix>_pseudobulk.npz  (sums float32 [n_groups, n_genes], group table, gene names)
Heavy: run through Slurm.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

BUCKETS = (2, 4, 8, 16, 32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--cell-qc", default=None, help="parquet with BARCODE_SUB_LIB_ID, pass_filter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    h = h5py.File(args.h5ad, "r")
    from anndata.experimental import read_elem
    obs = read_elem(h["obs"])
    var_names = read_elem(h["var"]).index.astype(str).to_numpy()
    n_cells, n_genes = h["X"].attrs["shape"]
    for c in ["sample", "plate", "drug"]:
        obs[c] = obs[c].astype(str)
    obs["is_dmso"] = obs.drug.str.startswith("DMSO")
    keep = np.ones(n_cells, dtype=bool)
    qc_info = {}
    if args.cell_qc:
        qc = pd.read_parquet(args.cell_qc, columns=["BARCODE_SUB_LIB_ID", "pass_filter"])
        pf = obs.BARCODE_SUB_LIB_ID.astype(str).map(qc.set_index("BARCODE_SUB_LIB_ID").pass_filter.astype(str))
        qc_info = {"h5ad_cells_by_pass_filter": pf.fillna("NOT_IN_OBS_METADATA").value_counts().to_dict()}
        keep = (pf == "full").to_numpy()
    print("cells", n_cells, "kept", int(keep.sum()), qc_info, flush=True)

    samples = np.array(sorted(obs["sample"].unique()))
    s_idx = pd.Series(np.arange(len(samples)), index=samples)
    g_sample = s_idx.loc[obs["sample"]].to_numpy()
    groups = [dict(group=s, kind="sample", plate=None, B=0, bucket=-1) for s in samples]
    smp_plate = obs.groupby("sample").plate.first()
    for g in groups:
        g["plate"] = smp_plate[g["group"]]
    # DMSO buckets per plate (kept cells only)
    bucket_cols = {}
    for B in BUCKETS:
        col = np.full(n_cells, -1, dtype=np.int64)
        for plate in sorted(obs.plate.unique()):
            idx = np.where(obs.is_dmso.to_numpy() & (obs.plate.to_numpy() == plate) & keep)[0]
            lab = rng.permutation(np.arange(len(idx)) % B)
            for b in range(B):
                gid = len(groups)
                groups.append(dict(group=f"{plate}|B{B}|b{b}", kind="dmso_bucket", plate=plate, B=B, bucket=b))
                col[idx[lab == b]] = gid
        bucket_cols[B] = col
    n_groups = len(groups)
    sums = np.zeros((n_groups, n_genes), dtype=np.float64)
    n_in = np.zeros(n_groups, dtype=np.int64)
    umi = np.zeros(n_groups, dtype=np.float64)

    X = h["X"]
    indptr = X["indptr"][:]
    for a in range(0, n_cells, args.chunk):
        b = min(a + args.chunk, n_cells)
        lo, hi = indptr[a], indptr[b]
        data = X["data"][lo:hi]
        ind = X["indices"][lo:hi]
        m = sp.csr_matrix((data, ind, indptr[a:b + 1] - lo), shape=(b - a, n_genes))
        k = keep[a:b]
        rows, cols = [], []
        r = np.arange(b - a)
        rows.append(g_sample[a:b][k]); cols.append(r[k])
        for B in BUCKETS:
            gb = bucket_cols[B][a:b]
            sel = gb >= 0
            rows.append(gb[sel]); cols.append(r[sel])
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        ind_m = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_groups, b - a))
        sums += (ind_m @ m).toarray()
        n_in += np.bincount(rows, minlength=n_groups)
        umi += ind_m @ np.asarray(m.sum(axis=1)).ravel()
        if (a // args.chunk) % 10 == 0:
            print(f"{b:,}/{n_cells:,} cells {time.time()-t0:.0f}s", flush=True)

    gt = pd.DataFrame(groups)
    gt["n_cells"] = n_in
    gt["total_umi"] = umi
    smp_drug = obs.groupby("sample").drug.first()
    gt["drug"] = gt.group.map(smp_drug).fillna("DMSO_TF")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, sums=sums.astype(np.float32), genes=var_names,
                        groups=gt.to_json(orient="records"))
    meta = {"h5ad": args.h5ad, "n_cells": int(n_cells), "n_kept": int(keep.sum()), **qc_info,
            "n_groups": n_groups, "buckets": BUCKETS, "seed": args.seed, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, default=str), flush=True)


if __name__ == "__main__":
    main()
