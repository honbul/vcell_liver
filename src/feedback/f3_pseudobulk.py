#!/usr/bin/env python
"""F3 step 2: per-line pseudobulk sufficient statistics for 4-line CRISPRi LOCO.

Streams the dense raw-count matrix of one source h5ad (read-only) in row blocks with h5py,
normalises each cell to 1e4 total UMIs (obs UMI_count, i.e. the whole-transcriptome library,
not the row sum of the author-filtered panel), applies log1p, and accumulates per target-gene
group (including the non-targeting control group):
  n, sum(log), sum(log^2), sum(linear normalised), sum(log) for random cell half 0,
  target x gem_group cell counts, per gem_group control sums, control detection counts,
  per-guide (gene_transcript) cell counts and self-target expression.
Nothing is loaded densely beyond one block. Output: <out>/<line>.npz + <line>_qc.json.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

LINES = {
    "K562": "/mnt/S1/sdata/pubdata/Virtual_Cell/Replogle2022/K562_essential_raw_singlecell_01.h5ad",
    "RPE1": "/mnt/S1/sdata/pubdata/Virtual_Cell/Replogle2022/rpe1_raw_singlecell_01.h5ad",
    "HepG2": "/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE264667/GSE264667_hepg2_raw_singlecell_01.h5ad",
    "Jurkat": "/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE264667/GSE264667_jurkat_raw_singlecell_01.h5ad",
}
CONTROL = "non-targeting"
TARGET_SUM = 1e4


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)


def read_cat(f, col):
    codes = f[f"obs/{col}"][:]
    cats = _dec(f[f"obs/__categories/{col}"][:])
    return codes, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True, choices=sorted(LINES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()
    t0 = time.time()
    path = LINES[args.line]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as f:
        ncell, ngene = f["X"].shape
        var_ids = _dec(f["var/gene_id"][:])
        gn_codes = f["var/gene_name"][:]
        gn_cats = _dec(f["var/__categories/gene_name"][:])
        var_names = gn_cats[gn_codes]
        g_codes, g_cats = read_cat(f, "gene")
        gid_codes, gid_cats = read_cat(f, "gene_id")
        gt_codes, gt_cats = read_cat(f, "gene_transcript")
        gem = f["obs/gem_group"][:].astype(np.int64)
        umi = f["obs/UMI_count"][:].astype(np.float64)
        mito = f["obs/mitopercent"][:].astype(np.float64)

        tsym = g_cats[g_codes]
        tgid = gid_cats[gid_codes]
        guide = gt_cats[gt_codes]
        groups, grp = np.unique(tsym, return_inverse=True)
        G = len(groups)
        # Ensembl ID per target symbol (most frequent)
        group_gid = []
        for gi in range(G):
            ids, cnt = np.unique(tgid[grp == gi], return_counts=True)
            group_gid.append(ids[np.argmax(cnt)])
        group_gid = np.array(group_gid, dtype=object)
        gems, gem_idx = np.unique(gem, return_inverse=True)
        guides, guide_idx = np.unique(guide, return_inverse=True)
        guide_group = np.zeros(len(guides), dtype=np.int64)
        guide_group[guide_idx] = grp
        ctrl_g = int(np.where(groups == CONTROL)[0][0])
        is_ctrl = grp == ctrl_g
        rng = np.random.default_rng(args.seed + int(hashlib.md5(args.line.encode()).hexdigest()[:6], 16))
        half = rng.integers(0, 2, ncell)

        # self-target column index per group (by Ensembl ID, fallback symbol)
        id2col = {g: i for i, g in enumerate(var_ids)}
        name2col = {g: i for i, g in enumerate(var_names)}
        self_col = np.array([id2col.get(group_gid[i], name2col.get(groups[i], -1)) for i in range(G)])
        cell_self_col = self_col[grp]

        S = np.zeros((G, ngene)); SS = np.zeros((G, ngene)); SL = np.zeros((G, ngene)); S0 = np.zeros((G, ngene))
        n = np.bincount(grp, minlength=G).astype(np.int64)
        n0 = np.bincount(grp[half == 0], minlength=G).astype(np.int64)
        n_gem = np.zeros((G, len(gems)), dtype=np.int64)
        np.add.at(n_gem, (grp, gem_idx), 1)
        cS = np.zeros((len(gems), ngene)); cSS = np.zeros((len(gems), ngene))
        c_nnz = np.zeros(ngene, dtype=np.int64)
        guide_n = np.bincount(guide_idx, minlength=len(guides)).astype(np.int64)
        guide_self_lin = np.zeros(len(guides)); guide_self_log = np.zeros(len(guides))
        rowsum = np.zeros(ncell)
        all_int = True; xmin = np.inf; xmax = -np.inf

        for s in range(0, ncell, args.block):
            e = min(ncell, s + args.block)
            X = f["X"][s:e].astype(np.float64)
            b = e - s
            rowsum[s:e] = X.sum(1)
            if all_int and not np.all(X == np.round(X)):
                all_int = False
            xmin = min(xmin, X.min()); xmax = max(xmax, X.max())
            lin = X * (TARGET_SUM / umi[s:e, None])
            del X
            L = np.log1p(lin)
            gb = grp[s:e]
            M = sp.csr_matrix((np.ones(b), (gb, np.arange(b))), shape=(G, b))
            S += M @ L; SS += M @ (L * L); SL += M @ lin
            h0 = half[s:e] == 0
            M0 = sp.csr_matrix((np.ones(h0.sum()), (gb[h0], np.where(h0)[0])), shape=(G, b))
            S0 += M0 @ L
            cm = is_ctrl[s:e]
            if cm.any():
                Mc = sp.csr_matrix((np.ones(cm.sum()), (gem_idx[s:e][cm], np.where(cm)[0])), shape=(len(gems), b))
                cS += Mc @ L; cSS += Mc @ (L * L)
                c_nnz += (lin[cm] > 0).sum(0)
            sc = cell_self_col[s:e]
            ok = sc >= 0
            vals = np.full(b, np.nan)
            vals[ok] = lin[np.where(ok)[0], sc[ok]]
            gi = guide_idx[s:e]
            np.add.at(guide_self_lin, gi[ok], vals[ok])
            np.add.at(guide_self_log, gi[ok], np.log1p(vals[ok]))
            print(f"{args.line} rows {e}/{ncell} {time.time() - t0:.0f}s", flush=True)

    # per-target knockdown: residual self-expression relative to batch-matched control (linear scale)
    ctrl_self_lin = np.full(G, np.nan)
    ctrl_lin_mean = SL[ctrl_g] / n[ctrl_g]
    for gi in range(G):
        if self_col[gi] >= 0:
            ctrl_self_lin[gi] = ctrl_lin_mean[self_col[gi]]
    np.savez(
        out / f"{args.line}.npz",
        groups=groups.astype(str), group_gid=group_gid.astype(str), ctrl_index=ctrl_g,
        var_ids=var_ids.astype(str), var_names=var_names.astype(str), self_col=self_col,
        n=n, n0=n0, n_gem=n_gem, gems=gems, S=S, SS=SS, SL=SL, S0=S0, cS=cS, cSS=cSS,
        c_nnz=c_nnz, guides=guides.astype(str), guide_group=guide_group, guide_n=guide_n,
        guide_self_lin=guide_self_lin, guide_self_log=guide_self_log, ctrl_self_lin=ctrl_self_lin,
    )
    ratio = rowsum / umi
    tmask = ~is_ctrl
    per_target_umi = np.array([np.median(umi[grp == gi]) for gi in range(G)])
    qc = {
        "line": args.line, "path": path, "n_cells": int(ncell), "n_genes_panel": int(ngene),
        "raw_integer_counts": bool(all_int), "x_min": float(xmin), "x_max": float(xmax),
        "library_size_used": "obs UMI_count (whole-transcriptome UMIs)",
        "rowsum_over_UMI_count_pct": np.percentile(ratio, [0, 1, 50, 99, 100]).round(4).tolist(),
        "umi_count_median_control": float(np.median(umi[is_ctrl])),
        "umi_count_median_perturbed": float(np.median(umi[tmask])),
        "umi_count_pct_all": np.percentile(umi, [1, 50, 99]).round(1).tolist(),
        "mitopercent_median": float(np.median(mito)),
        "per_target_median_umi_range": [float(np.min(per_target_umi)), float(np.max(per_target_umi))],
        "n_groups_incl_control": int(G), "n_controls": int(n[ctrl_g]), "n_gem_groups": int(len(gems)),
        "half_split_seed": int(args.seed), "block_rows": args.block, "runtime_s": round(time.time() - t0, 1),
    }
    (out / f"{args.line}_qc.json").write_text(json.dumps(qc, indent=1))
    print(json.dumps(qc, indent=1))


if __name__ == "__main__":
    main()
