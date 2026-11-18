"""Split-half and per-well response material for the positive control (plan 5.4, R6-A4).

Only wells of the 209 TRAIN drug groups are read (the positive control lives on source held-fold
drugs). Validation and test wells are not opened.

Treated wells (public Tahoe-100M parquet mirror, read-only; same normalisation as the truth:
mean over cells of log1p(1e4 * count / library), library over all genes):
    for every (sample = well, cell line) of a TRAIN drug at 0.5 uM
        half A = cells with blake2b(barcode) <  0.5,  half B = the rest   (barcode-disjoint)
        stores mean_A, mean_B, mean_full [G] and the three cell counts
Control pools (the original R6 raw-count cache, read-only): the REFERENCE-role cells of every pool
(the cells behind the truth's control mean; evaluation- and fit-input cells excluded) are split by
alternating barcode-hash rank into two disjoint halves -> ref_A, ref_B, ref_full [n_pool, G].

A split half is a technical split of one well, NOT an independent culture replicate. Wells of the
same drug on different plates are the only replicate-like units in this panel.

Cross-checks written to the receipt: the full-well means reproduce the v2 truth (Y_v2) through the
plate weights; ref_full reproduces the contract's reference mean.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

from src.depth_robust_r6 import data as D
from src.depth_robust_r6.build_count_cache import NG, bhash, shard_samples

from . import common as C

COLS = ["genes", "expressions", "sample", "cell_line_id", "BARCODE_SUB_LIB_ID"]
_G: dict = {}


def read_shard(path: str):
    samples, lines, panel_tok, key = _G["samples"], _G["lines"], _G["panel_tok"], _G["key"]
    t = pq.read_table(path, columns=COLS)
    n_total = t.num_rows
    t = t.filter(pc.is_in(t.column("sample"), value_set=pa.array(samples)))
    t = t.filter(pc.is_in(t.column("cell_line_id"), value_set=pa.array(lines)))
    n = t.num_rows
    rec = dict(shard=os.path.basename(path), bytes=os.path.getsize(path), n_rows=n_total, n_kept=n)
    if n == 0:
        return rec, None
    g = t.column("genes").combine_chunks(); e = t.column("expressions").combine_chunks()
    off = g.offsets.to_numpy(); off = off - off[0]
    gv = g.flatten().to_numpy(); ev = e.flatten().to_numpy()
    assert np.all(gv[off[:-1]] < 3) and int((gv < 3).sum()) == n, f"special-token layout in {path}"
    keep = np.ones(len(gv), bool); keep[off[:-1]] = False
    ev, gv = ev[keep], gv[keep]
    rec["n_non_integer"] = int((ev != np.rint(ev)).sum())
    X = sp.csr_matrix((ev.astype(np.float64), gv.astype(np.int32), off - np.arange(n + 1)), shape=(n, NG))
    X.sum_duplicates()
    lib = np.asarray(X.sum(axis=1)).ravel()
    P = X[:, panel_tok].tocsr()
    P.data = np.log1p(P.data * np.repeat(1e4 / np.maximum(lib, 1), np.diff(P.indptr)))
    smp = t.column("sample").to_pylist(); cl = t.column("cell_line_id").to_pylist(); bc = t.column("BARCODE_SUB_LIB_ID").to_pylist()
    wid = np.array([key.get((s, c), -1) for s, c in zip(smp, cl)])
    half = np.array([bhash(b) >= 0.5 for b in bc]).astype(int)
    ok = wid >= 0
    gid = wid[ok] * 2 + half[ok]
    u, inv = np.unique(gid, return_inverse=True)
    S = sp.csr_matrix((np.ones(ok.sum()), (inv, np.arange(ok.sum()))), shape=(len(u), int(ok.sum()))) @ P[ok]
    return rec, (u, np.asarray(S.todense()), np.bincount(inv, minlength=len(u)))


def control_halves(d, out: Path) -> dict:
    root = C.orig_assets() / "cache" / "dmso_counts_v1"
    meta = pd.read_csv(root / "cells.tsv", sep="\t")
    Cn = sp.load_npz(root / "panel_counts.npz").tocsr().astype(np.float64)
    lib = meta.library_size.to_numpy().astype(np.float64)
    Cn.data = np.log1p(Cn.data * np.repeat(1e4 / lib, np.diff(Cn.indptr)))
    G, n_pool = Cn.shape[1], len(d.pools)
    ref = np.zeros((3, n_pool, G), np.float32); n = np.zeros((3, n_pool), int)
    grp = {k: g.index.to_numpy() for k, g in meta.groupby(["plate_index", "line"], sort=True)}
    for r in d.pools.itertuples():
        idx = grp[(r.plate_index, r.line)]                          # already in barcode-hash order
        assert len(idx) == r.n_total
        rr = idx[int(r.n_total - r.n_reference):]
        for j, sel in enumerate((rr[0::2], rr[1::2], rr)):
            ref[j, r.pool_id] = np.asarray(Cn[sel].mean(axis=0)).ravel(); n[j, r.pool_id] = len(sel)
    np.save(out / "ref_halves.npy", ref)
    rm = np.load(C.orig_assets() / "cache" / "contract_v1" / "ref_mean.npy")
    worst = max(float(np.abs(ref[2, r.pool_id] - rm[r.plate_index, r.line]).max()) for r in d.pools.itertuples())
    return dict(n_pools=n_pool, split="reference-role cells, alternating barcode-hash rank", n_ref_half_A=pd.Series(n[0]).describe().to_dict(),
                max_abs_diff_ref_full_vs_contract=worst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=40)
    a = ap.parse_args()
    t0 = time.time()
    out = C.assets() / "cache" / "halves_v1"
    out.mkdir(parents=True, exist_ok=False)                           # immutable: never overwrite a cache
    d = C.load(eval_views=False)
    rec = dict(control=control_halves(d, out))
    print(json.dumps(rec, default=float), flush=True)

    wells = pd.read_csv(Path(os.environ["R3_CACHE"]) / "wells.tsv", sep="\t")
    tr = set(C.train_drugs(d))
    w = wells[(wells.kind == "treated") & wells.drug.isin(tr)].reset_index(drop=True)
    assert set(w.role) == {"train"}, set(w.role)
    w["wid"] = np.arange(len(w))
    panel = pd.read_csv("manifests/chemcpa_moe_v2/genes.tsv", sep="\t").sort_values("panel_index")
    samples = sorted(w["sample"].unique())
    _G.update(samples=samples, lines=sorted(w.cell_line_id.unique()), panel_tok=panel.token_id.to_numpy(),
              key={(s, c): i for s, c, i in zip(w["sample"], w.cell_line_id, w.wid)})
    shards = sorted(str(p) for p in (Path(os.environ["TAHOE_RAW"]) / "data").glob("train-*.parquet"))
    with mp.Pool(a.procs) as pool:
        scan = pool.map(shard_samples, shards, chunksize=16)
    use = [p for p, lo, hi, _ in scan if any(lo <= s <= hi for s in samples)]
    print(f"footer scan {time.time()-t0:.0f}s: {len(shards)} shards; {len(use)} overlap the {len(samples)} TRAIN-drug samples", flush=True)
    G = len(panel)
    S = np.zeros((len(w) * 2, G)); N = np.zeros(len(w) * 2, int)
    recs = []
    with mp.Pool(min(a.procs, len(use))) as pool:
        for i, (r, res) in enumerate(pool.imap_unordered(read_shard, use, chunksize=1)):
            recs.append(r)
            if res is not None:
                u, s, n = res
                S[u] += s; N[u] += n
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(use)} shards, {int(N.sum()):,} cells, {time.time()-t0:.0f}s", flush=True)
    assert sum(r.get("n_non_integer", 0) for r in recs) == 0
    S = S.reshape(len(w), 2, G); N = N.reshape(len(w), 2)
    mean = np.zeros((3, len(w), G), np.float32)
    mean[0] = S[:, 0] / np.maximum(N[:, 0], 1)[:, None]; mean[1] = S[:, 1] / np.maximum(N[:, 1], 1)[:, None]
    mean[2] = S.sum(1) / np.maximum(N.sum(1), 1)[:, None]
    w["n_A"], w["n_B"], w["n_full"] = N[:, 0], N[:, 1], N.sum(1)
    np.save(out / "treated_halves.npy", mean)
    w.to_csv(out / "wells.tsv", sep="\t", index=False)

    # ---- cross-check: full-well means reproduce the v2 truth through the plate weights -----------
    V2 = Path(os.environ["R6_V2DATA"])
    Yv2, valid2, pw, dm = np.load(V2 / "Y.npy"), np.load(V2 / "valid.npy"), C.plate_weights(), np.load(V2 / "dmso_plate_mean.npy")
    diffs, n_chk, n_well_mismatch = [], 0, 0
    for (l, g), ww in w[w.n_full > 0].groupby(["line", "drug"]):
        if not valid2[l, g]:
            continue
        use_w = ww[ww.valid] if "valid" in ww and ww.valid.any() else ww
        pl = use_w.plate_index.to_numpy()
        frac = np.bincount(pl, minlength=pw.shape[2]) / len(pl)
        if not np.allclose(frac, pw[l, g], atol=1e-3):
            n_well_mismatch += 1
            continue
        y = (mean[2, use_w.wid.to_numpy()] - dm[pl, l]).mean(0)
        diffs.append(float(np.abs(y - Yv2[l, g]).max())); n_chk += 1
    rec.update(treated=dict(n_wells=int(len(w)), n_samples=len(samples), n_train_drugs=int(w.drug.nunique()), n_shards_total=len(shards),
                            n_shards_read=len(use), n_cells=int(N.sum()), cells_per_well=pd.Series(N.sum(1)).describe().to_dict(),
                            cells_per_half=pd.Series(N.ravel()).describe().to_dict(), n_wells_without_cells=int((N.sum(1) == 0).sum()),
                            half_rule="A: blake2b(barcode) < 0.5; B: otherwise"),
               cross_check_v2_truth=dict(n_conditions_compared=n_chk, n_conditions_plate_weight_mismatch=n_well_mismatch,
                                         max_abs_diff=float(np.max(diffs)) if diffs else None,
                                         median_abs_max_diff=float(np.median(diffs)) if diffs else None,
                                         frac_within_1e_4=float(np.mean(np.asarray(diffs) < 1e-4)) if diffs else None),
               files={f: dict(bytes=os.path.getsize(out / f), sha256=D.sha256_file(out / f)) for f in ["ref_halves.npy", "treated_halves.npy", "wells.tsv"]},
               shards=recs, cache_dir=str(out), elapsed_s=round(time.time() - t0, 1), commit=C.git_commit(),
               exposure="TRAIN drug wells only; validation and test wells were not read")
    C.dump_json(rec, "manifests/r6_followup_v1/halves_receipt.json")
    print(json.dumps({k: v for k, v in rec.items() if k != "shards"}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
