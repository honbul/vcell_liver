"""R6 raw-count receipt: read the DMSO cells' integer UMI counts straight from the Tahoe shards.

Why a new pass.  Every single-cell array earlier studies kept is log1p(CP10k) on the 2,000-gene
panel. Binomial thinning is only defined on the integer counts, and the library size must include
the genes outside the panel, so the public parquet shards are read again (read-only) for the DMSO
wells of the five plates that carry the 0.5 uM treated wells. Treated cells are NOT read: the
response truth is fixed at full observation and is never thinned.

For every DMSO cell of the 50 study cell lines this writes
    cells.tsv          barcode, sample, plate, cell line, library size (all genes), panel sum,
                       C_rest = library - panel sum, barcode hash u
    panel_counts.npz   CSR int32 [n_cells, 2000] in panel order (kept on the server only)
and a receipt with the integer check, shard list with sha256, and two cross-checks against the
earlier pipeline (R3's sampled cells; the v2 full-pool DMSO mean).
"""
from __future__ import annotations

import argparse
import hashlib
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

NG = 62713
COLS = ["genes", "expressions", "sample", "cell_line_id", "BARCODE_SUB_LIB_ID", "plate"]
_G: dict = {}


def bhash(s: str) -> float:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") / 2.0 ** 64


def sha256_file(path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def shard_samples(path: str):
    """Min/max of `sample` over the shard, from the parquet footer only."""
    md = pq.ParquetFile(path).metadata
    j = [i for i in range(md.num_columns) if md.row_group(0).column(i).path_in_schema == "sample"][0]
    lo = min(md.row_group(r).column(j).statistics.min for r in range(md.num_row_groups))
    hi = max(md.row_group(r).column(j).statistics.max for r in range(md.num_row_groups))
    return path, lo, hi, md.num_rows


def read_shard(path: str):
    samples, lines, panel_tok = _G["samples"], _G["lines"], _G["panel_tok"]
    t = pq.read_table(path, columns=COLS)
    n_total = t.num_rows
    t = t.filter(pc.is_in(t.column("sample"), value_set=pa.array(samples)))
    n_dmso_any_line = t.num_rows
    t = t.filter(pc.is_in(t.column("cell_line_id"), value_set=pa.array(lines)))
    n = t.num_rows
    rec = dict(shard=os.path.basename(path), bytes=os.path.getsize(path), sha256=sha256_file(path),
               n_rows=n_total, n_dmso_rows=n_dmso_any_line, n_kept=n)
    if n == 0:
        return rec, None
    g = t.column("genes").combine_chunks()
    e = t.column("expressions").combine_chunks()
    off = g.offsets.to_numpy()
    off = off - off[0]
    gv = g.flatten().to_numpy()
    ev = e.flatten().to_numpy()
    assert np.all(gv[off[:-1]] < 3) and int((gv < 3).sum()) == n, f"special-token layout in {path}"
    keep = np.ones(len(gv), bool)
    keep[off[:-1]] = False                                  # drop the leading special token
    ev, gv = ev[keep], gv[keep]
    # the integer check the whole study rests on
    rec.update(expr_dtype=str(e.type.value_type), n_values=int(len(ev)),
               n_non_integer=int((ev != np.rint(ev)).sum()), n_negative=int((ev < 0).sum()),
               n_zero_stored=int((ev == 0).sum()), max_count=float(ev.max()))
    X = sp.csr_matrix((ev.astype(np.int64), gv.astype(np.int32), off - np.arange(n + 1)), shape=(n, NG))
    X.sum_duplicates()
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.int64)
    P = X[:, panel_tok].tocsr().astype(np.int32)
    P.eliminate_zeros()
    meta = pd.DataFrame(dict(
        barcode=t.column("BARCODE_SUB_LIB_ID").to_pylist(), sample=t.column("sample").to_pylist(),
        cell_line_id=t.column("cell_line_id").to_pylist(), plate=t.column("plate").to_pylist(),
        library_size=lib, panel_sum=np.asarray(P.sum(axis=1)).ravel().astype(np.int64),
        n_genes_all=np.diff(X.indptr).astype(np.int32), n_genes_panel=np.diff(P.indptr).astype(np.int32)))
    return rec, (meta, P)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tahoe", default=os.environ.get("TAHOE_RAW"))
    ap.add_argument("--r3-cache", default=os.environ.get("R3_CACHE"))
    ap.add_argument("--v2-data", default=os.environ.get("R6_V2DATA"))
    ap.add_argument("--manifest-dir", default="manifests/chemcpa_moe_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--receipt", default="manifests/depth_robust_r6/raw_count_receipt.json")
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)                 # immutable: never overwrite a cache
    M = Path(a.manifest_dir)
    wells = pd.read_csv(Path(a.r3_cache) / "wells.tsv", sep="\t")
    ctrl = wells[wells.kind == "ctrl"]
    samples = sorted(ctrl["sample"].unique())
    lines_t = pd.read_csv(M / "lines.tsv", sep="\t")
    plates_t = pd.read_csv(M / "plates.tsv", sep="\t")
    panel = pd.read_csv(M / "genes.tsv", sep="\t").sort_values("panel_index")
    _G.update(samples=samples, lines=sorted(lines_t.cell_line_id), panel_tok=panel.token_id.to_numpy())
    print(f"DMSO samples {samples}; {len(_G['lines'])} lines; panel {len(panel)}", flush=True)

    shards = sorted(str(p) for p in (Path(a.tahoe) / "data").glob("train-*.parquet"))
    with mp.Pool(a.procs) as pool:
        scan = pool.map(shard_samples, shards, chunksize=16)
    use = [p for p, lo, hi, _ in scan if any(lo <= s <= hi for s in samples)]
    print(f"footer scan {time.time()-t0:.0f}s: {len(shards)} shards, {sum(r[3] for r in scan):,} rows; "
          f"{len(use)} shards overlap the DMSO samples", flush=True)
    with mp.Pool(min(a.procs, len(use))) as pool:
        res = pool.map(read_shard, use, chunksize=1)
    recs = [r for r, _ in res]
    meta = pd.concat([d[0] for _, d in res if d is not None], ignore_index=True)
    P = sp.vstack([d[1] for _, d in res if d is not None]).tocsr()
    assert meta.barcode.is_unique, "duplicate barcode among DMSO cells"
    n_nonint = sum(r.get("n_non_integer", 0) for r in recs)
    n_neg = sum(r.get("n_negative", 0) for r in recs)
    assert n_nonint == 0 and n_neg == 0, f"raw values are not non-negative integers: {n_nonint} {n_neg}"

    meta["line"] = meta.cell_line_id.map(dict(zip(lines_t.cell_line_id, lines_t.line_index))).astype(int)
    meta["plate_index"] = meta.plate.map(dict(zip(plates_t.plate, plates_t.plate_index))).astype(int)
    meta["c_rest"] = meta.library_size - meta.panel_sum
    meta["u"] = [bhash(b) for b in meta.barcode]
    order = np.lexsort((meta.u.to_numpy(), meta.line.to_numpy(), meta.plate_index.to_numpy()))
    meta = meta.iloc[order].reset_index(drop=True)
    P = P[order]
    assert (meta.c_rest >= 0).all()

    # ---- cross-check 1: R3's sampled DMSO cells (same barcodes) --------------------------------
    cz = np.load(Path(a.r3_cache) / "cells.npz")
    X3 = np.load(Path(a.r3_cache) / "X.npy", mmap_mode="r")
    pos = {}
    for r in ctrl.itertuples():
        pos[(r.sample, r.cell_line_id)] = (r.start, r.end)
    chk = dict(n_r3_ctrl_cells=int(ctrl.n_cells.sum()), n_groups=int(len(ctrl)))
    max_abs, n_cmp, lib_bad = 0.0, 0, 0
    key = meta["sample"] + "|" + meta.cell_line_id
    for (s, c), (st, en) in pos.items():
        idx = np.flatnonzero((key == s + "|" + c).to_numpy())[: en - st]   # lowest-hash cells of the group
        if len(idx) != en - st:
            chk.setdefault("groups_short", []).append([s, c, int(len(idx)), int(en - st)])
            continue
        cnt = np.asarray(P[idx].todense(), dtype=np.float64)
        lib = meta.library_size.to_numpy()[idx].astype(np.float64)
        ln = np.log1p(1e4 * cnt / lib[:, None]).astype(np.float32)
        max_abs = max(max_abs, float(np.abs(ln - np.asarray(X3[st:en])).max()))
        lib_bad += int((np.abs(lib - cz["library_size"][st:en]) > 0.5).sum())
        n_cmp += len(idx)
    chk.update(n_cells_compared=n_cmp, max_abs_diff_log1p_cp10k=max_abs, n_library_size_mismatch=lib_bad)

    # ---- cross-check 2: the v2 full-pool DMSO mean ----------------------------------------------
    dm = np.load(Path(a.v2_data) / "dmso_plate_mean.npy")
    worst, n_pool = 0.0, 0
    libf = meta.library_size.to_numpy().astype(np.float64)
    for (p, l), g in meta.groupby(["plate_index", "line"]):
        idx = g.index.to_numpy()
        Pn = P[idx].astype(np.float64).tocsr()
        Pn.data = np.log1p(Pn.data * np.repeat(1e4 / libf[idx], np.diff(Pn.indptr)))
        m = np.asarray(Pn.mean(axis=0)).ravel()
        if np.abs(dm[p, l]).sum() > 0:
            worst = max(worst, float(np.abs(m - dm[p, l]).max())); n_pool += 1
    chk.update(n_pools_compared=n_pool, max_abs_diff_pool_mean=worst)

    meta.to_csv(out / "cells.tsv", sep="\t", index=False)
    sp.save_npz(out / "panel_counts.npz", P)
    pools = meta.groupby(["plate_index", "line"]).size()
    rec = dict(
        source=str(a.tahoe), source_note="public Tahoe-100M parquet mirror, opened read-only; nothing downloaded",
        dmso_samples=samples, n_shards_total=len(shards), n_shards_read=len(use), shards=recs,
        integer_check=dict(expr_dtype=sorted({r["expr_dtype"] for r in recs if "expr_dtype" in r}),
                           n_values=int(sum(r.get("n_values", 0) for r in recs)),
                           n_non_integer=int(n_nonint), n_negative=int(n_neg),
                           n_zero_stored=int(sum(r.get("n_zero_stored", 0) for r in recs)),
                           max_count=float(max(r.get("max_count", 0) for r in recs))),
        n_cells=int(len(meta)), n_pools=int(len(pools)), pool_size=pools.describe().to_dict(),
        library_size=meta.library_size.describe().to_dict(), panel_sum=meta.panel_sum.describe().to_dict(),
        panel_fraction_of_library=float(meta.panel_sum.sum() / meta.library_size.sum()),
        n_genes_panel=meta.n_genes_panel.describe().to_dict(),
        cross_check=chk,
        outputs={f: dict(bytes=os.path.getsize(out / f), sha256=sha256_file(out / f))
                 for f in ["cells.tsv", "panel_counts.npz"]},
        cache_dir=str(out), elapsed_s=round(time.time() - t0, 1))
    Path(a.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(a.receipt).write_text(json.dumps(rec, indent=1, default=float))
    (out / "raw_count_receipt.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: v for k, v in rec.items() if k != "shards"}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
