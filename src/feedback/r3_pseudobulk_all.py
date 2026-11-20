"""R3 step A1: per-well pseudobulk for all 50 Tahoe-100M cell lines in one pass over the local shards.

Source (read-only, other user's mirror): <tahoe>/data/train-*.parquet (3,388 shards; a shard holds cells of
one plate (a few straddle two plates; rows are filtered by plate), all 50 lines). Each row is one cell: `genes` = token ids, `expressions` = raw UMI counts; token ids 0-2 are
special (every cell starts with token 1 / value -2) and are dropped. Gene token ids 3..62712 index
metadata/gene_metadata.parquet (token_id).

Unit: group = (sample, cell_line_id). A Tahoe `sample` is one well (one drug x dose) on one plate; all 50
lines share that well (MOSAIC pool). DMSO_TF wells (2 per plate) give the plate-matched control per line.

Accumulated per group:
  raw   = sum of raw counts                          (float32 on disk; exact integers < 2^24 checked)
  norm  = sum over cells of CP10k-normalised counts  (per-cell library size = sum of its counts)
  n_cells, total_umi
Expression used downstream: log1p(norm / n_cells)  ("log1p of the mean library-normalised profile").

QC cell set: Tahoe pass_filter == "full". The shards were checked against obs_metadata: every shard cell is
looked up in the set of pass_filter=="minimal" barcodes of its plate (must be 0 hits) and per-group cell
counts are compared with obs_metadata full-cell counts per (sample, cell_line).

Output: <out>/plate_<p>_raw.npy, plate_<p>_norm.npy (float32 [n_groups_p, 62713]), groups.tsv, genes.tsv,
        build_record.json.
Heavy: Slurm only (one worker process per plate).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

NG = 62713  # token id space (0..62712); 0-2 special
COLS = ["genes", "expressions", "sample", "cell_line_id", "BARCODE_SUB_LIB_ID", "plate"]

_G = {}  # fork-shared globals


def shard_plate(path: str) -> tuple[str, str, int]:
    t = pq.read_table(path, columns=["plate"])
    u = pc.unique(t.column("plate")).to_pylist()
    return path, "|".join(sorted(u)), t.num_rows


def run_plate(args):
    plate, part, shards, out_dir = args
    t0 = time.time()
    samples = _G["plate_samples"][plate]
    lines = _G["lines"]
    s_idx = {s: i for i, s in enumerate(samples)}
    l_idx = {l: i for i, l in enumerate(lines)}
    n_grp = len(samples) * len(lines)
    # float32 accumulators: raw sums are integers (< 2^24 asserted -> exact); norm relative error ~1e-7
    raw = np.zeros((n_grp, NG), dtype=np.float32)
    nrm = np.zeros((n_grp, NG), dtype=np.float32)
    n_cells = np.zeros(n_grp, dtype=np.int64)
    umi = np.zeros(n_grp, dtype=np.float64)
    minimal = _G["minimal_by_plate"].get(plate, set())
    n_minimal_hits = 0
    n_rows = 0
    nonint = 0
    for k, f in enumerate(shards):
        t = pq.read_table(f, columns=COLS)
        if len(pc.unique(t.column("plate"))) > 1:  # a few shards straddle two plates
            t = t.filter(pc.equal(t.column("plate"), plate))
        n = t.num_rows
        n_rows += n
        g = t.column("genes").combine_chunks()
        e = t.column("expressions").combine_chunks()
        off = g.offsets.to_numpy()
        off = off - off[0]
        gv = g.flatten().to_numpy()  # flatten() honours slice offsets
        ev = e.flatten().to_numpy().astype(np.float64)
        assert len(gv) == off[-1] == len(ev)
        # every cell starts with exactly one special token (id 1, value -2); verified, then dropped
        assert np.all(gv[off[:-1]] < 3) and int((gv < 3).sum()) == n, f"unexpected special-token layout in {f}"
        keep = np.ones(len(gv), bool)
        keep[off[:-1]] = False
        X = sp.csr_matrix((ev[keep], gv[keep].astype(np.int32), off - np.arange(n + 1)), shape=(n, NG))
        X.has_sorted_indices = False
        X.sum_duplicates()
        if k % 50 == 0:
            nonint += int(np.sum(X.data != np.round(X.data))) + int(np.sum(X.data <= 0))
        lib = np.asarray(X.sum(axis=1)).ravel()
        smp = t.column("sample").to_pylist()
        cl = t.column("cell_line_id").to_pylist()
        bc = t.column("BARCODE_SUB_LIB_ID").to_pylist()
        n_minimal_hits += sum(1 for b in bc if b in minimal)
        gid = np.fromiter((s_idx[s] * len(lines) + l_idx[c] for s, c in zip(smp, cl)), dtype=np.int64, count=n)
        ind = sp.csr_matrix((np.ones(n), (gid, np.arange(n))), shape=(n_grp, n))
        # sparse group x gene products; coo from a csr product has unique coordinates -> safe fancy add
        R = (ind @ X).tocoo()
        raw[R.row, R.col] += R.data.astype(np.float32)
        Xn = X.copy()
        Xn.data *= np.repeat(1e4 / np.maximum(lib, 1), np.diff(Xn.indptr))
        R = (ind @ Xn).tocoo()
        nrm[R.row, R.col] += R.data.astype(np.float32)
        n_cells += np.bincount(gid, minlength=n_grp)
        umi += np.bincount(gid, weights=lib, minlength=n_grp)
        if k % 25 == 0:
            print(f"[{plate}.{part}] shard {k+1}/{len(shards)} cells={n_rows:,} {time.time()-t0:.0f}s", flush=True)
    assert raw.max() < 2 ** 24, "raw sum exceeds float32 exact-integer range"
    np.save(Path(out_dir) / "tmp" / f"plate_{plate}_p{part}_raw.npy", raw)
    np.save(Path(out_dir) / "tmp" / f"plate_{plate}_p{part}_norm.npy", nrm)
    gt = pd.DataFrame({"plate": plate, "row": np.arange(n_grp),
                       "sample": np.repeat(samples, len(lines)), "cell_line_id": np.tile(lines, len(samples)),
                       "n_cells": n_cells, "total_umi": umi})
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return dict(plate=plate, part=part, n_shards=len(shards), n_cells=int(n_rows), n_minimal_hits=int(n_minimal_hits),
                nonint_values_checked_shards=int(nonint), elapsed_s=round(time.time() - t0, 1),
                worker_maxrss_GB=round(ru.ru_maxrss / 1e6, 2)), gt


def combine(a):
    """sum the per-part partial sums of one plate (float64 accumulation) -> float32 file."""
    out, p, kind, parts = a
    out = Path(out)
    acc = None
    for k in range(parts):
        f = out / "tmp" / f"plate_{p}_p{k}_{kind}.npy"
        x = np.load(f).astype(np.float64)
        acc = x if acc is None else acc + x
        f.unlink()
    if kind == "raw":
        assert acc.max() < 2 ** 24
    np.save(out / f"plate_{p}_{kind}.npy", acc.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tahoe", default="/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--parts", type=int, default=2, help="shard subsets per plate (parallel workers per plate)")
    ap.add_argument("--max-shards-per-plate", type=int, default=0, help="test mode")
    ap.add_argument("--plates", default="", help="comma list, test mode")
    args = ap.parse_args()
    t0 = time.time()
    T = Path(args.tahoe)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shards = sorted(str(p) for p in (T / "data").glob("train-*.parquet"))
    with mp.Pool(min(32, args.procs * 2)) as pool:
        sp_ = pool.map(shard_plate, shards, chunksize=8)
    sp_df = pd.DataFrame(sp_, columns=["shard", "plate", "n_rows"])
    n_multi = int(sp_df.plate.str.contains(r"\|").sum())
    print(f"shard->plate scan {time.time()-t0:.0f}s; shards={len(sp_df)} rows={sp_df.n_rows.sum():,} "
          f"multi-plate shards={n_multi}", flush=True)
    sp_long = sp_df.assign(plate=sp_df.plate.str.split("|")).explode("plate")

    sm = pd.read_parquet(T / "metadata/sample_metadata.parquet")
    obs = pq.read_table(T / "metadata/obs_metadata.parquet",
                        columns=["plate", "BARCODE_SUB_LIB_ID", "sample", "cell_line", "pass_filter"])
    pf = obs.column("pass_filter").cast(pa.string())
    obs = obs.set_column(obs.schema.get_field_index("pass_filter"), "pass_filter", pf)
    obs = obs.set_column(obs.schema.get_field_index("cell_line"), "cell_line", obs.column("cell_line").cast(pa.string()))
    cnt = obs.group_by(["sample", "cell_line", "pass_filter"]).aggregate([("plate", "count")]).to_pandas()
    mini = obs.filter(pc.equal(obs.column("pass_filter"), "minimal")).select(["plate", "BARCODE_SUB_LIB_ID"]).to_pandas()
    _G["minimal_by_plate"] = {p: set(g.BARCODE_SUB_LIB_ID) for p, g in mini.groupby("plate")}
    lines = sorted(cnt.cell_line.unique())
    _G["lines"] = lines
    _G["plate_samples"] = {p: sorted(g["sample"]) for p, g in sm.groupby("plate")}
    del obs, mini
    print(f"obs_metadata loaded {time.time()-t0:.0f}s; lines={len(lines)}", flush=True)

    plates = sorted(sp_long.plate.unique())
    if args.plates:
        plates = [p for p in plates if p in args.plates.split(",")]
    tasks = []
    (out / "tmp").mkdir(exist_ok=True)
    for p in plates:
        fs = sp_long[sp_long.plate == p].shard.tolist()
        if args.max_shards_per_plate:
            fs = fs[: args.max_shards_per_plate]
        for k in range(args.parts):
            tasks.append((p, k, fs[k::args.parts], str(out)))
    tasks.sort(key=lambda x: -len(x[2]))
    with mp.Pool(args.procs) as pool:
        res = pool.map(run_plate, tasks, chunksize=1)
    recs = [r for r, _ in res]
    gts = {}
    for _, g in res:  # parts of one plate share the same group layout; sum counts
        p = g.plate.iloc[0]
        if p in gts:
            gts[p]["n_cells"] += g.n_cells.to_numpy(); gts[p]["total_umi"] += g.total_umi.to_numpy()
        else:
            gts[p] = g.copy()
    gt = pd.concat([gts[p] for p in sorted(gts)], ignore_index=True)
    with mp.Pool(min(args.procs, 2 * len(gts))) as pool:
        pool.map(combine, [(str(out), p, kind, args.parts) for p in sorted(gts) for kind in ("raw", "norm")])
    print(f"combined parts {time.time()-t0:.0f}s", flush=True)
    (out / "tmp").rmdir()

    # verification vs obs_metadata full-cell counts
    full = cnt[cnt.pass_filter == "full"].rename(columns={"cell_line": "cell_line_id", "plate_count": "n_full_obs"})
    gt = gt.merge(full[["sample", "cell_line_id", "n_full_obs"]], on=["sample", "cell_line_id"], how="left")
    gt["n_full_obs"] = gt.n_full_obs.fillna(0).astype(int)
    mism = int((gt.n_cells != gt.n_full_obs).sum())
    clm = pd.read_parquet(T / "metadata/cell_line_metadata.parquet").drop_duplicates("Cell_ID_Cellosaur")
    gt = gt.merge(clm[["Cell_ID_Cellosaur", "cell_name", "Organ"]].rename(columns={"Cell_ID_Cellosaur": "cell_line_id"}),
                  on="cell_line_id", how="left")
    gt = gt.merge(sm[["sample", "drug", "drugname_drugconc"]], on="sample", how="left")
    gt.to_csv(out / "groups.tsv", sep="\t", index=False)
    gm = pd.read_parquet(T / "metadata/gene_metadata.parquet")
    gm.sort_values("token_id").to_csv(out / "genes.tsv", sep="\t", index=False)

    ru_c = resource.getrusage(resource.RUSAGE_CHILDREN)
    ru_s = resource.getrusage(resource.RUSAGE_SELF)
    rec = dict(source=str(T), n_shards=int(len(sp_df)), shard_rows=int(sp_df.n_rows.sum()), multi_plate_shards=n_multi,
               full_cells_obs=int(full.n_full_obs.sum()), cells_accumulated=int(gt.n_cells.sum()),
               group_count_mismatches_vs_obs_full=mism,
               minimal_cells_found_in_shards=int(sum(r["n_minimal_hits"] for r in recs)),
               n_groups=int(len(gt)), n_groups_with_cells=int((gt.n_cells > 0).sum()), lines=len(lines),
               plates=recs, procs=args.procs, parts_per_plate=args.parts, elapsed_s=round(time.time() - t0, 1),
               main_maxrss_GB=round(ru_s.ru_maxrss / 1e6, 2), max_child_maxrss_GB=round(ru_c.ru_maxrss / 1e6, 2),
               cpu_user_h_children=round(ru_c.ru_utime / 3600, 2), host=os.uname().nodename,
               slurm_job=os.environ.get("SLURM_JOB_ID"))
    (out / "build_record.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "plates"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
