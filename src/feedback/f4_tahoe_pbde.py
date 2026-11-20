"""F4: summarise Tahoe-100M official pseudobulk differential expression (read-only, 1026 shards, ~89 GB).

For every (cell line, drug, concentration, plate) condition it counts genes tested and
significant genes (padj < 0.05, with |log2FC| > 0.5 / > 1), and keeps the full rows of the
liver lines (SNU-423 CVCL_0366, HepG2/C3A CVCL_1098) as parquet in assets for concordance.

Outputs:
  <out_dir>/tahoe_pbde_condition_summary.tsv.gz   one row per condition (all 50 lines)
  <asset_dir>/pbde_<line>.parquet                  liver-line rows
Heavy: run through Slurm (cpu partition, multiprocessing over shards).
"""
from __future__ import annotations

import argparse
import glob
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = "/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/pseudobulk_differential_expression/train-*.parquet"
KEYS = ["Cell_ID_Cellosaur", "drug", "concentration", "plate"]
COLS = KEYS + ["gene_name", "baseMean", "log2FoldChange", "padj", "n_cells_trt", "n_cells_ctrl"]
LIVER = {"CVCL_0366": "snu423", "CVCL_1098": "hepg2c3a"}


def one(path: str):
    t = pq.read_table(path, columns=COLS)
    padj = pc.fill_null(t["padj"], 1.0)
    alfc = pc.abs(pc.fill_null(t["log2FoldChange"], 0.0))
    sig = pc.less(padj, 0.05)
    t2 = pa.table({
        **{k: t[k] for k in KEYS},
        "tested": pc.cast(pc.is_valid(t["padj"]), pa.int64()),
        "sig05": pc.cast(sig, pa.int64()),
        "sig05_lfc05": pc.cast(pc.and_(sig, pc.greater(alfc, 0.5)), pa.int64()),
        "sig05_lfc1": pc.cast(pc.and_(sig, pc.greater(alfc, 1.0)), pa.int64()),
        "n_cells_trt": t["n_cells_trt"], "n_cells_ctrl": t["n_cells_ctrl"],
    })
    g = t2.group_by(KEYS).aggregate([("tested", "sum"), ("sig05", "sum"), ("sig05_lfc05", "sum"),
                                     ("sig05_lfc1", "sum"), ("n_cells_trt", "max"), ("n_cells_ctrl", "max"),
                                     ("tested", "count")]).to_pandas()
    liver = {}
    for cid, tag in LIVER.items():
        sub = t.filter(pc.equal(t["Cell_ID_Cellosaur"], cid))
        if sub.num_rows:
            liver[tag] = sub.drop(["Cell_ID_Cellosaur"])
    return g, liver


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(SRC))
    if args.limit:
        files = files[: args.limit]
    t0 = time.time()
    parts, liver = [], {k: [] for k in LIVER.values()}
    with Pool(args.procs) as pool:
        for i, (g, lv) in enumerate(pool.imap_unordered(one, files, chunksize=1)):
            parts.append(g)
            for k, v in lv.items():
                liver[k].append(v)
            if i % 50 == 0:
                print(f"{i+1}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(parts, ignore_index=True)
    agg = {c: "sum" for c in df.columns if c.endswith("_sum") or c.endswith("_count")}
    agg.update({"n_cells_trt_max": "max", "n_cells_ctrl_max": "max"})
    df = df.groupby(KEYS, as_index=False).agg(agg)
    df = df.rename(columns={"tested_count": "n_gene_rows", "tested_sum": "n_tested", "sig05_sum": "n_sig05",
                            "sig05_lfc05_sum": "n_sig05_lfc05", "sig05_lfc1_sum": "n_sig05_lfc1",
                            "n_cells_trt_max": "n_cells_trt", "n_cells_ctrl_max": "n_cells_ctrl"})
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    adir = Path(args.asset_dir); adir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tahoe_pbde_condition_summary.tsv.gz", sep="\t", index=False)
    for k, v in liver.items():
        if v:
            pq.write_table(pa.concat_tables(v), adir / f"pbde_{k}.parquet", compression="zstd")
    print("conditions", len(df), "lines", df.Cell_ID_Cellosaur.nunique(), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
