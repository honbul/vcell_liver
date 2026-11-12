"""One pass over the Tahoe-100M shards to build the v2 common ground truth and a fixed single-cell
training sample.

Why a new pass at all.  The v1 study -- and every artifact under `prior_assets/r3` -- aggregates a
well as `log1p(mean CP10k)`: it sums CP10k over the cells of a (well, cell line) group and takes the
log of the mean. The v2 contract is `mean(log1p(CP10k))`: the log is taken per cell and then
averaged. The second cannot be recovered from the first, because the stored sums have already
discarded the per-cell values. So the shards are read again.

What is accumulated, per group = (well sample, cell line):

    logsum[g, j] = sum over the group's cells of log1p(1e4 * count_ij / library_size_i)
    ncells[g]

Both restricted to the 11,132 genes that passed the earlier study's measurability filter. The v2
truth of a (cell line, drug) condition is then

    y = mean over the drug's valid wells of ( logsum/ncells  -  logsum/ncells of the same plate's
                                              DMSO pool for that cell line )

with replicate wells weighted equally, exactly as in v1 -- only the within-well aggregation changes.

Single cells.  chemCPA trains on individual cells, not on pseudobulk rows, so this pass also writes
a fixed sample of real cells. Membership is decided by a hash of the cell's barcode against a
per-group threshold set from that group's known cell count, so it depends on nothing but the
barcode and is identical however the shards are ordered or split across workers. The selected cells
are stored on the frozen 2,000-gene panel as log1p(CP10k), which is the same quantity the v2 truth
is built from.

Only the wells this study can use are read: the 0.5 uM wells of the 349 structure groups and the 28
DMSO wells. The raw mirror is opened read-only and nothing is downloaded.

Heavy: Slurm only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import resource
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

_G = {}


def barcode_keys(barcodes: list[str]) -> np.ndarray:
    """A stable 64-bit key per cell. blake2b, not Python's hash, which is salted per process."""
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(b.encode(), digest_size=8).digest(), "big")
         for b in barcodes), dtype=np.uint64, count=len(barcodes))


def shard_plate(path: str) -> tuple[str, str, int]:
    t = pq.read_table(path, columns=["plate"])
    return path, "|".join(sorted(pc.unique(t.column("plate")).to_pylist())), t.num_rows


def run_plate(task):
    plate, part, shards, out_dir, n_cells_target = task
    t0 = time.time()
    samples = _G["plate_samples"][plate]            # only the wells this study uses
    lines = _G["lines"]
    s_idx = {s: i for i, s in enumerate(samples)}
    l_idx = {l: i for i, l in enumerate(lines)}
    n_grp = len(samples) * len(lines)
    keep_tok = _G["keep_tok"]                       # token ids of the 11,132 measurable genes
    tok2col = _G["tok2col"]                         # token id -> column in logsum, -1 if dropped
    panel_cols = _G["panel_cols"]                   # columns of the 2,000-gene panel within those
    thresh = _G["thresh"]                           # per-group acceptance fraction in [0, 1]

    logsum = np.zeros((n_grp, len(keep_tok)), dtype=np.float64)
    ncells = np.zeros(n_grp, dtype=np.int64)
    cell_rows, cell_meta = [], []
    minimal = _G["minimal_by_plate"].get(plate, set())
    n_minimal_hits, n_rows, n_kept_rows = 0, 0, 0

    for k, f in enumerate(shards):
        t = pq.read_table(f, columns=COLS)
        if len(pc.unique(t.column("plate"))) > 1:
            t = t.filter(pc.equal(t.column("plate"), plate))
        t = t.filter(pc.is_in(t.column("sample"), value_set=pa.array(samples)))
        t = t.filter(pc.is_in(t.column("cell_line_id"), value_set=pa.array(lines)))
        n = t.num_rows
        n_rows += n
        if n == 0:
            continue
        g = t.column("genes").combine_chunks()
        e = t.column("expressions").combine_chunks()
        off = g.offsets.to_numpy()
        off = off - off[0]
        gv = g.flatten().to_numpy()
        ev = e.flatten().to_numpy().astype(np.float64)
        assert np.all(gv[off[:-1]] < 3) and int((gv < 3).sum()) == n, f"special-token layout in {f}"
        keep = np.ones(len(gv), bool)
        keep[off[:-1]] = False                      # drop the leading special token
        X = sp.csr_matrix((ev[keep], gv[keep].astype(np.int32), off - np.arange(n + 1)), shape=(n, NG))
        X.sum_duplicates()
        lib = np.asarray(X.sum(axis=1)).ravel()

        smp = t.column("sample").to_pylist()
        cl = t.column("cell_line_id").to_pylist()
        bc = t.column("BARCODE_SUB_LIB_ID").to_pylist()
        n_minimal_hits += sum(1 for b in bc if b in minimal)
        gid = np.fromiter((s_idx[s] * len(lines) + l_idx[c] for s, c in zip(smp, cl)),
                          dtype=np.int64, count=n)

        # CP10k then log1p, per cell -- this is the whole point of the second pass
        Xn = X[:, keep_tok].tocsr()
        Xn.data = np.log1p(Xn.data * np.repeat(1e4 / np.maximum(lib, 1),
                                               np.diff(Xn.indptr)))
        ind = sp.csr_matrix((np.ones(n), (gid, np.arange(n))), shape=(n_grp, n))
        R = (ind @ Xn).tocoo()
        logsum[R.row, R.col] += R.data
        ncells += np.bincount(gid, minlength=n_grp)
        n_kept_rows += n

        # the fixed single-cell sample
        sel = (barcode_keys(bc).astype(np.float64) / 2.0 ** 64) < thresh[gid]
        if sel.any():
            cell_rows.append(np.asarray(Xn[sel][:, panel_cols].todense(), dtype=np.float32))
            cell_meta.append(pd.DataFrame(dict(
                barcode=[b for b, s in zip(bc, sel) if s],
                sample=[s for s, q in zip(smp, sel) if q],
                cell_line_id=[c for c, q in zip(cl, sel) if q],
                plate=plate, library_size=lib[sel])))
        if k % 25 == 0:
            print(f"[{plate}.{part}] shard {k+1}/{len(shards)} kept={n_kept_rows:,} "
                  f"{time.time()-t0:.0f}s", flush=True)

    out = Path(out_dir)
    np.save(out / "tmp" / f"logsum_{plate}_p{part}.npy", logsum.astype(np.float64))
    np.save(out / "tmp" / f"ncells_{plate}_p{part}.npy", ncells)
    if cell_rows:
        np.save(out / "tmp" / f"cells_{plate}_p{part}.npy", np.concatenate(cell_rows))
        pd.concat(cell_meta, ignore_index=True).to_csv(
            out / "tmp" / f"cells_{plate}_p{part}.tsv", sep="\t", index=False)
        n_sel = sum(len(c) for c in cell_rows)
    else:
        n_sel = 0
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return dict(plate=plate, part=part, n_shards=len(shards), n_cells_kept=int(n_kept_rows),
                n_cells_sampled=int(n_sel), n_minimal_hits=int(n_minimal_hits),
                elapsed_s=round(time.time() - t0, 1),
                worker_maxrss_GB=round(ru.ru_maxrss / 1e6, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tahoe", default="/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M")
    ap.add_argument("--prep-dir", required=True, help="prior_assets/r3/prep, for wells/lines/genes")
    ap.add_argument("--v1-manifest-dir", required=True, help="manifests/moe_drug_v1, for the split")
    ap.add_argument("--v1-data-dir", required=True, help="the v1 data dir, for the frozen gene panel")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cells-per-group", type=int, default=64)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--plates", default="")
    ap.add_argument("--max-shards-per-plate", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    T, out = Path(args.tahoe), Path(args.out_dir)
    (out / "tmp").mkdir(parents=True, exist_ok=True)

    # ---- which wells, lines and genes -----------------------------------------------------
    wells = pd.read_csv(Path(args.prep_dir) / "wells.tsv", sep="\t")
    dmso = pd.read_csv(Path(args.prep_dir) / "dmso_wells.tsv", sep="\t")
    splits = pd.read_csv(Path(args.v1_manifest_dir) / "splits.tsv", sep="\t")
    lines_keep = pd.read_csv(Path(args.v1_manifest_dir) / "lines.tsv", sep="\t")
    panel = pd.read_csv(Path(args.v1_manifest_dir) / "genes.tsv", sep="\t")
    meas = pd.read_csv(Path(args.prep_dir) / "genes.tsv", sep="\t")

    lines = sorted(lines_keep.cell_line_id.tolist())
    dose = wells[np.isclose(wells.dose_uM, 0.5)]
    use_samples = sorted(set(dose["sample"]) | set(dmso["sample"]))
    print(f"wells: {len(dose)} treated at 0.5 uM + {len(dmso)} DMSO = {len(use_samples)} samples; "
          f"{len(lines)} cell lines", flush=True)

    keep_tok = meas.token_id.to_numpy()
    tok2col = np.full(NG, -1, np.int64)
    tok2col[keep_tok] = np.arange(len(keep_tok))
    panel_cols = tok2col[panel.token_id.to_numpy()]
    assert (panel_cols >= 0).all(), "the frozen panel must sit inside the measurable gene set"

    # ---- per-group acceptance threshold for the single-cell sample --------------------------
    pb = pd.read_csv(Path(args.prep_dir).parent / "pb" / "groups.tsv", sep="\t",
                     usecols=["sample", "cell_line_id", "n_cells"])
    pb = pb[pb["sample"].isin(use_samples) & pb.cell_line_id.isin(lines)]
    size = {(r["sample"], r.cell_line_id): int(r.n_cells) for _, r in pb.iterrows()}

    sm = pd.read_parquet(T / "metadata/sample_metadata.parquet")
    # The 0.5 uM wells sit on five plates only. A DMSO well is a control for its own plate and for
    # no other, so the other nine plates' DMSO wells are unusable here and their shards are not read.
    treated_plates = sorted(set(sm[sm["sample"].isin(set(dose["sample"]))].plate))
    plate_samples = {p: sorted(set(g["sample"]) & set(use_samples))
                     for p, g in sm.groupby("plate") if p in treated_plates}
    plate_samples = {p: s for p, s in plate_samples.items() if s}
    use_samples = sorted({s for v in plate_samples.values() for s in v})
    print(f"plates carrying 0.5 uM wells: {treated_plates}; "
          f"{len(use_samples)} samples read", flush=True)

    obs = pq.read_table(T / "metadata/obs_metadata.parquet",
                        columns=["plate", "BARCODE_SUB_LIB_ID", "pass_filter"])
    pf = obs.column("pass_filter").cast(pa.string())
    obs = obs.set_column(obs.schema.get_field_index("pass_filter"), "pass_filter", pf)
    mini = obs.filter(pc.equal(obs.column("pass_filter"), "minimal")).select(
        ["plate", "BARCODE_SUB_LIB_ID"]).to_pandas()
    _G["minimal_by_plate"] = {p: set(g.BARCODE_SUB_LIB_ID) for p, g in mini.groupby("plate")}
    del obs, mini

    _G.update(lines=lines, keep_tok=keep_tok, tok2col=tok2col, panel_cols=panel_cols,
              plate_samples=plate_samples)

    shards = sorted(str(p) for p in (T / "data").glob("train-*.parquet"))
    with mp.Pool(min(32, args.procs * 2)) as pool:
        sp_ = pool.map(shard_plate, shards, chunksize=8)
    sp_df = pd.DataFrame(sp_, columns=["shard", "plate", "n_rows"])
    sp_long = sp_df.assign(plate=sp_df.plate.str.split("|")).explode("plate")
    print(f"shard scan {time.time()-t0:.0f}s; {len(sp_df)} shards, {sp_df.n_rows.sum():,} rows",
          flush=True)

    plates = sorted(p for p in sp_long.plate.unique() if p in plate_samples)
    if args.plates:
        plates = [p for p in plates if p in args.plates.split(",")]

    tasks, group_index = [], []
    for p in plates:
        smpl = plate_samples[p]
        thr = np.zeros(len(smpl) * len(lines), np.float64)
        for i, s in enumerate(smpl):
            for j, l in enumerate(lines):
                n = size.get((s, l), 0)
                # take every cell of a small group, otherwise about `cells_per_group` of them
                thr[i * len(lines) + j] = 1.0 if n <= args.cells_per_group else args.cells_per_group / n
                group_index.append((p, s, l, n))
        _G.setdefault("thresh_by_plate", {})[p] = thr
        fs = sp_long[sp_long.plate == p].shard.tolist()
        if args.max_shards_per_plate:
            fs = fs[: args.max_shards_per_plate]
        for k in range(args.parts):
            tasks.append((p, k, fs[k::args.parts], str(out), args.cells_per_group))

    # Workers are forked after this point, so each inherits the whole threshold table and picks
    # out its own plate's slice in `_run`.
    _G["thresh_all"] = _G["thresh_by_plate"]

    tasks.sort(key=lambda x: -len(x[2]))
    with mp.Pool(args.procs) as pool:
        res = pool.map(_run, tasks, chunksize=1)

    gi = pd.DataFrame(group_index, columns=["plate", "sample", "cell_line_id", "n_cells_total"])
    gi.to_csv(out / "groups.tsv", sep="\t", index=False)

    # ---- combine -------------------------------------------------------------------------
    logsum = np.zeros((len(gi), len(keep_tok)), np.float64)
    ncells = np.zeros(len(gi), np.int64)
    base = 0
    cells_X, cells_meta = [], []
    for p in plates:
        n_grp = len(plate_samples[p]) * len(lines)
        for k in range(args.parts):
            logsum[base: base + n_grp] += np.load(out / "tmp" / f"logsum_{p}_p{k}.npy")
            ncells[base: base + n_grp] += np.load(out / "tmp" / f"ncells_{p}_p{k}.npy")
            f = out / "tmp" / f"cells_{p}_p{k}.npy"
            if f.exists():
                cells_X.append(np.load(f))
                cells_meta.append(pd.read_csv(out / "tmp" / f"cells_{p}_p{k}.tsv", sep="\t"))
        base += n_grp

    np.save(out / "logsum.npy", logsum.astype(np.float32))
    np.save(out / "ncells.npy", ncells)
    X = np.concatenate(cells_X)
    np.save(out / "cells_X.npy", X)
    cm = pd.concat(cells_meta, ignore_index=True)
    cm.to_csv(out / "cells_meta.tsv", sep="\t", index=False)

    rec = dict(tahoe=str(T), n_shards=len(shards), n_samples=len(use_samples),
               n_treated_wells=int(len(dose)), n_dmso_wells=int(len(dmso)),
               n_lines=len(lines), n_measurable_genes=int(len(keep_tok)),
               n_panel_genes=int(len(panel_cols)), cells_per_group=args.cells_per_group,
               n_groups=int(len(gi)), n_cells_sampled=int(len(X)),
               n_cells_aggregated=int(ncells.sum()),
               truth_definition="mean over cells of log1p(1e4 * count / library_size)",
               v1_truth_definition="log1p(1e4 * sum(count) / sum(library_size))",
               workers=res, elapsed_s=round(time.time() - t0, 1))
    (out / "build_record.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "workers"}, indent=1), flush=True)


def _run(task):
    _G["thresh"] = _G["thresh_all"][task[0]]
    return run_plate(task)


if __name__ == "__main__":
    main()
