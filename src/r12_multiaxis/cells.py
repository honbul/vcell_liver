"""Freeze a metadata-seeded sample of own-control cells for the context experiments."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

from src.r11_chemical_probe.common import config, sha256, write_json
from src.r11_chemical_probe.dataset import KEY, table
from src.r11_chemical_probe.prepare import check_revision, sample_bounds, unique


STATE = {}
COLUMNS = ["sample", "cell_line_id", "BARCODE_SUB_LIB_ID", "genes", "expressions"]


def init(samples, lines, seed, limit):
    STATE.update(samples=pa.array(samples), lines=pa.array(lines), seed=seed, limit=limit)


def rank(well, line, barcode):
    key = f"{STATE['seed']}:{well}:{line}:{barcode}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def merge(top, new):
    for key, entries in new.items():
        top[key] = sorted(top.get(key, []) + entries, key=lambda v: (v[0], v[1]))[:STATE["limit"]]


def scan(task):
    path, row_groups = task
    parquet = pq.ParquetFile(path)
    top = {}
    for batch in parquet.iter_batches(batch_size=8192, row_groups=row_groups, columns=COLUMNS, use_threads=False):
        mask = pc.and_(pc.is_in(batch["sample"], value_set=STATE["samples"]),
                       pc.is_in(batch["cell_line_id"], value_set=STATE["lines"]))
        values = batch.filter(mask)
        if not len(values):
            continue
        keys = list(zip(*(values[name].to_pylist() for name in COLUMNS[:3])))
        keep = {}
        for index, (well, line, barcode) in enumerate(keys):
            keep.setdefault((well, line), []).append((rank(well, line, barcode), str(barcode), index))
        selected = {}
        for unit, entries in keep.items():
            for score, barcode, index in sorted(entries)[:STATE["limit"]]:
                genes = values["genes"][index].as_py()
                counts = values["expressions"][index].as_py()
                if len(genes) != len(counts) or genes[0] != 1:
                    raise ValueError("Malformed Tahoe cell vector")
                if sum(counts[1:]) > 0:
                    selected.setdefault(unit, []).append((score, barcode, genes[1:], counts[1:]))
        merge(top, selected)
    return top


def collect(cfg, out):
    start = time.monotonic()
    r11 = Path(cfg["r11_assets"])
    root = Path(cfg["tahoe_root"])
    samples = table(r11 / "metadata/samples.tsv")
    controls = samples[samples.control_role.isin(["A", "B"]) & samples.plate_id.isin(cfg["plates"])]
    wells = controls.physical_well_id.tolist()
    lines = table(r11 / "metadata/lines.tsv").cell_line_id.tolist()
    genes = table(r11 / "metadata/genes.tsv")
    width = int(genes.token_id.max()) + 1
    options = cfg["optional"]
    init(wells, lines, options["cell_seed"], options["cells_per_control_line"])
    workers = 7
    selected, receipts = [], []
    out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, bounds in pool.map(sample_bounds, sorted((root / "data").glob("train-*.parquet"))):
            active = [i for i, lo, hi in bounds if any(lo <= well <= hi for well in wells)]
            if active:
                tag = check_revision(root, path.relative_to(root), cfg["revision"])
                receipts.append(dict(shard=path.name, etag=tag, row_groups=len(active)))
                selected.append((path, active))
    print(f"Control sample: {len(wells)} wells in {len(selected)} shards", flush=True)
    pd.DataFrame(receipts).to_csv(out / "shards.tsv", sep="\t", index=False)
    top = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=init,
                             initargs=(wells, lines, options["cell_seed"], options["cells_per_control_line"])) as pool:
        for start_at in range(0, len(selected), workers * 2):
            for results in pool.map(scan, selected[start_at:start_at + workers * 2]):
                merge(top, results)
            print(f"{min(start_at + workers * 2, len(selected))}/{len(selected)} shards", flush=True)

    key_rows, col, val, ptr = [], [], [], [0]
    roles = controls.set_index(KEY[0]).control_role
    plates = controls.set_index(KEY[0]).plate_id
    for (well, line), entries in sorted(top.items()):
        for i, (_, barcode, tokens, counts) in enumerate(entries):
            col.extend(tokens)
            val.extend(counts)
            ptr.append(len(col))
            key_rows.append((well, line, barcode, roles[well], plates[well], i % 2))
    rows = pd.DataFrame(key_rows, columns=KEY + ["barcode", "role", "plate_id", "view"])
    unique(rows, KEY + ["barcode"])
    counts = sp.csr_matrix((np.asarray(val, dtype=np.int32), col, ptr), shape=(len(rows), width))
    counts.sort_indices()
    if counts.min() < 0 or counts[:, :3].nnz:
        raise ValueError("Invalid mapped count")
    files = ["shards.tsv"]
    for role in ("A", "B"):
        use = rows.role.eq(role)
        row_name, count_name = f"{role}_rows.tsv", f"{role}_cells.npz"
        rows[use].to_csv(out / row_name, sep="\t", index=False)
        sp.save_npz(out / count_name, counts[use.to_numpy()], compressed=False)
        files.extend([row_name, count_name])
    write_json(out / "receipt.json", dict(revision=cfg["revision"], n_cells=len(rows),
               n_units=len(top), elapsed_seconds=time.monotonic() - start,
               max_cells=options["cells_per_control_line"], seed=options["cell_seed"],
               files={name: sha256(out / name) for name in files}))
    print(f"{len(rows)} cells across {len(top)} own-role units", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(config(args.config), args.out)


if __name__ == "__main__":
    main()
