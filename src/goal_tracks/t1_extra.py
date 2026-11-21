"""Cache the physically paired, parent-disjoint Tahoe 5 µM wells outside the 203-parent cohort."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, save_npz, vstack

from src.moe_drug_v1.prepare import canonical_parent
from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import LINE, table
from src.r11_chemical_probe.prepare import check_revision, dose, read_shard, sample_bounds, set_scan

from .t1_data import PROTOCOL_SHA256, WELL, array, file_hashes, slice_name, validate_protocol, verify_one


def metadata(cfg, coordinates, partition, r6_split):
    root = Path(cfg["tahoe_root"])
    rel_sample = Path("metadata/sample_metadata.parquet")
    rel_drug = Path("metadata/drug_metadata.parquet")
    for rel in (rel_sample, rel_drug):
        check_revision(root, rel, cfg["revision"])
    samples = pq.read_table(root / rel_sample, columns=["sample", "plate", "drug", "drugname_drugconc"]).to_pandas()
    samples = samples.loc[samples.plate.isin(("plate6", "plate14"))].copy()
    samples["dose_uM"] = samples.drugname_drugconc.map(dose)
    samples["drug"] = samples.drug.str.strip()
    drugs = pq.read_table(root / rel_drug, columns=["drug", "canonical_smiles"]).to_pandas()
    drugs["drug"] = drugs.drug.str.strip()
    drugs = drugs.loc[drugs.drug.isin(samples.loc[samples.dose_uM.eq(5), "drug"])].copy()
    drugs["canonical_parent"] = drugs.canonical_smiles.map(
        lambda s: canonical_parent(s) if isinstance(s, str) and s else None
    ).map(lambda p: p[1] if p is not None else "")
    book = table(coordinates / "positions.tsv")
    if sha256(coordinates / "positions.tsv") != read_json(coordinates / "receipt.json")["positions_sha256"]:
        raise ValueError("Barcode release receipt changed")
    samples = samples.merge(book[["sample", "plate", "barcode_position"]], on=["sample", "plate"], validate="one_to_one")
    samples = samples.loc[samples.dose_uM.eq(5)].merge(drugs, on="drug", how="left", validate="many_to_one")
    samples["canonical_parent"] = samples.canonical_parent.fillna("")
    cols = ["barcode_position", "drug", "canonical_parent"]
    left = samples.loc[samples.plate.eq("plate6")].set_index(cols, verify_integrity=True)
    right = samples.loc[samples.plate.eq("plate14")].set_index(cols, verify_integrity=True)
    pairs = left[["sample"]].join(right[["sample"]], how="inner", lsuffix="_plate6", rsuffix="_plate14").reset_index()
    legacy = read_json(r6_split)["group_smiles"]
    seen = {legacy[g] for g in read_json(partition)["outer"]}
    outside = pairs.loc[~pairs.canonical_parent.isin(seen)].reset_index(drop=True).copy()
    if len(pairs) != 93 or len(outside) != 40:
        raise ValueError(f"Expected 93 total and 40 reserved pairs, found {len(pairs)} / {len(outside)}")
    outside["parent_verified"] = outside.canonical_parent.ne("")
    if not outside.parent_verified.all():
        raise ValueError("A repeat pair lacks independently verified parent identity")
    return outside, {str(p): sha256(root / p) for p in (rel_sample, rel_drug)}


def stage_extra(args):
    validate_protocol(args.protocol)
    cfg = config(args.config)
    root = Path(cfg["tahoe_root"])
    partitions = args.stage / "split.json"
    pairs, inputs = metadata(cfg, args.coordinates, partitions, args.r6_split)
    r11_meta = Path(cfg["r11_assets"]) / "metadata"
    lines = table(r11_meta / "lines.tsv")[LINE].tolist()
    width = len(table(r11_meta / "genes.tsv")) + 3
    params = read_json(args.stage / "receipt.json")
    panels = []
    for f in range(3):
        folder = args.stage / f"f{f}"
        verify_one(args.stage, params, folder / "panels.json")
        lookup = read_json(folder / "panels.json")
        panels.extend(lookup[slice_name(5, n, f, "panel") + "_-1"] for n in (20, 200))
    panel = np.unique(np.concatenate(panels))
    rows = pd.concat([
        pairs[["barcode_position", "drug", "canonical_parent", "parent_verified", f"sample_{p}"]]
        .rename(columns={f"sample_{p}": WELL}).assign(plate_id=p, dose_uM=5.0, time_h=24)
        for p in ("plate6", "plate14")
    ], ignore_index=True)
    well_map = {row[WELL]: (i * len(lines), row["plate_id"]) for i, row in rows.iterrows()}
    line_map = {line: i for i, line in enumerate(lines)}
    paths = sorted((root / "data").glob("train-*.parquet"))
    tasks, revision = [], cfg["revision"]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for path, ranges in executor.map(sample_bounds, paths):
            active = [rg for rg, low, high in ranges if any(low <= well <= high for well in well_map)]
            if active:
                etag = check_revision(root, path.relative_to(root), revision)
                tasks.append((path, active, etag))
    args.out.mkdir(parents=True, exist_ok=False)
    pairs.drop(columns="canonical_parent").to_csv(args.out / "pairs.tsv", sep="\t", index=False)
    pd.DataFrame([(p.name, len(rg), etag) for p, rg, etag in tasks], columns=["shard", "row_groups", "etag"]).to_csv(
        args.out / "shards.tsv", sep="\t", index=False)
    print(f"reserved {len(pairs)} well pairs across {len(tasks)} verified Tahoe shards", flush=True)
    grid = rows.merge(pd.DataFrame({LINE: lines}), how="cross")
    matrices = {}
    cells, library = np.zeros(len(grid), dtype=np.int64), np.zeros(len(grid), dtype=np.int64)
    zero_library = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=set_scan,
                            initargs=(well_map, line_map, panel, width, 256)) as executor:
        scan_tasks = [(p, rg) for p, rg, _ in tasks]
        batch_size = args.workers * 3
        for offset in range(0, len(tasks), batch_size):
            batch = scan_tasks[offset:offset + batch_size]
            for result, zero in executor.map(read_shard, batch):
                zero_library += zero
                for i, (value, n_cells, size) in result.items():
                    matrices[i] = matrices[i] + value if i in matrices else value
                    cells[i] += n_cells
                    library[i] += size
            if offset == 0 or offset % (batch_size * 4) == 0:
                print(f"{min(offset+batch_size,len(tasks))}/{len(tasks)} shards, {cells.sum():,} cells", flush=True)
    zero = csr_matrix((1, len(panel)), dtype=np.int64)
    counts = vstack([matrices.get(i, zero) for i in range(len(grid))], format="csr")
    if not matrices or np.any(np.asarray(counts.sum(axis=1)).ravel() > library):
        raise ValueError("Reserved panel counts exceed their full libraries")
    grid.assign(n_cells=cells, mapped_library=library).drop(columns="canonical_parent").to_csv(
        args.out / "rows.tsv", sep="\t", index=False)
    save_npz(args.out / "counts.npz", counts, compressed=False)
    np.save(args.out / "panel_union.npy", panel, allow_pickle=False)
    write_json(args.out / "receipt.json", dict(
        protocol_sha256=PROTOCOL_SHA256, stage_sha256=sha256(args.stage / "receipt.json"),
        provenance=inputs, r6_split_sha256=sha256(args.r6_split), design="reserved for prospective evaluation",
        well_pairs=len(pairs), parent_verified_pairs=int(pairs.parent_verified.sum()), rows=len(grid), cells=int(cells.sum()), zero_library=zero_library,
        files=file_hashes(args.out),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for what in ("config", "stage", "coordinates", "r6-split", "protocol", "out"):
        parser.add_argument(f"--{what}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    stage_extra(parser.parse_args())


if __name__ == "__main__":
    main()
