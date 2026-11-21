"""Prepare normalized partitions and independently staged additional-dose rows."""

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, checked_table, normalize, table
from src.r11_chemical_probe.prepare import counts, dose
from src.r11_chemical_probe.report import align


def partition(cfg, out, split_path):
    r11 = Path(cfg["r11_assets"])
    source = r11 / "source"
    rows = checked_table(source, "treated_rows.tsv")
    common = table(split_path / "source_rows.tsv")
    order = align(common, rows)
    file = source / "treated_counts.npz"
    if sha256(file) != read_json(source / "receipt.json")["files"][file.name]:
        raise ValueError("Source counts differ from receipt")
    raw = sp.load_npz(file)
    if raw.shape[0] != len(rows):
        raise ValueError("Source row count mismatch")
    split = read_json(split_path / "split.json")
    outer = common[GROUP].map({int(k): v for k, v in split["outer"].items()})
    out.mkdir(parents=True, exist_ok=False)
    for k in range(3):
        sub = out / f"outer{k}"
        sub.mkdir()
        for name, mask in (("fit", outer.ne(k)), ("score", outer.eq(k))):
            index = order[mask]
            matrix = normalize(raw[index], rows.iloc[index].mapped_library.to_numpy()).astype(np.float32)
            np.save(sub / f"{name}-data.npy", matrix.data)
            np.save(sub / f"{name}-indices.npy", matrix.indices)
            np.save(sub / f"{name}-indptr.npy", matrix.indptr)
            common.iloc[np.flatnonzero(mask)].to_csv(sub / f"{name}-rows.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", dict(genes=raw.shape[1], source_sha256=sha256(file),
               split_sha256=sha256(split_path / "split.json"),
               files={str(p.relative_to(out)): sha256(p) for p in out.glob("outer*/*")}))


def load_staged(path, side, width):
    data = tuple(np.load(path / f"{side}-{name}.npy", mmap_mode="r") for name in ("data", "indices", "indptr"))
    rows = table(path / f"{side}-rows.tsv")
    return rows, sp.csr_matrix(data, shape=(len(rows), width))


def dose_manifest(cfg, out):
    r11 = Path(cfg["r11_assets"])
    thaw = Path(cfg["tahoe_root"])
    original = table(r11 / "metadata/samples.tsv")
    entities = original.loc[original.split.eq("train") & original.plate_id.isin(cfg["plates"]),
                            ["compound_entity_id", GROUP, "smiles"]].drop_duplicates()
    raw = pq.read_table(thaw / "metadata/sample_metadata.parquet").to_pandas()
    raw = raw.rename(columns={"sample": KEY[0], "plate": "plate_id", "drug": "compound_entity_id"})
    raw["dose_uM"] = raw.drugname_drugconc.map(dose)
    extra = raw.loc[raw.dose_uM.isin([0.05, 5.0]) & raw.plate_id.ne("plate13")].merge(
        entities, on="compound_entity_id", validate="many_to_one")
    extra = extra.assign(time_h=cfg["time_h"], control_role="treated", split="train", fold=-1, quarantined=False)
    primary_keys = set(original.physical_well_id[original.dose_uM.eq(0.5)])
    if primary_keys & set(extra.physical_well_id):
        raise ValueError("Extra dose shares primary treatment well")
    controls = raw[raw.compound_entity_id.eq("DMSO_TF") & raw.plate_id.isin(extra.plate_id)].copy()
    controls = controls.sort_values(KEY[0])
    if set(controls.groupby("plate_id").size()) != {2}:
        raise ValueError("Extra dose lacks paired physical DMSO")
    controls = controls.assign(control_role=controls.groupby("plate_id").cumcount().map({0: "A", 1: "B"}),
                               time_h=cfg["time_h"], split="control", fold=-1, quarantined=False)
    out.mkdir(parents=True, exist_ok=False)
    meta = out / "metadata"
    meta.mkdir()
    combined = pd.concat([extra, controls], ignore_index=True)
    combined[original.columns].to_csv(meta / "samples.tsv", sep="\t", index=False)
    for name in ("lines.tsv", "genes.tsv"):
        shutil.copyfile(r11 / "metadata" / name, meta / name)
    legacy = {"data": dict(root=str(thaw), workers=7, batch_cells=8192,
                          revision=cfg["revision"], minimum_cells=cfg["minimum_cells"])}
    counts(legacy, out, "source", out / "source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    args = parser.parse_args()
    cfg = config(args.config)
    if args.split is None:
        dose_manifest(cfg, args.out)
    else:
        partition(cfg, args.out, args.split)


if __name__ == "__main__":
    main()
