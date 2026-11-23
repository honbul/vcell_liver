"""Build exact-parent nonhepatic Tahoe profiles using reproducibly thinned single cells."""
import argparse
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import RDLogger

from nextgen_h2.assets import COLD, KEY, LIMORE, LINCS, canonical_chemistry
from nextgen_h2.run import read_json
from vc_build.io import sha256
from .assets import TAHOE, V1_ASSETS, publish, registries

META = TAHOE / "metadata"
PARTS = 16
HASH_MODULUS = 100
HASH_KEEP = 6
MIN_CELLS = 16
DOSES = (.05, .5, 5.)


def register(out):
    registry, gdsc = registries()
    licob = pd.read_parquet(COLD / "query.parquet", columns=[KEY])
    limore = pd.read_parquet(LIMORE / "test_inputs/rows.parquet", columns=[KEY, "cellosaurus_id"])
    roster = set(registry.index) | set(gdsc[KEY]) | set(licob[KEY]) | set(limore[KEY])
    metadata = pd.read_parquet(META / "cell_line_metadata.parquet")
    cells = metadata[["Cell_ID_Cellosaur", "Cell_ID_DepMap", "cell_name", "Organ"]].drop_duplicates()
    if not cells.Cell_ID_Cellosaur.is_unique:
        raise ValueError("Ambiguous Tahoe line identity")
    lower = cells.Organ.str.lower()
    allowed = cells[lower.notna() & ~lower.str.contains("liver|hepatic|unknown", na=True)
                    & ~cells.Cell_ID_Cellosaur.isin(limore.cellosaurus_id)]
    drugs = pd.read_parquet(META / "drug_metadata.parquet").dropna(subset=["canonical_smiles"])

    def exact(value):
        try:
            return canonical_chemistry(value)[1]
        except (ValueError, TypeError, RuntimeError):
            return ""

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    drugs[KEY] = drugs.canonical_smiles.map(exact)
    drugs = drugs.loc[drugs[KEY].ne("")]
    if not drugs.drug.is_unique:
        raise ValueError("Ambiguous Tahoe drug names")
    samples = pd.read_parquet(META / "sample_metadata.parquet")
    condition = samples.drugname_drugconc.map(ast.literal_eval)
    samples["vehicle"] = samples.drug.str.startswith("DMSO")
    samples[KEY] = samples.drug.map(drugs.set_index("drug")[KEY])
    samples["dose"] = condition.map(lambda v: v[0][1] if len(v) == 1 and v[0][2] == "uM" else np.nan)
    samples = samples.loc[samples.vehicle | (samples[KEY].isin(roster) & samples.dose.isin(DOSES))]
    samples = samples.loc[:, ["sample", "plate", "drug", KEY, "dose", "vehicle"]]
    if not samples["sample"].is_unique:
        raise ValueError("Ambiguous Tahoe samples")
    genes = pd.read_parquet(META / "gene_metadata.parquet")
    landmark = pd.read_csv(LINCS / "geneinfo_beta.txt", sep="\t")
    sys = sorted(set(landmark.loc[landmark.feature_space.eq("landmark"), "gene_symbol"]) & set(genes.gene_symbol))
    genes = genes[genes.gene_symbol.isin(sys)]
    out.mkdir(exist_ok=False, parents=True)
    genes.to_parquet(out / "genes.parquet", index=False)
    samples.to_parquet(out / "samples.parquet", index=False)
    allowed.to_parquet(out / "cells.parquet", index=False)
    files = sorted((TAHOE / "data").glob("train-*.parquet"))
    if len(files) != 3388:
        raise ValueError("Tahoe raw raw-count shard manifest is incomplete")
    (out / "shards.txt").write_text("\n".join(str(path) for path in files) + "\n")
    publish(out, {"dataset": "Tahoe-100M local Parquet release", "lineages": len(allowed),
                  "excluded_cells": sorted(set(cells.cell_name) - set(allowed.cell_name)),
                  "all_atlas_parents_resolved": drugs[KEY].nunique(),
                  "matched_drp_exact_parents": len(set(samples[KEY]) & set(registry.index)),
                  "query_parents_including_external": samples[KEY].nunique(),
                  "samples": len(samples), "plates": samples.plate.nunique(), "shards": len(files),
                  "leaf_bytes": sum(f.stat().st_size for f in files), "landmarks_shared": len(sys),
                  "hash_modulus": HASH_MODULUS, "hash_keep": HASH_KEEP,
                  "metadata_sha256": {name: sha256(META / f"{name}.parquet") for name in
                                      ("drug_metadata", "cell_line_metadata", "sample_metadata", "gene_metadata")}})


def elements(lst):
    return lst.offsets.to_numpy(zero_copy_only=False), lst.values.to_numpy(zero_copy_only=False)


def map_shards(contract, out, part):
    samples = pd.read_parquet(contract / "samples.parquet")
    cells = pd.read_parquet(contract / "cells.parquet")
    genes = pd.read_parquet(contract / "genes.parquet")
    features = pd.Index(sorted(genes.gene_symbol.unique()))
    sample_index = pd.Index(samples["sample"])
    cell_index = pd.Index(cells.Cell_ID_Cellosaur)
    map_gene = np.full(int(genes.token_id.max()) + 1, -1, dtype=np.int32)
    map_gene[genes.token_id.to_numpy()] = features.get_indexer(genes.gene_symbol)
    n = len(samples) * len(cells)
    shape = (n, len(features))
    summed = np.zeros(shape, np.float32)
    cell_sum = np.zeros(shape, np.float32)
    cell_square = np.zeros(shape, np.float32)
    umi = np.zeros(n, np.float64)
    count = np.zeros(n, np.int64)
    paths = (contract / "shards.txt").read_text().splitlines()[part::PARTS]
    selected_total = 0
    for i, path in enumerate(paths):
        shard = pq.ParquetFile(path)
        # Minimize nested list I/O for ineligible rows and uninterested row groups.
        for row_group in range(shard.num_row_groups):
            ids = shard.read_row_group(row_group, columns=["sample", "cell_line_id", "BARCODE_SUB_LIB_ID"]).to_pandas()
            s = sample_index.get_indexer(ids["sample"])
            c = cell_index.get_indexer(ids.cell_line_id)
            hash_value = pd.util.hash_pandas_object(ids.BARCODE_SUB_LIB_ID, index=False).to_numpy()
            position = np.flatnonzero((s >= 0) & (c >= 0) & (hash_value % HASH_MODULUS < HASH_KEEP))
            if not len(position):
                continue
            part_data = shard.read_row_group(row_group, columns=["genes", "expressions"]).take(pa.array(position))
            gene_offset, gene_ids = elements(part_data["genes"].combine_chunks())
            offset, values = elements(part_data["expressions"].combine_chunks())
            if not np.array_equal(offset, gene_offset):
                raise ValueError("Tahoe sparse counts disagree with gene keys")
            prefix = np.r_[0, np.cumsum(values, dtype=np.int64)]
            cell_umi = np.diff(prefix[offset])
            good = (cell_umi >= 500) & (cell_umi <= 100000) & (np.diff(offset) >= 200)
            cell_id = c[position] * len(samples) + s[position]
            np.add.at(count, cell_id[good], 1)
            np.add.at(umi, cell_id[good], cell_umi[good])
            rows = np.repeat(np.arange(len(position), dtype=np.int32), np.diff(offset))
            feasible_gene = (gene_ids < len(map_gene)) & (gene_ids >= 0)
            match = np.full(len(gene_ids), -1, np.int32)
            match[feasible_gene] = map_gene[gene_ids[feasible_gene]]
            selected = (match >= 0) & good[rows]
            observed_row = rows[selected]
            ix = (cell_id[observed_row], match[selected])
            observed_value = values[selected]
            logged = np.log1p(observed_value * (10000. / cell_umi[observed_row]))
            np.add.at(summed, ix, observed_value)
            np.add.at(cell_sum, ix, logged)
            np.add.at(cell_square, ix, logged * logged)
            selected_total += int(len(position))
        if i % 20 == 0:
            print(part, i, len(paths), int(count.sum()), flush=True)
    use = np.flatnonzero(count)
    out.mkdir(exist_ok=False, parents=True)
    np.savez_compressed(out / "moments.npz", index=use, count=count[use], umi=umi[use],
                        sums=summed[use], cell_sum=cell_sum[use], cell_square=cell_square[use])
    publish(out, {"part": part, "raw_shards": len(paths), "barcode_sample_before_QC": selected_total,
                  "cells_after_qc": int(count.sum()), "nonempty_sample_lineages": len(use),
                  "frozen_tahoe_manifest_sha256": sha256(contract / "receipt.json")})


def assemble(contract, parts, out):
    samples = pd.read_parquet(contract / "samples.parquet")
    cells = pd.read_parquet(contract / "cells.parquet")
    genes = pd.Index(sorted(pd.read_parquet(contract / "genes.parquet").gene_symbol.unique()))
    size = len(samples) * len(cells)
    count, umi = np.zeros(size, np.int64), np.zeros(size, np.float64)
    moments = [np.zeros((size, len(genes)), np.float32) for _ in range(3)]
    for i in range(PARTS):
        path = parts / f"{i:02d}"
        manifest = read_json(path / "receipt.json")
        if manifest["frozen_tahoe_manifest_sha256"] != sha256(contract / "receipt.json") or manifest["part"] != i:
            raise ValueError("Unexpected Tahoe map receipt")
        if sha256(path / "moments.npz") != manifest["artifacts"]["moments.npz"]:
            raise ValueError("Changed Tahoe raw moments")
        with np.load(path / "moments.npz") as block:
            take = block["index"]
            count[take] += block["count"]
            umi[take] += block["umi"]
            for dest, key in zip(moments, ("sums", "cell_sum", "cell_square")):
                dest[take] += block[key]
    active = np.flatnonzero(count >= MIN_CELLS)
    tags = samples.iloc[active % len(samples)].reset_index(drop=True)
    tags["lineage"] = cells.Cell_ID_Cellosaur.to_numpy()[active // len(samples)]
    tags["n_cells"] = count[active]
    aggregate = np.log1p(moments[0][active] * (10000. / umi[active, None]))
    mean = moments[1][active] / count[active, None]
    sd = np.sqrt(np.maximum(moments[2][active] / count[active, None] - mean ** 2, 0.))
    ctl = tags[tags.vehicle].groupby(["lineage", "plate"]).indices
    ctl_index = np.flatnonzero(tags.vehicle)
    centers = {key: [a[ctl_index[idx]].mean(axis=0) for a in (aggregate, sd)]
               for key, idx in ctl.items()}
    retain, change, variability = [], [], []
    for i, row in tags[~tags.vehicle].iterrows():
        key = (row.lineage, row.plate)
        if key in centers:
            retain.append(i)
            change.append(aggregate[i] - centers[key][0])
            variability.append(sd[i] - centers[key][1])
    t = tags.loc[retain].reset_index(drop=True)
    if t.empty:
        raise ValueError("No Tahoe treatment/DMSO pairing survived")
    change = np.asarray(change, np.float32)
    variability = np.asarray(variability, np.float32)
    out.mkdir(exist_ok=False, parents=True)
    results = {}
    def pack(name, x, labels):
        frame = pd.DataFrame(x, index=pd.Index(labels, name=KEY))
        frame.columns = [f"x{i}" for i in range(frame.shape[1])]
        frame.to_parquet(out / f"{name}.parquet")
        results[name] = {"parents": len(frame), "features": len(frame.columns)}
    dose_groups = t.groupby([KEY, "lineage", "dose"], sort=True).indices
    keys = pd.MultiIndex.from_tuples(list(dose_groups))
    dose_mu = np.stack([change[idx].mean(axis=0) for idx in dose_groups.values()])
    dose_sd = np.stack([variability[idx].mean(axis=0) for idx in dose_groups.values()])
    per_cell = pd.Series(np.arange(len(keys)), index=keys).groupby(level=[0, 1]).indices
    per_cell_mu = np.stack([dose_mu[idx].mean(axis=0) for idx in per_cell.values()])
    per_cell_sd = np.stack([dose_sd[idx].mean(axis=0) for idx in per_cell.values()])
    by_drug = pd.Series(np.arange(len(per_cell)), index=pd.MultiIndex.from_tuples(list(per_cell))).groupby(level=0).indices
    pack("mean", np.stack([per_cell_mu[idx].mean(axis=0) for idx in by_drug.values()]), list(by_drug))
    eligible = {drug: idx for drug, idx in by_drug.items() if len(idx) >= 2}
    pack("heterogeneity", np.stack([np.r_[per_cell_mu[idx].mean(axis=0), per_cell_mu[idx].std(axis=0),
                                         per_cell_sd[idx].mean(axis=0)] for idx in eligible.values()]), list(eligible))
    slopes = {}
    dose_axis = keys.get_level_values(2).to_numpy(dtype=float)
    # Estimate matched cell dose slopes on the registered log-dose axis, then average cells.
    for (drug, _), idx in per_cell.items():
        if len(idx) < 2:
            continue
        x = np.log10(dose_axis[idx])
        x -= x.mean()
        slopes.setdefault(drug, []).append(x @ dose_mu[idx] / (x @ x))
    pack("dose", np.stack([np.r_[per_cell_mu[by_drug[key]].mean(axis=0), np.mean(x, axis=0)]
                           for key, x in slopes.items()]), list(slopes))
    registry, _ = registries()
    for name in results:
        keys = pd.read_parquet(out / (name + ".parquet"), columns=[]).index
        matched = registry.loc[registry.index.intersection(keys)]
        results[name]["drp_parents"] = len(matched)
        results[name]["drp_scaffolds"] = matched.scaffold.nunique()
    t.to_parquet(out / "paired_wells.parquet", index=False)
    publish(out, {"registered_sha256": sha256(contract / "receipt.json"), "cells_after_qc": int(count.sum()),
                  "minimum_cells": MIN_CELLS, "matched_treatment_wells": len(t), "matched_cells": t.lineage.nunique(),
                  "variants": results, "dose_values": list(DOSES), "raw_shards": 3388})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for action, names in {"register": ("out",), "map": ("contract", "out"),
                          "assemble": ("contract", "parts", "out")}.items():
        child = sub.add_parser(action)
        for name in names:
            child.add_argument(name, type=Path)
        if action == "map":
            child.add_argument("--part", type=int)
    a = parser.parse_args()
    if a.stage == "register":
        register(a.out)
    elif a.stage == "map":
        part = a.part if a.part is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        map_shards(a.contract, a.out / f"{part:02d}", part)
    else:
        assemble(a.contract, a.parts, a.out)
