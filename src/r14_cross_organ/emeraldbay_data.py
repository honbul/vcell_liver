"""Build EmeraldBay liver pseudobulk targets from the archived Arrow release."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

from src.r11_chemical_probe.common import sha256
from src.r14_cross_organ.donor_data import PROTOCOL, ensemble, identity, logcp, write_output


LINES = ["CVCL_0366", "CVCL_1098"]
CONTROL = "DMSO_TF"
COLUMNS = ["cell_line", "sample", "drug", "drugname_drugconc"]


def profile_shard(path, width):
    file = pq.ParquetFile(path)
    counts, info, cells = {}, {}, Counter()
    for batch in file.iter_batches(batch_size=8192, columns=["genes", "expressions", *COLUMNS]):
        frame = pa.Table.from_batches([batch])
        selected = frame.filter(pc.is_in(frame["cell_line"], value_set=pa.array(LINES)))
        if not len(selected):
            continue
        meta = selected.select(COLUMNS).to_pandas()
        groups = pd.MultiIndex.from_frame(meta[COLUMNS[:2]])
        group_codes, keys = pd.factorize(groups, sort=True)
        feature = selected["genes"].combine_chunks()
        value = selected["expressions"].combine_chunks()
        idx = pc.list_flatten(feature).to_numpy()
        data = pc.list_flatten(value).to_numpy()
        if (not np.array_equal(feature.offsets.to_numpy(), value.offsets.to_numpy())
                or idx.max() >= width or idx.min() < 0 or np.any(data < 0) or not np.all(np.isfinite(data))):
            raise ValueError(f"Invalid released count arrays in {path.name}")
        per_cell = np.repeat(group_codes, np.diff(feature.offsets.to_numpy()))
        pseudobulk = sparse.coo_matrix((data, (per_cell, idx)), shape=(len(keys), width)).toarray()
        grouped = meta.drop_duplicates().set_index(COLUMNS[:2], verify_integrity=True)
        for i, key in enumerate(keys):
            label = tuple(grouped.loc[key, COLUMNS[2:]])
            if key in info and info[key] != label:
                raise ValueError("Multiple condition annotations for one physical well and line")
            info[key] = label
            cells[key] += int(np.count_nonzero(group_codes == i))
            if key in counts:
                counts[key] += pseudobulk[i]
            else:
                counts[key] = pseudobulk[i].copy()
    return counts, info, cells


def profiles(paths, width, workers):
    sums, info, cells = {}, {}, Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (partial, label, count) in enumerate(pool.map(lambda p: profile_shard(p, width), paths)):
            overlap = info.keys() & label.keys()
            if any(info[key] != label[key] for key in overlap):
                raise ValueError("Sample annotation differs between EmeraldBay shards")
            info.update(label)
            cells.update(count)
            for key, val in partial.items():
                if key in sums:
                    sums[key] += val
                else:
                    sums[key] = val
            if i % 10 == 0:
                print(f"EmeraldBay {i+1}/{len(paths)} shards", flush=True)
    return sums, info, cells


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "protocol", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if sha256(args.protocol) != PROTOCOL:
        raise ValueError("Execution protocol mismatch")
    ref_file = args.source / "metadata/gene_metadata.parquet"
    drug_file = args.source / "metadata/drug_metadata.parquet"
    manifest = args.source / "filelist.tsv"
    paths = sorted((args.source / "expression_data").glob("*.parquet"))
    listed = dict(pd.read_csv(manifest, sep="\t", header=None).itertuples(index=False, name=None))
    if len(paths) != 116 or any(p.stat().st_size != listed[str(p.relative_to(args.source))] for p in paths):
        raise ValueError("Release paths or sizes differ from filelist")
    ref = pd.read_parquet(ref_file)
    take, genes = ensemble(ref.ensembl_id)
    ids = ref.token_id.iloc[take].to_numpy()
    drug = pd.read_parquet(drug_file).set_index("drug", verify_integrity=True)
    sums, labels, n_cells = profiles(paths, int(ref.token_id.max()) + 1, args.workers)
    keys = sorted(sums)
    # The library size includes non-mapped expression; no DMSO_T0 enters the endpoint contrast.
    normalized, totals = logcp(np.stack([sums.pop(key) for key in keys]))
    v = {key: normalized[i, ids] for i, key in enumerate(keys) if totals[i] > 0 and n_cells[key] >= 20}
    ctrl = {}
    for line in LINES:
        valid = [v[key] for key in v if key[0] == line and labels[key][0] == CONTROL]
        if len(valid) < 2:
            raise ValueError(f"Insufficient physical endpoint vehicle wells: {line}")
        ctrl[line] = np.mean(valid, axis=0)
    rows, deltas, controls = [], [], []
    excluded = Counter()
    for (line, sample), val in v.items():
        name, condition = labels[line, sample]
        if name.startswith("DMSO"):
            continue
        units = ast.literal_eval(condition)
        if len(units) != 1:
            excluded["combinationRows"] += 1
            continue
        _, dose, unit = units[0]
        if unit != "uM" or dose <= 0 or name not in drug.index:
            excluded["metadataRows"] += 1
            continue
        # A disconnected platinum record does not encode the intact assay compound.
        key, smi = identity(drug.at[name, "canonical-smiles"]) if name != "Oxaliplatin" else ("", "")
        if not key:
            excluded["nonfingerprintableRows"] += 1
            continue
        rows.append(dict(parent_key=key, compound=name, smiles=smi, cell_line_id=line,
                         dose_uM=float(dose), time_h=120, plate_id="EmeraldBay/" + line,
                         physical_well_id="EmeraldBay/" + sample, n_cells=int(n_cells[line,sample]),
                         scope="external"))
        deltas.append(val - ctrl[line])
        controls.append(ctrl[line])
    write_output(args.out, pd.DataFrame(rows), genes, np.stack(deltas), np.stack(controls),
                 [manifest, ref_file, drug_file],
                 dict(expression_input_shards=len(paths), expression_size_validation="filelist.tsv",
                      actual_plate_location="not in the release expression rows",
                      plate_id="line-specific release-level vehicle pool, not asserted physical plate",
                      response="log1p CP10k, endpoint DMSO_TF mean over physical control samples",
                      qc="released cells, min20 cells per physical well and line", excluded=dict(excluded)))


if __name__ == "__main__":
    main()
