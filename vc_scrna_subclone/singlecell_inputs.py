"""Prepare donor-keyed malignant cells and held-out pharmaceutical transfer queries."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .cohorts import file_record
from .prepare_patient import strings
from .source_panels import frozen, partition


COHORTS = ("GSE149614", "GSE151530", "GSE125449-set1", "GSE125449-set2")


def stable_seed(key, seed):
    return int.from_bytes(sha256(f"scrna:{seed}:{key}".encode()).digest()[:8], "big")


def sparse_h5_rows(node, indices, n_genes):
    bounds = node["indptr"]
    at = bounds[indices]
    end = bounds[indices + 1]
    counts = end - at
    row_offsets = np.concatenate(([0], counts.cumsum()))
    values = np.empty(row_offsets[-1], np.float32)
    cols = np.empty_like(values, dtype=np.int32)
    for a, b, start, stop in zip(at, end, row_offsets[:-1], row_offsets[1:]):
        values[start:stop] = node["data"][a:b]
        cols[start:stop] = node["indices"][a:b]
    return sparse.csr_matrix((values, cols, row_offsets), shape=(len(indices), n_genes))


def tahoe(path, cell_registry, seed, count):
    crosswalk = pd.read_csv(cell_registry, sep="\t", dtype=str)
    rows = crosswalk[crosswalk.cellosaurus_id.eq("CVCL_0366") & crosswalk.included.eq("True")]
    if len(rows) != 1:
        raise ValueError("Cannot map SNU-423 to a single exact lineage and culture")
    parent = rows.iloc[0]
    with h5py.File(path, "r") as data:
        obs = data["obs"]
        drug = strings(obs, "drug")
        controls = np.flatnonzero(drug == "DMSO_TF")
        if len(controls) < 400:
            raise ValueError("Too few exact registered Tahoe vehicle cells")
        sample = strings(obs, "sample")
        labels = pd.DataFrame({"idx": controls, "sample": sample[controls]})
        groups = len(labels["sample"].unique())
        take = max(1, count // groups)
        rng = np.random.default_rng(seed)
        indices = np.sort(np.concatenate([
            rng.choice(group.idx.to_numpy(), size=min(take, len(group)), replace=False)
            for _, group in labels.groupby("sample", sort=True)
        ]))
        var = data["var"]
        genes = pd.Index(strings(var, var.attrs["_index"]))
        filtered = sparse_h5_rows(data["X"], indices, len(genes))
        barcode = obs[obs.attrs["_index"]].asstr()[indices]
    if len(indices) != len(set(barcode)):
        raise ValueError("Repeated Tahoe single cells in the DMSO query")
    meta = pd.DataFrame({
        "id": ["Tahoe:SNU423:" + str(b) for b in barcode], "sample": sample[indices],
        "role": "external_real_single_cell", "cell_id": parent.model_id, "cell_group": parent.family_id,
        "fold": partition(parent.family_id),
    })
    return filtered, genes, meta, len(controls)


def select_patient(cohort, metadata):
    hcc = metadata.histology.eq("HCC")
    technical = metadata.umi.ge(300) & metadata.detected_genes.ge(100) & metadata.mitochondrial_fraction.le(0.25)
    if cohort == "GSE149614":
        training_candidate = metadata.malignancy.eq("needs_cnv") & metadata.epithelial_marker_count.ge(2)
    else:
        training_candidate = metadata.malignancy.eq("author_malignant")
    return (hcc & technical & training_candidate).to_numpy()


def synthetic_profiles(bulk, queries, seed):
    output = []
    data = []
    for cell, rows in queries.groupby("cell_id", sort=True):
        roles = rows[["cell_group", "split"]].drop_duplicates()
        if len(roles) != 1:
            raise ValueError("A registered culture occurs in multiple pseudo-single-cell roles")
        expression = np.maximum(np.exp2(bulk.loc[cell].to_numpy(np.float64)) - 1, 0)
        if expression.sum() <= 0:
            raise ValueError(f"No bulk counts to downsample for {cell}")
        rng = np.random.default_rng(stable_seed(cell, seed))
        group, fold = roles.iloc[0].to_list()
        for depth in (1000, 4000):
            draw = rng.multinomial(depth, expression / expression.sum(), size=24)
            summed = draw.sum(axis=0, keepdims=True)
            data.extend((sparse.csr_matrix(draw, dtype=np.float32), sparse.csr_matrix(summed, dtype=np.float32)))
            for index in range(25):
                role = "pseudo_single_cell" if index < 24 else "pseudo_bulk"
                output.append({
                    "id": f"pseudo:{cell}:{depth}:{index}", "cell_id": cell, "cell_group": group,
                    "fold": int(fold), "role": role, "depth": depth,
                })
    return sparse.vstack(data, format="csr"), pd.DataFrame(output)


def files_receipt(root):
    return {str(p.relative_to(root)): file_record(p) for p in sorted(root.rglob("*")) if p.is_file()}


def save_counts(root, counts, keys, symbol_axis):
    if len(symbol_axis) < 3000 or counts.shape != (len(keys), len(symbol_axis)):
        raise ValueError(f"Wrong count matrix dimensions for {root}")
    root.mkdir()
    if keys.id.duplicated().any() or not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError(f"Invalid single-cell count rows for {root}")
    sparse.save_npz(root / "counts.npz", counts)
    pd.DataFrame({"symbol": symbol_axis}).to_csv(root / "genes.tsv", sep="\t", index=False)
    keys.to_parquet(root / "rows.parquet", index=False)


def prepare(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    patients = json.loads((a.patients / "receipt.json").read_text())
    source = json.loads((a.sources / "receipt.json").read_text())
    bulk = pd.read_parquet(frozen(a.sources, "cell_lines_rna.parquet", source))
    folds = pd.concat([pd.read_parquet(frozen(a.sources, f"fold-{i}/query.parquet", source))
                       for i in range(5)], ignore_index=True)
    raw_tahoe, tahoe_axis, tahoe_meta, controls = tahoe(a.tahoe, a.cell_registry, a.seed, 2000)
    common = set(bulk.columns) & set(tahoe_axis[~tahoe_axis.duplicated(keep=False)])
    patient_data = {}
    for name in COHORTS:
        gene_axis = pd.read_csv(frozen(a.patients, f"{name}/genes.tsv", patients), sep="\t")
        private = pd.read_parquet(frozen(a.patients, f"{name}/metadata.parquet", patients))
        axis = pd.Index(gene_axis.symbol)
        common &= set(axis[~axis.duplicated(keep=False)])
        patient_data[name] = (axis, private)
    symbols = pd.Index(sorted(common))
    if len(symbols) < 4000:
        raise ValueError("Too few shared gene symbols across bulk and independent assays")
    q_bulk = bulk.loc[:, symbols]
    pseudo, pseudometa = synthetic_profiles(q_bulk, folds, a.seed)
    summary = {}
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as temporary:
        stage = Path(temporary)
        save_counts(stage / "tahoe", raw_tahoe[:, unique_positions(tahoe_axis, symbols)], tahoe_meta, symbols)
        save_counts(stage / "pseudo", pseudo, pseudometa, symbols)
        for name, (axis, private) in patient_data.items():
            eligible = select_patient(name, private)
            metadata = private.loc[eligible].reset_index(drop=True)
            counts = sparse.load_npz(frozen(a.patients, f"{name}/counts.npz", patients))
            expression = counts[eligible][:, unique_positions(axis, symbols)]
            output = stage / name
            # Freeze only grouping information in the prediction interface.
            save_counts(output, expression, metadata[["id", "cohort", "donor", "fold"]], symbols)
            metadata.to_parquet(output / "scoring_metadata.parquet", index=False)
            summary[name] = {"eligible_cells": len(metadata), "donors": metadata.donor.nunique(),
                             "malignant_candidates": int(metadata.malignant_candidate.sum())}
        (stage / "receipt.json").write_text(json.dumps({
            "inputs": {"patients": file_record(a.patients / "receipt.json"),
                       "sources": file_record(a.sources / "receipt.json"),
                       "tahoe": file_record(a.tahoe), "cell_crosswalk": file_record(a.cell_registry)},
            "shared_genes": len(symbols), "tahoe_vehicle_cells_available": controls,
            "tahoe_vehicle_cells_used": len(tahoe_meta), "pseudo_lines": pseudometa.cell_group.nunique(),
            "pseudo_rows": len(pseudometa), "patient_counts": summary, "seed": a.seed,
            "files": files_receipt(stage),
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def unique_positions(axis, symbols):
    index = pd.Series(np.arange(len(axis)), index=axis)
    result = index[~axis.duplicated(keep=False)].reindex(symbols)
    if result.isna().any():
        raise ValueError("Unregistered duplicate gene symbol")
    return result.to_numpy(dtype=np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("patients", "sources", "tahoe", "cell-registry", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seed", type=int, default=51122)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
