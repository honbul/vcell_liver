"""Stream public dense H5ADs into matched-control gene intervention profiles."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

CONTROL = "non-targeting"


def record(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def field(group, name):
    ds = group[name]
    if isinstance(ds, h5py.Group):
        categories = ds["categories"].asstr()[:]
        code = ds["codes"][:]
    elif ds.dtype.kind in "ui" and "__categories" in group and name in group["__categories"]:
        categories = group["__categories"][name].asstr()[:]
        code = ds[:]
    elif ds.dtype.kind in "OUS":
        return ds.asstr()[:]
    else:
        return ds[:]
    values = np.full(len(code), "", dtype=object)
    valid = code >= 0
    values[valid] = categories[code[valid]]
    return values


def axis(group):
    raw = field(group, "gene_id")
    symbols = field(group, "gene_name")
    matrix = pd.DataFrame({"gene_id": [g.split(".")[0] for g in raw], "symbol": symbols})
    if matrix["gene_id"].duplicated().any() or not matrix["gene_id"].str.startswith("ENSG").all():
        raise ValueError("Invalid or duplicated Ensembl expression identifiers")
    return matrix


def collapse_h5ad(path, assay, minimum_cells, minimum_match, chunk_rows):
    with h5py.File(path, "r") as h:
        counts = h["X"]
        if not isinstance(counts, h5py.Dataset) or len(counts.shape) != 2:
            raise ValueError(f"Expected the publisher's dense H5AD expression matrix: {path}")
        obs = h["obs"]
        ids = field(obs, "gene_id")
        genes = field(obs, "gene")
        plates = field(obs, "gem_group")
        genes_axis = axis(h["var"])
        if len(ids) != counts.shape[0] or counts.shape[1] != len(genes_axis):
            raise ValueError("Expression and annotation dimensions differ")
        target = np.asarray([i.split(".")[0] for i in ids], dtype=str)
        # One row per guide-target/GEM group is combined by its number of
        # cells. A published target identity and matched negative controls
        # are required for measured effects. GEM groups are technical units.
        control = (genes == CONTROL) & (target == CONTROL)
        valid_gene = np.asarray([bool(re.fullmatch(r"ENSG[0-9]+", g)) for g in target])
        target_table = pd.DataFrame({"gene_id": target[valid_gene], "symbol": genes[valid_gene]})
        aliases = target_table.groupby("gene_id").symbol.agg(lambda g: sorted(set(g)))
        if not np.any(control):
            raise ValueError(f"No registered negative control in {path}")
        control_by_plate = Counter(map(str, plates[control]))
        legal_plate = {p for p, n in control_by_plate.items() if n >= minimum_match}
        targets = [(str(g), str(p)) if keep and str(p) in legal_plate else None
                   for g, p, keep in zip(target, plates, valid_gene)]
        for i in np.flatnonzero(control):
            p = str(plates[i])
            if p in legal_plate:
                targets[i] = (CONTROL, p)
        sizes = Counter(key for key in targets if key is not None)
        included = sorted(key for key, n in sizes.items()
                          if key[0] == CONTROL or n >= minimum_cells)
        if not included:
            raise ValueError("No perturbations with matched controls")
        groups = {key: i for i, key in enumerate(included)}
        code = np.asarray([groups.get(k, -1) for k in targets])
        totals = np.zeros((len(groups), counts.shape[1]), dtype=np.float64)
        ns = np.zeros(len(groups), dtype=np.int64)
        for start in range(0, len(code), chunk_rows):
            stop = min(start + chunk_rows, len(code))
            choose = np.flatnonzero(code[start:stop] >= 0)
            if not len(choose):
                continue
            entries = counts[start:stop][choose].astype(np.float32)
            if (entries < 0).any() or not np.isfinite(entries).all():
                raise ValueError("Nonfinite or negative raw counts")
            library = entries.sum(axis=1)
            nonzero = library > 0
            entries = np.log1p(entries[nonzero] * (1e4 / library[nonzero, None]))
            batch_ids = code[start + choose[nonzero]]
            present, positions, counts_in_chunk = np.unique(
                batch_ids, return_inverse=True, return_counts=True,
            )
            selector = csr_matrix(
                (np.ones(len(entries)), (positions, np.arange(len(entries)))),
                shape=(len(present), len(entries)),
            )
            totals[present] += selector @ entries
            ns[present] += counts_in_chunk

    metadata, effects, anchor_rows = [], [], []
    pooled = np.zeros(len(genes_axis), dtype=np.float64)
    reference_n = 0
    for (target_id, plate), index in groups.items():
        if target_id == CONTROL and ns[index] >= minimum_match:
            reference_n += int(ns[index])
            pooled += totals[index]
    if not reference_n:
        raise ValueError(f"No matched negative-control counts left for {assay}")
    for (target_id, plate), i in groups.items():
        if target_id == CONTROL or ns[i] < minimum_cells:
            continue
        ci = groups[(CONTROL, plate)]
        if ns[ci] < minimum_match:
            continue
        anchor = totals[ci] / ns[ci]
        delta = totals[i] / ns[i] - anchor
        effects.append(delta.astype(np.float32))
        anchor_rows.append(anchor.astype(np.float32))
        metadata.append({"assay": assay, "target_ensembl": target_id,
                         "target_symbol": "|".join(aliases.get(target_id, [])),
                         "gem_group": plate, "cells": int(ns[i]), "control_cells": int(ns[ci])})
    if not effects:
        raise ValueError(f"No measured profiles left for {assay}")
    audit = {
        "assay": assay, "raw_cells": len(target), "raw_controls": int(control.sum()),
        "matched_controls": reference_n, "used_gem_groups": len(set(x["gem_group"] for x in metadata)),
        "measured_targets": len(set(x["target_ensembl"] for x in metadata)),
        "profiles": len(metadata), "genes": len(genes_axis),
        "measurement": "mean single-cell log1p(CP10k) minus same-GEM-group negative control mean",
        "identity": "Ensembl stable gene identifier; all GEM groups for one target stay in one split",
    }
    return genes_axis, pd.DataFrame(metadata), np.stack(effects), np.stack(anchor_rows), pooled / reference_n, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--assay", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--min-treated", type=int, default=8)
    p.add_argument("--min-control", type=int, default=20)
    p.add_argument("--chunk-rows", type=int, default=1024)
    a = p.parse_args()
    if min(a.min_treated, a.min_control, a.chunk_rows) < 1:
        p.error("Thresholds and streaming block must be positive")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    feature, labels, effect, anchors, background, receipt = collapse_h5ad(
        a.source, a.assay, a.min_treated, a.min_control, a.chunk_rows,
    )
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        stage = Path(directory)
        feature.to_csv(stage / "genes.tsv", sep="\t", index=False)
        labels.to_parquet(stage / "profiles.parquet", index=False)
        np.save(stage / "outcome_delta.npy", effect, allow_pickle=False)
        np.save(stage / "features_control_context.npy", anchors, allow_pickle=False)
        np.save(stage / "features_cell_background.npy", background.astype(np.float32), allow_pickle=False)
        receipt["input"] = record(a.source)
        receipt["files"] = {p.name: record(p) for p in sorted(stage.iterdir())}
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


if __name__ == "__main__":
    main()
