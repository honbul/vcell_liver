"""Register source-only and gene-cold HepG2 prediction panels."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .perturbseq import record


OUTCOMES = "outcome_delta.npy"
CONTEXT = "features_control_context.npy"


def validate_source(directory):
    receipt = json.loads((directory / "receipt.json").read_text())
    for name in ("genes.tsv", "profiles.parquet", OUTCOMES, CONTEXT):
        if record(directory / name) != receipt["files"][name]:
            raise ValueError(f"Invalid assay input {directory / name}")
    genes = pd.read_csv(directory / "genes.tsv", sep="\t").set_index("gene_id")
    metadata = pd.read_parquet(directory / "profiles.parquet")
    delta = np.load(directory / OUTCOMES, allow_pickle=False, mmap_mode="r")
    anchor = np.load(directory / CONTEXT, allow_pickle=False, mmap_mode="r")
    if tuple(delta.shape) != (len(metadata), len(genes)) or delta.shape != anchor.shape:
        raise ValueError(f"Misaligned source arrays {directory}")
    if metadata.assay.nunique() != 1 or metadata.duplicated(["target_ensembl", "gem_group"]).any():
        raise ValueError(f"Ambiguous interventions {directory}")
    return genes, metadata, delta, anchor


def gene_order(gene, seed):
    return sha256(f"track4-gene-cold:{seed}:{gene}".encode()).digest()


def summarize_assay(data, common):
    genes, meta, values, anchors = data
    indices = genes.index.get_indexer(common)
    if np.any(indices < 0):
        raise ValueError("Missing registered output gene")
    rows, outcomes, contexts = [], [], []
    for gene, group in meta.groupby("target_ensembl", sort=True):
        weights = group.cells.to_numpy()
        match = group.index.to_numpy()
        aliases = sorted({symbol for field in group.target_symbol for symbol in field.split("|") if symbol})
        assay = str(group.assay.iloc[0])
        rows.append({
            "id": f"{assay}:{gene}", "assay": assay, "target_ensembl": gene,
            "target_symbols": "|".join(aliases), "n_cells": int(weights.sum()),
            "gem_groups": len(group),
        })
        outcomes.append(np.average(values[np.ix_(match, indices)], weights=weights, axis=0))
        contexts.append(np.average(anchors[np.ix_(match, indices)], weights=weights, axis=0))
    return pd.DataFrame(rows), np.asarray(outcomes, dtype=np.float32), np.asarray(contexts, dtype=np.float32)


def write_role(root, role, rows, outcomes, context, use):
    root = root / role
    root.mkdir()
    rows.loc[use].to_parquet(root / "identities.parquet", index=False)
    np.save(root / "control_context.npy", context[use], allow_pickle=False)
    if outcomes is not None:
        np.save(root / "delta.npy", outcomes[use], allow_pickle=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path, nargs="+")
    p.add_argument("--hepg2", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=41026)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    train = [validate_source(s) for s in a.source]
    test = validate_source(a.hepg2)
    source_assays = [x[1].assay.iloc[0] for x in train]
    if test[1].assay.iloc[0] in source_assays or len(set(source_assays)) != len(source_assays):
        raise ValueError("Duplicate assay identity or target assay in source training")
    common = sorted(set.intersection(*(set(source[0].index) for source in [*train, test])))
    if len(common) < 20:
        raise ValueError("Too few shared measured transcript genes")
    train_roles = [summarize_assay(record, common) for record in train]
    meta = pd.concat([x[0] for x in train_roles], ignore_index=True)
    y = np.concatenate([x[1] for x in train_roles])
    anchors = np.concatenate([x[2] for x in train_roles])
    q, qy, _ = summarize_assay(test, common)
    background_path = a.hepg2 / "features_cell_background.npy"
    background_receipt = json.loads((a.hepg2 / "receipt.json").read_text())
    if record(background_path) != background_receipt["files"][background_path.name]:
        raise ValueError("Invalid HepG2 control-only reference")
    reference = np.load(background_path, allow_pickle=False)
    indices = test[0].index.get_indexer(common)
    if reference.shape != (len(test[0]),):
        raise ValueError("HepG2 control reference has incompatible output genes")
    # Query conditions all get the same independently observed negative control.
    # GEM coverage and cell counts of the treated observations remain scorer-side.
    qx = np.broadcast_to(reference[indices], (len(q), len(common)))
    external_identities = q[["id", "assay", "target_ensembl", "target_symbols"]]

    all_source = set(meta.target_ensembl)
    external_targets = set(q.target_ensembl)
    shared = sorted(all_source & external_targets, key=lambda g: gene_order(g, a.seed))
    unavailable = external_targets - all_source
    if len(shared) < 10:
        raise ValueError("Too little same-gene overlap for matched transfer benchmarks")
    cold = set(shared[::5]) | unavailable
    seen = external_targets - cold
    fit_pool = all_source - cold
    # Keep every same-gene evaluation gene represented in source training.
    # Tune on source genes that will never appear in either HepG2 panel.
    validation_pool = fit_pool - external_targets
    if len(validation_pool) < 10:
        raise ValueError("Too few source-only targets for gene-cold tuning")
    inner_valid = set(sorted(validation_pool, key=lambda g: gene_order(g, a.seed + 1))[::5])
    inner_train = fit_pool - inner_valid
    if (cold & fit_pool or cold & seen or inner_train & inner_valid or not seen <= inner_train
            or external_targets & inner_valid or not seen or not cold
            or set(meta.loc[meta.target_ensembl.isin(fit_pool), "target_ensembl"]) & cold):
        raise ValueError("Invalid source-to-HepG2 gene split")

    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        out = Path(directory)
        pd.DataFrame({"gene_id": common}).to_csv(out / "genes.tsv", sep="\t", index=False)
        write_role(out, "fit", meta, y, anchors, meta.target_ensembl.isin(inner_train).to_numpy())
        write_role(out, "validation", meta, y, anchors, meta.target_ensembl.isin(inner_valid).to_numpy())
        for name, targets in (("gene_cold", cold), ("same_gene", seen)):
            mask = q.target_ensembl.isin(targets).to_numpy()
            write_role(out, "query_" + name, external_identities, None, qx, mask)
            labels = out / ("labels_" + name)
            labels.mkdir()
            np.save(labels / "delta.npy", qy[mask], allow_pickle=False)
            q.loc[mask, ["id", "n_cells", "gem_groups"]].to_parquet(labels / "identities.parquet", index=False)
        receipt = {
            "source_assays": source_assays,
            "external_context": {"assay_control_only": record(background_path)},
            "assays": {x.name: record(x / "receipt.json") for x in [*a.source, a.hepg2]},
            "identity_level": "Human Ensembl stable gene",
            "hcc_context": "HepG2 is a hepatoblastoma-derived line. Source-only supervised fit.",
            "cold_genes": len(cold), "natural_unseen_genes": len(unavailable),
            "same_genes": len(seen), "training_genes": len(inner_train),
            "validation_genes": len(inner_valid), "output_genes": len(common),
            "seed": a.seed, "files": {str(f.relative_to(out)): record(f) for f in sorted(out.rglob("*")) if f.is_file()},
        }
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        out.rename(a.output)


if __name__ == "__main__":
    main()
