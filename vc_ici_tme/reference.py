"""Build donor-equally weighted HCC cellular reference and LIANA prior."""
import argparse
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from .cohorts import record


TYPES = ("malignant", "cd8", "treg", "tam", "caf", "endothelial", "other")
TRACK5 = ("GSE149614", "GSE151530")
CURATED_EDGES = {
    ("PVR", "TIGIT"): "https://doi.org/10.1073/pnas.1120606109",
    ("NECTIN2", "TIGIT"): "https://pmc.ncbi.nlm.nih.gov/articles/PMC5500806/",
}
REGULATORS = {
    "VEGFA_KDR": {"VEGFA", "KDR"},
    "TGFB1": {"TGFB1", "TGFBR1", "TGFBR2"},
    "SPP1_CD44": {"SPP1", "CD44"},
    "WNT": {"CTNNB1", "AXIN2", "LRP5", "LRP6", "WNT1", "WNT2", "WNT3A", "WNT5A",
            "FZD1", "FZD2", "FZD4", "FZD5", "FZD7"},
    "TIGIT": {"TIGIT", "PVR", "NECTIN2"},
    "TIM3": {"HAVCR2", "LGALS9", "CEACAM1"},
    "LAG3": {"LAG3", "FGL1", "HLA-DRA", "HLA-DRB1"},
}


def checked(root, path, receipt):
    file = root / path
    if record(file) != receipt["files"][path]:
        raise ValueError(f"Unverified HCC reference source {file}")
    return file


def classify(counts, symbols, meta):
    def expressed(marker):
        columns = np.flatnonzero(symbols.isin(marker))
        if not len(columns):
            raise ValueError(f"No representative immune reference genes in {marker}")
        return np.asarray(counts[:, columns].sum(axis=1)).ravel() > 0

    category = meta.category.to_numpy()
    subtype = np.full(len(meta), "other", dtype="U12")
    t_cell = (category == "t_nk") & expressed(("CD3D", "CD3E", "TRAC"))
    treg = t_cell & (expressed(("FOXP3",)) |
                    (expressed(("CD4",)) & expressed(("IL2RA",)) & expressed(("CTLA4",))))
    cd8 = t_cell & ~treg & expressed(("CD8A", "CD8B"))
    for label, mask in (
        ("malignant", meta.malignant_candidate.to_numpy()),
        ("cd8", cd8), ("treg", treg), ("tam", category == "myeloid"),
        ("caf", category == "fibroblast"), ("endothelial", category == "endothelial"),
    ):
        subtype[mask] = label
    return subtype


def load_cohorts(path):
    receipt = json.loads((path / "receipt.json").read_text())
    if not set(TRACK5) <= set(receipt["cohorts"]):
        raise ValueError("Missing a registered HCC single-cell reference cohort")
    matrices = []
    axis = None
    for cohort in TRACK5:
        prefix = cohort + "/"
        metadata = pd.read_parquet(checked(path, prefix + "metadata.parquet", receipt))
        names = pd.Index(pd.read_csv(checked(path, prefix + "genes.tsv", receipt), sep="\t").symbol)
        counts = load_npz(checked(path, prefix + "counts.npz", receipt))
        if len(metadata) != counts.shape[0] or len(names) != counts.shape[1]:
            raise ValueError(f"Count and metadata dimensions differ for {cohort}")
        good = (metadata.histology.eq("HCC") & metadata.umi.ge(300)
                & metadata.detected_genes.ge(100) & metadata.mitochondrial_fraction.le(0.25))
        if cohort == "GSE149614":
            good &= metadata.site.eq("Tumor")
        use = good.to_numpy()
        selected = counts[use]
        meta = metadata.loc[use].copy()
        meta["cell_type"] = classify(selected, names, meta)
        unique = ~names.duplicated(keep=False)
        axes = names[unique]
        axis = axes if axis is None else axis.intersection(axes)
        matrices.append((selected[:, unique], axes, meta))
    if len(axis) < 5000:
        raise ValueError("HCC reference lacks a robust common expression axis")
    return axis, matrices


def build_reference(axis, matrices):
    reference, cell_sums, support = {}, {}, {}
    by_cluster = {}
    # Count each culture donor once inside each subtype. Equalize donors
    # after transforming each type-specific pseudobulk to counts per million.
    for counts, symbols, labels in matrices:
        for (cohort, donor, kind), group in labels.reset_index(drop=True).groupby(["cohort", "donor", "cell_type"]):
            if len(group) < 12:
                continue
            sample = np.asarray(counts[group.index.to_numpy()].sum(axis=0)).ravel()
            if sample.sum() <= 0:
                continue
            sample = sample[symbols.get_indexer(axis)] * (1e6 / sample.sum())
            by_cluster.setdefault(kind, []).append((cohort, donor, len(group), sample))
    for type_name in TYPES:
        captures = by_cluster.get(type_name, [])
        if len({donor for _, donor, _, _ in captures}) < 4:
            raise ValueError(f"Insufficient independent HCC donors for {type_name}")
        reference[type_name] = np.mean(np.stack([x[3] for x in captures]), axis=0)
        support[type_name] = sorted({donor for _, donor, _, _ in captures})
        cell_sums[type_name] = int(sum(count for _, _, count, _ in captures))
    result = pd.DataFrame(reference, index=axis).astype(np.float32)
    return result, {type_name: {"donors": len(support[type_name]), "cells": cell_sums[type_name]}
                    for type_name in TYPES}


def interaction_edges(path, axis):
    records = pd.read_csv(path, usecols=["source_genesymbol", "target_genesymbol", "resource"], dtype=str)
    use = records[records.resource.str.lower().eq("consensus")].dropna()
    genes = set(axis)
    result = []
    for ligand, receptor, _ in use.itertuples(index=False, name=None):
        left, right = [tuple(re.split(r"_", gene)) for gene in (ligand, receptor)]
        if genes.issuperset(left) and genes.issuperset(right):
            result.append((ligand, receptor))
    result.extend((ligand, receptor) for ligand, receptor in CURATED_EDGES if genes.issuperset((ligand, receptor)))
    edges = pd.DataFrame(result, columns=["ligand", "receptor"]).drop_duplicates().sort_values(["ligand", "receptor"])
    if len(edges) < 100:
        raise ValueError("Too few verified human ligand and receptor complexes")
    return edges


def gene_markers(reference, per_type):
    # Reference-only selectivity. No treatment label or external RNA
    # distribution changes which genes are chosen for deconvolution.
    markers = {}
    for name in TYPES:
        other = reference.drop(columns=name).mean(axis=1)
        o = np.log2((reference[name] + 1) / (other + 1))
        legal = reference[name].ge(15)
        ranked = o[legal].sort_index().sort_values(ascending=False, kind="stable")
        markers[name] = ranked.iloc[:per_type].index.to_list()
    return markers


def prepare(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    axis, cohorts = load_cohorts(a.patients)
    ref, support = build_reference(axis, cohorts)
    special = interaction_edges(a.liana, ref.index)
    markers = gene_markers(ref, a.markers)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        stage = Path(d)
        ref.to_parquet(stage / "donor_equal_reference.parquet")
        special.to_csv(stage / "ligand_receptor.tsv", sep="\t", index=False)
        (stage / "markers.json").write_text(json.dumps(markers, indent=2, sort_keys=True) + "\n")
        (stage / "receipt.json").write_text(json.dumps({
            "inputs": {"patients": record(a.patients / "receipt.json"), "liana": record(a.liana)},
            "cell_type_support": support, "genes": len(ref), "registered_interactions": len(special),
            "structure_supported_edges": {f"{ligand}:{receptor}": source
                                          for (ligand, receptor), source in CURATED_EDGES.items()
                                          if set((ligand, receptor)).issubset(ref.index)},
            "registry": "Donor-equally weighted annotated HCC malignant, immune and stromal pseudobulks",
            "files": {p.name: record(p) for p in sorted(stage.iterdir())},
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("patients", "liana", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--markers", type=int, default=60)
    a = p.parse_args()
    if a.markers < 5:
        p.error("Reference requires independent markers for all cell types")
    prepare(a)


if __name__ == "__main__":
    main()
