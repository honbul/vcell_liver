"""Fold-specific expression baselines and frozen representation alignment."""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

from .artifacts import verify_record


def read_vectors(directory):
    receipt = json.loads((directory / "receipt.json").read_text())
    path = directory / "vectors.npz"
    verify_record(path, receipt["output"])
    with np.load(path, allow_pickle=False) as data:
        ids, vectors = data["id"], data["vectors"].astype(np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid frozen embedding {directory}")
    frame = pd.DataFrame(vectors, index=ids)
    if not frame.index.is_unique:
        raise ValueError(f"Duplicate embedding IDs {directory}")
    return frame


def signed_pathways(standardized_ranks, genes, weights, hallmark):
    table = pd.read_csv(weights, sep="\t")
    sets = []
    with hallmark.open() as stream:
        for line in stream:
            label, _, *members = line.rstrip("\n").split("\t")
            sets.extend((gene, label, 1) for gene in members)
    table = pd.concat([table, pd.DataFrame(sets, columns=["source", "target", "weight"])])
    table = table.loc[table.source.isin(genes)].copy()
    use = table.groupby("target").source.transform("nunique").ge(10)
    overlap = table.loc[use].pivot_table(index="source", columns="target", values="weight",
                                       aggfunc="mean", fill_value=0)
    matrix = overlap.reindex(genes, fill_value=0).to_numpy(np.float32)
    denominator = np.abs(matrix).sum(axis=0)
    return standardized_ranks @ (matrix / np.maximum(denominator, 1e-6))


def rank_block(profiles, cells, mode, path_weights=None, hallmark=None):
    # Per-profile ranking uses no cohort statistics. HVG selection and any
    # centering or scaling are learned anew inside the current fitting split.
    genes = profiles.columns.to_numpy()
    data = profiles.to_numpy(dtype=np.float32)
    ranks = rankdata(data, method="average", axis=1).astype(np.float32) / len(genes)
    in_fit = profiles.index.get_indexer(sorted(set(cells)))
    if (in_fit < 0).any():
        raise ValueError("Missing training RNA")
    if mode == "paths":
        scaled = StandardScaler().fit(ranks[in_fit])
        values = signed_pathways(scaled.transform(ranks), genes, path_weights, hallmark)
    else:
        variance = data[in_fit].var(axis=0)
        columns = np.lexsort((genes, -variance))[:512]
        values = ranks[:, columns]
    return pd.DataFrame(values, index=profiles.index)
