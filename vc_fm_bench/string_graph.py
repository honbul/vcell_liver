"""Freeze human STRING association embeddings using only the public graph."""
import argparse
import gzip
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from sklearn.utils.extmath import randomized_svd

from .artifacts import file_record, new_directory, write_json


def human_gene_aliases(path):
    aliases = {}
    with gzip.open(path, "rt") as stream:
        next(stream)
        for line in stream:
            protein, alias, source = line.rstrip("\n").split("\t")
            if "Ensembl_HGNC_symbol" in source.split():
                aliases.setdefault(protein, set()).add(alias)
    return {key: sorted(values)[0] for key, values in aliases.items() if len(values) == 1}


def adjacency(path, aliases, min_score):
    blocks = []
    for table in pd.read_csv(path, sep=" ", usecols=["protein1", "protein2", "combined_score"],
                             chunksize=500_000):
        table = table.loc[table.combined_score.ge(min_score) & table.protein1.isin(aliases)
                          & table.protein2.isin(aliases)].copy()
        for column in ("protein1", "protein2"):
            table[column] = table[column].map(aliases)
        table = table.loc[table.protein1 != table.protein2]
        blocks.append(table)
    edges = pd.concat(blocks)
    edges = edges.groupby(["protein1", "protein2"], as_index=False).combined_score.max()
    ids = pd.Index(sorted(set(edges.protein1) | set(edges.protein2)))
    source = ids.get_indexer(edges.protein1)
    target = ids.get_indexer(edges.protein2)
    weights = edges.combined_score.to_numpy(np.float32) / 1000
    sparse = coo_matrix((weights, (source, target)), shape=(len(ids), len(ids))).tocsr()
    sparse = sparse.maximum(sparse.T)
    degree = np.asarray(sparse.sum(axis=1)).ravel()
    scale = diags(1 / np.sqrt(degree))
    return ids, scale @ sparse @ scale, sparse.nnz // 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--links", type=Path, required=True)
    p.add_argument("--aliases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-score", type=int, default=700)
    p.add_argument("--dimension", type=int, default=96)
    a = p.parse_args()
    start = perf_counter()
    aliases = human_gene_aliases(a.aliases)
    ids, matrix, edges = adjacency(a.links, aliases, a.min_score)
    if a.dimension >= len(ids) or a.dimension < 1:
        raise ValueError("Embedding dimension must be smaller than the number of graph nodes")
    vectors, values, _ = randomized_svd(matrix, n_components=a.dimension, n_iter=7, random_state=42)
    vectors = (vectors * np.sqrt(values)).astype(np.float32)
    with new_directory(a.output) as out:
        path = out / "vectors.npz"
        np.savez(path, id=ids.to_numpy(dtype=str), vectors=vectors)
        write_json(out / "receipt.json", {
            "model": "STRING-human-v12-symmetric-normalized-adjacency-SVD",
            "device": "cpu", "input": {"links": file_record(a.links), "aliases": file_record(a.aliases)},
            "output": file_record(path), "dimension": a.dimension,
            "genes": len(ids), "undirected_edges": edges, "min_score": a.min_score,
            "wall_seconds": perf_counter() - start,
        })


if __name__ == "__main__":
    main()
