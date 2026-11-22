"""Verified protein and human interaction features with stable target identities."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags

from .perturbseq import record


def verified_vectors(directory):
    path = directory / "vectors.npz"
    receipt = json.loads((directory / "receipt.json").read_text())
    if record(path) != receipt["output"]:
        raise ValueError(f"Unverified embedding {path}")
    with np.load(path, allow_pickle=False) as arrays:
        ids = pd.Index(arrays["id"])
        values = arrays["vectors"].astype(np.float32)
    if not ids.is_unique or len(ids) != len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid embedding axis or values in {path}")
    return pd.DataFrame(values, index=ids)


def unambiguous_symbol_map(annotations):
    genes = pd.read_csv(annotations, usecols=["ensembl_gene_id", "symbol"], dtype=str).dropna().drop_duplicates()
    genes = genes[genes.ensembl_gene_id.str.fullmatch(r"ENSG[0-9]+")]
    # Do not pick an arbitrary symbol or Ensembl ID for ambiguous loci.
    unique = ~genes.ensembl_gene_id.duplicated(keep=False) & ~genes.symbol.duplicated(keep=False)
    return genes[unique].set_index("ensembl_gene_id").symbol


def protein_by_symbol(esm2, table):
    identifiers = pd.read_csv(table, sep="\t", usecols=["id", "gene_symbol"], dtype=str)
    if not identifiers.id.is_unique:
        raise ValueError("Duplicate UniProt identifier")
    identifiers = identifiers.dropna().set_index("id")
    if not esm2.index.isin(identifiers.index).all():
        raise ValueError("ESM-2 includes proteins outside the registered UniProt reference")
    # Some HGNC symbols have several reviewed UniProt accessions. Pool all
    # accessions within the symbol, without using any CRISPRi outcome.
    return esm2.groupby(identifiers.gene_symbol).mean()


def string_neighbors(links, aliases, ids, values, threshold=700):
    proteins = {}
    with gzip.open(aliases, "rt") as source:
        next(source)
        for line in source:
            protein, symbol, kinds = line.rstrip("\n").split("\t")
            if "Ensembl_HGNC_symbol" in kinds.split() and symbol in ids:
                proteins.setdefault(protein, set()).add(symbol)
    protein_to_symbol = {protein: next(iter(symbols)) for protein, symbols in proteins.items()
                         if len(symbols) == 1}
    edges = []
    for frame in pd.read_csv(links, sep=r"\s+", usecols=["protein1", "protein2", "combined_score"],
                             chunksize=400_000):
        frame = frame[frame.combined_score.ge(threshold)].copy()
        frame[["protein1", "protein2"]] = frame[["protein1", "protein2"]].apply(
            lambda column: column.map(protein_to_symbol)
        )
        frame = frame.dropna()
        edges.append(frame[frame.protein1 != frame.protein2])
    pairs = pd.concat(edges).groupby(["protein1", "protein2"], as_index=False).combined_score.max()
    weights = pairs.combined_score.to_numpy(np.float32) / 1000
    matrix = coo_matrix(
        (weights, (ids.get_indexer(pairs.protein1), ids.get_indexer(pairs.protein2))),
        shape=(len(ids), len(ids)),
        dtype=np.float32,
    ).tocsr()
    matrix = matrix.maximum(matrix.T)
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees > 0)
    operator = diags(inverse) @ matrix
    first = operator @ values
    second = operator @ first
    return first, second, matrix.nnz // 2


class GeneFeatures:
    def __init__(self, store, proteins, annotations, links, aliases):
        self.symbols = unambiguous_symbol_map(annotations)
        self.esm = protein_by_symbol(verified_vectors(store / "esm2"), proteins)
        self.graph = verified_vectors(store / "string")
        ids = pd.Index(sorted(set(self.esm.index) | set(self.graph.index)))
        self.esm = self.esm.reindex(ids)
        self.graph = self.graph.reindex(ids)
        base = self.lookup(ids, "combined")
        one_hop, two_hop, edges = string_neighbors(links, aliases, ids, base)
        self.one_hop = pd.DataFrame(one_hop, index=ids)
        self.two_hop = pd.DataFrame(two_hop, index=ids)
        self.receipt = {
            "edges": int(edges), "nodes": len(ids), "edge_threshold": 700,
            "files": {name: record(Path(path)) for name, path in {
                "esm2": store / "esm2" / "receipt.json", "string": store / "string" / "receipt.json",
                "protein_crosswalk": proteins, "hgnc": annotations, "links": links, "aliases": aliases,
            }.items()},
        }

    def lookup(self, symbols, representation):
        tables = {
            "esm": [self.esm], "string": [self.graph], "combined": [self.esm, self.graph],
        }[representation]
        blocks = []
        for table in tables:
            chosen = table.reindex(symbols)
            blocks.append(chosen.fillna(0).to_numpy(np.float32))
            blocks.append(chosen.notna().any(axis=1).to_numpy(np.float32)[:, None])
        return np.column_stack(blocks)

    def encode(self, ensembl, representation):
        symbols = self.symbols.reindex(ensembl.to_numpy()).fillna("").to_numpy()
        if representation == "graph":
            return np.column_stack([
                self.lookup(symbols, "combined"),
                self.one_hop.reindex(symbols).fillna(0).to_numpy(np.float32),
                self.two_hop.reindex(symbols).fillna(0).to_numpy(np.float32),
            ])
        return self.lookup(symbols, representation)
