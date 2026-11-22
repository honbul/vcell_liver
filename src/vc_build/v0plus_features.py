"""Train-only RNA representations and public biological priors for V0+."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def hallmarks(path):
    records = []
    for line in path.read_text().splitlines():
        pathway, _, *genes = line.split("\t")
        records.extend((pathway, gene, 1.) for gene in genes)
    return pd.DataFrame(records, columns=["signature", "gene", "weight"])


def signatures(hallmark, progeny, dorothea):
    paths = pd.read_csv(progeny, index_col=0).rename_axis("gene").reset_index()
    paths = paths.melt(id_vars="gene", var_name="signature", value_name="weight")
    paths = paths[paths.weight.ne(0)].assign(signature=lambda d: "PROGENY_" + d.signature)
    regulons = pd.read_csv(dorothea)
    regulons = regulons[regulons.confidence.isin(["A", "B", "C"])]
    regulons = regulons.assign(signature="DOROTHEA_" + regulons.tf,
                               gene=regulons.target, weight=regulons.mor * regulons.likelihood)
    edges = pd.concat([hallmarks(hallmark), paths, regulons[["signature", "gene", "weight"]]])
    return edges.groupby(["signature", "gene"], as_index=False).weight.mean()


def rna_values(rna, rank):
    values = rna.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("RNA includes nonfinite values")
    if rank:
        return ((rankdata(values, axis=1) - 0.5) / values.shape[1] - 0.5).astype("float32")
    return values


@dataclass
class ExpressionFeatures:
    genes: list
    columns: list
    indices: np.ndarray
    network: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    rank: bool

    @classmethod
    def fit(cls, frame, edges, targets, *, n_variable, rank):
        genes = sorted(frame.columns)
        if len(genes) < 50:
            raise ValueError("At least 50 comparable RNA genes are required")
        values = rna_values(frame[genes], rank)
        dispersion = values.std(axis=0)
        variable = np.argsort(-dispersion, kind="stable")[:n_variable]
        matched_targets = sorted(set().union(*targets.values()) & set(genes)) if targets else []
        positions = pd.Index(genes).get_indexer(matched_targets)
        indices = np.concatenate([variable, positions[~np.isin(positions, variable)]])
        edges = edges[edges.gene.isin(genes)]
        network = (edges.pivot(index="gene", columns="signature", values="weight")
                   .reindex(genes).fillna(0))
        network = network.loc[:, network.ne(0).sum().ge(10)]
        weights = network.to_numpy(dtype=np.float32)
        weights /= np.sqrt(np.square(weights).sum(axis=0))
        projected = np.column_stack((values[:, indices], values @ weights))
        columns = [genes[i] for i in indices] + network.columns.tolist()
        return cls(genes, columns, indices, weights, projected.mean(axis=0),
                   np.maximum(projected.std(axis=0), 1e-5), rank)

    def transform(self, frame):
        missing = set(self.genes) - set(frame.columns)
        if missing:
            raise ValueError(f"Profile lacks {len(missing)} genes from the registered measurement axis")
        values = rna_values(frame[self.genes], self.rank)
        projected = np.column_stack((values[:, self.indices], values @ self.network))
        standardized = np.clip((projected - self.center) / self.scale, -8., 8.)
        return pd.DataFrame(standardized, index=frame.index, columns=self.columns)


def validated_public_prior(path, *, kind, expected, digest):
    from .io import sha256

    actual = sha256(path)
    if actual != digest:
        raise ValueError(f"Unexpected {kind} data hash: {actual}")
    return {"source": expected, "sha256": actual}
