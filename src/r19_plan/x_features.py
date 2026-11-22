"""Matched-width cell ID and independent basal RNA input for held-context experiments."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.r19_plan.dev_selection import checked_array

CONTEXT_WIDTH = 16


def checked_contexts(values):
    labels = pd.Index(values)
    if labels.empty or labels.isna().any() or any(not isinstance(x, str) or not x.strip() for x in labels):
        raise ValueError("X requires an explicit canonical cell context for every row")
    return labels


@dataclass(frozen=True)
class CellIdentity:
    contexts: tuple[str, ...]

    def transform(self, contexts):
        requested = checked_contexts(contexts)
        index = pd.Index(self.contexts).get_indexer(requested)
        features = np.zeros((len(index), CONTEXT_WIDTH), dtype=np.float32)
        known = index >= 0
        features[np.flatnonzero(known), index[known]] = 1
        # An unseen cell has the same uniform average representation as the equally weighted fit contexts.
        features[~known, :len(self.contexts)] = 1. / len(self.contexts)
        return features


def fit_cell_identity(fitting):
    labels = checked_contexts(fitting).unique().sort_values()
    if len(labels) > CONTEXT_WIDTH or len(labels) < 2:
        raise ValueError("X1 requires at least two training contexts within the matched feature width")
    return CellIdentity(tuple(labels))


@dataclass(frozen=True)
class BasalContext:
    genes: tuple[int, ...]
    columns: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    basis: PCA
    component_scale: np.ndarray

    def transform(self, feature_genes, values):
        values = np.asarray(values)
        if tuple(feature_genes) != self.genes:
            raise ValueError("An X2 query uses a different basal gene order")
        array = checked_array(values, (len(values), len(self.genes)), "independent basal RNA")
        standardized = (array[:, self.columns] - self.mean) / self.scale
        mapped = self.basis.transform(standardized) / self.component_scale
        padded = np.zeros((len(array), CONTEXT_WIDTH), dtype=np.float32)
        padded[:, :self.basis.n_components_] = mapped
        return padded


def fit_basal_context(fitting, feature_contexts, feature_genes, values):
    """Fit one profile per authorized training context and keep the response axis out of the transform."""
    contexts = checked_contexts(feature_contexts)
    genes = tuple(feature_genes)
    if not contexts.is_unique or not genes or len(set(genes)) != len(genes):
        raise ValueError("X2 needs a unique profile and gene coordinate per cell context")
    profiles = checked_array(values, (len(contexts), len(genes)), "registered X2 basal profiles")
    fit_contexts = checked_contexts(fitting).unique().sort_values()
    source = contexts.get_indexer(fit_contexts)
    if (source < 0).any() or len(source) < 2:
        raise ValueError("X2 fitting lacks independent registered basal profiles")
    fit = profiles[source].astype(np.float64)
    spread = fit.std(axis=0)
    keep = spread > 1e-10
    if not keep.any():
        raise ValueError("X2 training profiles have no between-context variation")
    center, scale = fit[:, keep].mean(axis=0), spread[keep]
    standardized = (fit[:, keep] - center) / scale
    rank = min(CONTEXT_WIDTH, int(keep.sum()), np.linalg.matrix_rank(standardized), len(source) - 1)
    if rank == 0:
        raise ValueError("X2 basal context coordinates have zero supported rank")
    basis = PCA(n_components=rank, svd_solver="full").fit(standardized)
    component_scale = basis.transform(standardized).std(axis=0)
    return BasalContext(genes, keep, center, scale, basis, component_scale)
