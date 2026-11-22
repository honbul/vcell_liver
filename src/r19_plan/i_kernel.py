"""Fit-only feature maps and a PSD masked kernel for independent I-track views."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class FitProfile:
    """Training-only column policy, standardization and optional low-rank basis."""

    columns: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    basis: PCA | None

    @classmethod
    def learn(cls, training, rank=None):
        x = np.asarray(training, dtype=np.float64)
        if x.ndim != 2 or not len(x):
            raise ValueError("Profile fit requires a nonempty training matrix")
        good = np.isfinite(x)
        count = good.sum(axis=0)
        values = np.where(good, x, 0.0)
        mean = values.sum(axis=0) / np.maximum(count, 1)
        squared = np.where(good, (values - mean) ** 2, 0.0)
        scale = np.sqrt(squared.sum(axis=0) / np.maximum(count, 1))
        columns = (count >= 2) & (scale > 1e-12)
        profile = cls(columns, mean[columns], scale[columns], None)
        normalized, seen = profile.standardized(x)
        # Fit all preprocessing on measured training parents. Missing rows cannot
        # shift the training basis, and masked entries use the training mean.
        effective = min(rank or 0, int(seen.sum()) - 1, int(columns.sum()))
        if effective > 0:
            basis = PCA(n_components=effective, svd_solver="randomized", random_state=1501)
            basis.fit(normalized[seen])
            profile = cls(columns, profile.mean, profile.scale, basis)
        return profile

    def standardized(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != len(self.columns):
            raise ValueError("Source profile feature axis changed")
        selected = data[:, self.columns]
        seen = np.isfinite(selected)
        return np.where(seen, (selected - self.mean) / self.scale, 0.0), seen.any(axis=1)

    def transform(self, data):
        z, seen = self.standardized(data)
        if self.basis is not None:
            z = self.basis.transform(z)
        z[~seen] = 0.0
        norm = np.linalg.norm(z, axis=1)
        nonzero = norm > 1e-12
        z = z / np.where(nonzero, norm, 1.0)[:, None]
        return z, seen


def masked_gram(structure, views):
    """Sum congruences of PSD component matrices after per-row mask normalization.

    Structure is always present. Each optional block is a pair (features, mask)
    with unit view weight. No row-pair denominator is used.
    """
    structure = np.asarray(structure, dtype=np.float64)
    if structure.ndim != 2 or structure.shape[0] != structure.shape[1]:
        raise ValueError("Structure Gram matrix must be square")
    if not np.isfinite(structure).all() or not np.allclose(structure, structure.T):
        raise ValueError("Structure Gram matrix is not finite and symmetric")
    n = len(structure)
    mass = np.ones(n, dtype=np.float64)
    result = structure.copy()
    for vectors, present in views:
        vectors = np.asarray(vectors, dtype=np.float64)
        present = np.asarray(present, dtype=bool)
        if (vectors.ndim != 2 or vectors.shape[0] != n or present.shape != (n,)
                or not np.isfinite(vectors).all()):
            raise ValueError("Missing or invalid optional view axis")
        block = np.where(present[:, None], vectors, 0.0)
        mass += present
        result += block @ block.T
    scale = np.sqrt(mass)
    return result / (scale[:, None] * scale[None, :])
