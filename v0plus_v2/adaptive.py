"""Train-only expression views and adaptive per-drug linear and kernel models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet

NORMS = ("rank", "robust_z", "rank_z")
RIDGES = (1., 10., 100., 300., 1000., 3000., 10000., 30000.)
WIDTHS = (256, 512, 1024, 2048)
FAMILY = ("ridge", "enet", "pls", "rbf", "cosine")


def sample_rank(values):
    return ((rankdata(values, axis=1) - .5) / values.shape[1] - .5).astype(np.float32)


@dataclass
class Projection:
    genes: list
    norm: str
    center: np.ndarray
    scale: np.ndarray
    hvg: np.ndarray

    @classmethod
    def fit(cls, rna, norm):
        values = rna.to_numpy(dtype=np.float32)
        if norm != "robust_z":
            values = sample_rank(values)
        sort = np.argsort(-values.std(axis=0), kind="stable")
        if norm == "rank":
            center, scale = np.zeros(values.shape[1]), np.full(values.shape[1], .2)
        else:
            low, center, high = np.quantile(values, [.1, .5, .9], axis=0)
            scale = np.maximum((high - low) / 2.56, .03 if norm == "rank_z" else .10)
        return cls(list(rna.columns), norm, center.astype(np.float32), scale.astype(np.float32), sort)

    def transform(self, rna):
        if not rna.columns.equals(pd.Index(self.genes)):
            raise ValueError("Expected the frozen RNA gene axis")
        values = rna.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Expression values must be finite")
        if self.norm != "robust_z":
            values = sample_rank(values)
        return np.clip((values - self.center) / self.scale, -8, 8)


def selectors(x, y, hvg, known, strategy, width):
    if strategy == "hvg":
        return hvg[:width]
    yc = y - y.mean()
    centered = x - x.mean(axis=0)
    variance = np.maximum(np.linalg.norm(centered, axis=0), 1e-5)
    strength = np.abs(centered.T @ yc) / variance
    strength = .8 * strength + .2 * strength.max() * np.isin(np.arange(x.shape[1]), hvg[:2048])
    order = np.argsort(-strength, kind="stable")
    take = width if strategy == "target" else width // 2
    prefix = known if strategy == "target" else hvg[:take]
    return np.asarray(list(dict.fromkeys([*prefix, *order[:width * 2]]))[:width])


def kernel(method, x, y):
    if method == "ridge":
        return x @ y.T
    if method == "rbf":
        return np.exp(-cdist(x, y, "sqeuclidean") / (2 * x.shape[1]))
    a = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-5)
    b = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-5)
    return a @ b.T


@dataclass
class KernelHead:
    kind: str
    reference: np.ndarray
    coefficients: np.ndarray
    mean: float

    def predict(self, x):
        return self.mean + kernel(self.kind, x, self.reference) @ self.coefficients


def kernel_path(kind, fit, y, penalties):
    x = fit - fit.mean(axis=0)
    gram = kernel(kind, x, x)
    values, vectors = eigh(gram, check_finite=False, driver="evd")
    projections = vectors.T @ (y - y.mean())
    values = np.maximum(values, 0.)
    for a in penalties:
        coef = vectors @ (projections / (values + a))
        if kind == "ridge":
            yield a, float(y.mean()), x.T @ coef, None
        else:
            yield a, float(y.mean()), coef, x


@dataclass
class DrugHead:
    norm: str
    columns: np.ndarray
    center: np.ndarray
    fit: object
    mean: float | None = None

    def apply(self, x):
        active = x[:, self.columns] - self.center
        if self.mean is not None:
            return self.mean + active @ self.fit
        return self.fit.predict(active).ravel()


def candidate_grid():
    for norm in NORMS:
        for width in WIDTHS:
            strategies = ("hvg", "hybrid", "target") if width in (256, 512) else ("hvg",)
            for strategy in strategies:
                yield norm, width, strategy, "ridge", RIDGES
        yield norm, 256, "target", "enet", (.001, .004, .015)
        yield norm, 256, "hybrid", "pls", (3, 8, 16)
        yield norm, 512, "hybrid", "rbf", (.05, .3, 2., 12.)
        yield norm, 512, "hybrid", "cosine", (.05, .3, 2., 12.)


def fit_candidates(views, registry, before, after, y, known, *, configs=None):
    """One decomposition computes the entire ridge path within each feature view."""
    grid = candidate_grid() if configs is None else (
        (norm, width, strategy, family, (penalty,)) for norm, width, strategy, family, penalty in configs
    )
    for norm, width, strategy, family, penalties in grid:
        m = views[norm]
        x, q = m[before], m[after]
        columns = selectors(x, y, registry[norm].hvg, known, strategy, width)
        fit = x[:, columns]
        center = fit.mean(axis=0)
        if family in ("ridge", "rbf", "cosine"):
            for p, mean, coef, ref in kernel_path(family, fit, y, penalties):
                estimator = coef if ref is None else KernelHead(family, ref, coef, mean)
                head = DrugHead(norm, columns, center, estimator, mean if ref is None else None)
                yield (norm, width, strategy, family, p), head, head.apply(q) if len(q) else np.empty(0)
        else:
            for p in penalties:
                if p >= len(fit) and family == "pls":
                    continue
                estimator = (ElasticNet(alpha=p, l1_ratio=.35, max_iter=1250, tol=.0005)
                             if family == "enet" else PLSRegression(n_components=p, scale=False))
                head = DrugHead(norm, columns, center, estimator.fit(fit - center, y))
                yield (norm, width, strategy, family, p), head, head.apply(q) if len(q) else np.empty(0)
