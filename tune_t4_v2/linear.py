"""Full linear regularization path and RNA/compound conditional gain for T4."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import svd
from sklearn.linear_model import Ridge

from .data import parent_weights

RANKS = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 0)
PENALTIES = (.003, .03, .3, 1., 3., 10., 30., 100., 300., 1000.)
NUCLEAR = (.0, .2, .45, .7, .88)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)


@lru_cache(maxsize=45000)
def chemistry(parent):
    molecule = Chem.MolFromSmiles(parent)
    if molecule is None:
        raise ValueError(f"Unresolved registered chemical parent: {parent}")
    return MORGAN.GetFingerprintAsNumPy(molecule).astype(np.float32)


def conditions(rows, x, use_chemistry):
    source_rms = np.linalg.norm(x, axis=1) / np.sqrt(x.shape[1])
    c = np.column_stack([np.log(np.maximum(rows.dose_uM.to_numpy(float), 1e-6)),
                         np.log(np.maximum(source_rms, 1e-4))])
    c = np.column_stack([c, c**2, c[:, 0] * c[:, 1]])
    if use_chemistry:
        c = np.c_[c, .4 * np.stack([chemistry(p) for p in rows.parent])]
    return c


@dataclass
class Operator:
    x_mean: np.ndarray
    y_mean: np.ndarray
    weights: np.ndarray
    gate: object = None
    chemistry: bool = False

    def predict(self, x, rows):
        values = (x - self.x_mean) @ self.weights
        if self.gate is not None:
            gain = np.clip(self.gate.predict(conditions(rows, x, self.chemistry)), .15, 2.5)
            values *= gain[:, None]
        return self.y_mean + values


def weighted_svd(batch, whiten):
    weights = parent_weights(batch.rows)
    x_mean = np.average(batch.x, axis=0, weights=weights)
    y_mean = np.average(batch.y, axis=0, weights=weights)
    x = batch.x - x_mean
    y = batch.y - y_mean
    gene_scale = np.ones(x.shape[1])
    if whiten:
        dispersion = np.average(np.square(x), axis=0, weights=weights)
        tau = np.median(dispersion)
        gene_scale = np.sqrt(.5 + .5 * dispersion / max(tau, 1e-6))
    normalized = x / gene_scale
    scaled = normalized * np.sqrt(weights[:, None])
    u, s, vh = svd(scaled, full_matrices=False, check_finite=False)
    keep = s > max(s[0] * 1e-5, 1e-6)
    return x_mean, y_mean, gene_scale, u[:, keep], s[keep], vh[keep], y * np.sqrt(weights[:, None])


def point_fits(settings, batch, query_x, query_rows, *, need_model=False):
    """Evaluate the path in the small fitted sample subspace."""
    for whiten in sorted({s[0] for s in settings}):
        xm, ym, gs, u, s, v, y = weighted_svd(batch, whiten)
        q = ((query_x - xm) / gs) @ v.T
        uy = u.T @ y
        candidates = [k for k in settings if k[0] == whiten]
        for penalty in sorted({k[2] for k in candidates}):
            base = (s / (s * s + penalty))[:, None] * uy
            group = [k for k in candidates if k[2] == penalty]
            rrr = None
            if any(k[1] in ("reduced", "nuclear") for k in group):
                p = (s[:, None] * uy) / np.sqrt((s * s + penalty)[:, None])
                rrr = svd(p, full_matrices=False, check_finite=False)
            for config in group:
                _, family, _, capacity, gate = config
                if family == "pcr":
                    take = len(s) if capacity == 0 else min(capacity, len(s))
                    coefficients, query = base[:take], q[:, :take]
                    gene_basis = v[:take]
                else:
                    left, singular, right = rrr
                    if family == "reduced":
                        take = min(capacity, len(s))
                        adjusted = singular[:take]
                    else:
                        take = len(s)
                        adjusted = np.maximum(singular - capacity * singular[0], 0)
                    coefficients = ((left[:, :take] * adjusted) @ right[:take]) / np.sqrt(
                        (s * s + penalty)[:, None])
                    query, gene_basis = q, v
                predictions = ym + query @ coefficients
                fitted = None
                if need_model or gate:
                    matrix = (gene_basis.T @ coefficients) / gs[:, None]
                    fitted = Operator(xm, ym, matrix)
                if gate:
                    fitted.chemistry = gate == "chem"
                    # Cross-fitted residual direction keeps the gain from learning fitted noise.
                    fitted.gate = fit_gate(batch, config)
                    predictions = fitted.predict(query_x, query_rows) if len(query_x) else predictions
                yield config, predictions, fitted


def fit_gate(batch, config):
    from sklearn.model_selection import GroupKFold

    rows, x, y = batch.rows, batch.x, batch.y
    anchors = np.empty_like(y)
    residual = np.empty_like(y)
    bare = (*config[:-1], "")
    for fit, held in GroupKFold(3).split(x, groups=rows.scaffold):
        mask = np.arange(len(rows))
        tr = batch.select(np.isin(mask, fit))
        _, p, op = next(point_fits([bare], tr, x[held], rows.iloc[held], need_model=True))
        anchors[held] = op.y_mean
        residual[held] = p - op.y_mean
    strength = np.square(residual).sum(axis=1)
    gain = np.sum(residual * (y - anchors), axis=1) / np.maximum(strength, .01)
    weight = parent_weights(rows) * strength / np.maximum(np.median(strength), .01)
    return Ridge(alpha=30.).fit(conditions(rows, x, config[-1] == "chem"), np.clip(gain, 0, 3),
                                sample_weight=weight)


def base_grid():
    for whiten in (False, True):
        for alpha in PENALTIES:
            for rank in RANKS:
                yield (whiten, "pcr", alpha, rank, "")
        for alpha in (.3, 3., 30., 300.):
            for rank in (4, 8, 16, 32, 64):
                yield (whiten, "reduced", alpha, rank, "")
            for strength in NUCLEAR:
                yield (whiten, "nuclear", alpha, strength, "")


def loss(batch, prediction):
    per_well = np.mean(np.square(prediction - batch.y), axis=1)
    w = parent_weights(batch.rows)
    return float(np.average(per_well, weights=w))
