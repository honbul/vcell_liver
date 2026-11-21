"""Same-parent B donor rules for a separate X3 DEV transfer comparison."""

from dataclasses import dataclass

import numpy as np

from src.r19_plan.dev_selection import checked_array

TEMPERATURES = (.5, 1., 2.)
RIDGES = (.1, 1.)


def donor_mean(responses, cells, target, means=None, temperature=None):
    """Average unique donor contexts uniformly or by independent basal distance."""
    values = np.asarray(responses)
    if (values.ndim != 3 or values.shape[1] != len(cells) or len(set(cells)) != len(cells)
            or target in cells or len(cells) < 2 or not np.isfinite(values).all()):
        raise ValueError("X3 needs finite matched, non-target donor contexts in each query parent")
    if temperature is None:
        return values.mean(axis=1)
    if temperature not in TEMPERATURES or means is None or not {target, *cells} <= set(means):
        raise ValueError("The X3 similarity rule changed its registered scale or context axis")
    difference = np.stack([means[cell] - means[target] for cell in cells]).astype(np.float64)
    scores = -np.square(difference).sum(axis=1) / (2. * temperature ** 2)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite independent basal distance in X3")
    weights = np.exp(scores - scores.max())
    return np.einsum("pcg,c->pg", values, weights / weights.sum())


@dataclass(frozen=True)
class BasalResponseCorrection:
    fit_contexts: tuple[str, ...]
    scale: np.ndarray
    coefficients: np.ndarray
    width: int

    def predict(self, average, query, donor_contexts, means):
        adjusted = checked_array(average, (len(average), len(self.scale)), "X3 equal donor mean")
        if (len(set(donor_contexts)) != len(donor_contexts) or len(donor_contexts) < 2 or query in donor_contexts
                or query in self.fit_contexts or not set(donor_contexts) <= set(self.fit_contexts)):
            raise ValueError("X3 basal operator cannot address treatment from a held context")
        x = np.asarray(means[query])[:self.width] - np.mean([means[cell][:self.width] for cell in donor_contexts], axis=0)
        additive = x @ self.coefficients[:, :self.width].T
        multiplier = x @ self.coefficients[:, self.width:].T
        return adjusted + additive + (adjusted / self.scale) * multiplier


def fit_basal_response(responses, parents, cells, coordinates, *, ridge, held_parents, held_context):
    """Fit a gene-wise additive/multiplicative basal correction using only source parents and contexts."""
    if (ridge not in RIDGES or held_context in cells or len(set(cells)) != len(cells) or len(cells) < 3
            or len(set(parents)) != len(parents) or set(parents) & set(held_parents)):
        raise ValueError("X3 basal correction fit includes protected parents/contexts or unsupported settings")
    value = np.asarray(responses, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    if (value.ndim != 3 or value.shape[:2] != (len(parents), len(cells)) or len(parents) < 2
            or coords.ndim != 2 or coords.shape[0] != len(cells) or not np.isfinite(value).all()
            or not np.isfinite(coords).all()):
        raise ValueError("X3 source pairs require the same complete parent, context and basal axes")
    keep = np.flatnonzero(coords.std(axis=0) > 1e-10)
    if not len(keep) or not np.array_equal(keep, np.arange(len(keep))):
        raise ValueError("The X3 basal basis must be nonzero and left aligned")
    coords = coords[:, keep]
    other = (value.sum(axis=1, keepdims=True) - value) / (len(cells) - 1)
    displacement = (coords * len(cells) - coords.sum(axis=0)) / (len(cells) - 1)
    base = np.broadcast_to(displacement[:, :, None], (len(cells), len(keep), value.shape[2]))
    scale = np.maximum(np.sqrt(np.square(other).mean(axis=(0, 1))), .05)
    data = np.broadcast_to(base[None, :, :, :], (len(parents), *base.shape))
    design = np.concatenate((data, data * (other / scale)[:, :, None, :]), axis=2)
    design = design.reshape(-1, design.shape[2], design.shape[3])
    residual = (value - other).reshape(-1, value.shape[2])
    n = len(design)
    gram = np.einsum("nig,njg->gij", design, design) / n + ridge * np.eye(design.shape[1])[None, :, :]
    rhs = np.einsum("nig,ng->gi", design, residual) / n
    coefficients = np.linalg.solve(gram, rhs[:, :, None]).squeeze(-1)
    return BasalResponseCorrection(tuple(cells), scale, coefficients, len(keep))
