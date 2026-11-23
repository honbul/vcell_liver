"""Fit-only view geometry and PSD-safe missing-data kernels."""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import qr
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from torch.nn.functional import normalize


def rbf(values, width=1.0):
    squared = values.square().sum(dim=1)
    distance = (squared[:, None] + squared[None, :] - 2 * (values @ values.T)).clamp_min(0)
    return torch.exp(-distance / width)


def normalized_gram(gram):
    diagonal = gram.diagonal()
    if not bool(torch.isfinite(diagonal).all()) or bool((diagonal <= 0).any()):
        raise ValueError("PSD normalization requires finite, positive self-kernels")
    weights = diagonal.rsqrt()
    return weights[:, None] * gram * weights[None, :]


def mask_kernel(gram, available):
    return available[:, None] * gram * available[None, :]


def psd_sum(base, views, weights, *, diagonal_normalize=False):
    if len(views) != len(weights) or any(weight < 0 for weight in weights):
        raise ValueError("Each additional PSD term requires a nonnegative weight")
    total = base + sum(weight * mask_kernel(gram, present) for (gram, present), weight in zip(views, weights))
    return normalized_gram(total) if diagonal_normalize else total


def guarded(base, alternate, present, *, additive=False):
    if additive:
        return torch.where(present[None, :], base + alternate, base)
    return torch.where(present[None, :], alternate, base)


def shuffled(values, present, stop, seed):
    permutation = np.arange(len(present))
    present = present.cpu().numpy()
    for offset, limit, stream in [(0, stop, 0), (stop, len(present), 1)]:
        members = np.flatnonzero(present[offset:limit]) + offset
        rng = np.random.default_rng([seed, stream])
        permutation[members] = rng.permutation(members)
    return values[torch.as_tensor(permutation, device=values.device)]


@dataclass
class View:
    """Explicit feature matrix whose fitted statistics use present fit drugs only."""

    values: torch.Tensor
    present: torch.Tensor
    gram: torch.Tensor
    model_dimension: int


def view_geometry(raw, stop, transform, dimension, kind, device):
    values = raw.astype(np.float32, copy=True)
    present = np.isfinite(values).mean(axis=1) >= .75
    fit_rows = present & (np.arange(len(present)) < stop)
    if fit_rows.sum() < 10:
        raise ValueError("Insufficient observed fitting compounds")
    # Redistribute ties symmetrically and set middle ranks to zero in the signed variant.
    if transform == "rank":
        observed = values[present]
        rank = 2 * (rankdata(observed, axis=1) - .5) / observed.shape[1] - 1
        values[present] = np.where(np.abs(rank) >= .8, rank, 0.)
    admitted = np.isfinite(values[fit_rows]).all(axis=0)
    values = values[:, admitted]
    fit = values[fit_rows]
    location = fit.mean(axis=0)
    spread = fit.std(axis=0)
    keep = spread > 1e-6
    values = np.where(np.isfinite(values), values, location)
    if not keep.any():
        raise ValueError("No varying fitting features")
    if transform == "select":
        ranked = np.argsort(-spread, kind="stable")[:min(768, keep.sum())]
        standardized = (fit[:, ranked] - location[ranked]) / spread[ranked]
        _, tri, pivots = qr(standardized, pivoting=True, mode="economic")
        accepted = np.abs(np.diag(tri)) > .2 * np.sqrt(len(fit))
        selected = ranked[pivots[:len(accepted)][accepted][:dimension]]
        keep = np.isin(np.arange(values.shape[1]), selected)
    known = ((values[present][:, keep] - location[keep]) / spread[keep]).clip(-12, 12)
    if transform in {"pca", "white", "zca"}:
        fitted = ((fit[:, keep] - location[keep]) / spread[keep]).clip(-12, 12)
        count = min(dimension, fitted.shape[1], fitted.shape[0] - 1)
        pca = PCA(n_components=count, svd_solver="randomized", random_state=9171, iterated_power=4)
        pca.fit(fitted)
        known = pca.transform(known)
        if transform != "pca":
            # Whitening and sphering differ by the final rotation. A fixed noise floor
            # avoids singular amplification. Sphering also retains a fixed .10 isotropic
            # residual, so the compared geometries are not rotation-equivalent.
            known = known / np.sqrt(pca.explained_variance_ + .10)
        if transform == "zca":
            native = ((values[present][:, keep] - location[keep]) / spread[keep]).clip(-12, 12)
            native -= pca.mean_
            truncated = (native @ pca.components_.T) @ pca.components_
            known = known @ pca.components_ + .10 * (native - truncated)
    features = torch.zeros((len(values), known.shape[1]), dtype=torch.float32, device=device)
    mask = torch.as_tensor(present, device=device)
    features[mask] = torch.as_tensor(known, device=device)
    features = normalize(features, dim=1)
    return View(features, mask, value_kernel(features, kind), features.shape[1])


def value_kernel(features, kind):
    if kind == "cos":
        return features @ features.T
    return rbf(features, width={"rbf": 1.0, "rbf_wide": 2.0}[kind])


def stress_mask(available, stop, fraction, seed):
    present = available.clone()
    observed = np.flatnonzero(present[stop:].cpu().numpy()) + stop
    random = np.random.default_rng(seed).permutation(observed)
    masked = random[:round(len(random) * fraction)]
    present[torch.as_tensor(masked, device=present.device)] = False
    return present


def concat_kernel(chem, view, present, stop, *, mean_impute):
    imputed = view.clone()
    imputed[~present] = imputed[:stop][present[:stop]].mean(dim=0) if mean_impute else 0.
    features = torch.cat([chem, imputed], dim=1)
    return rbf(features, width=2.0)
