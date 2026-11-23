"""Fit-only anchor adaptation models and the shared CPU inference path."""
from dataclasses import dataclass
import re

import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf, OAS

KS = (0, 3, 5, 8)
RIDGES = (.03, .1, .3, 1., 3., 10.)
ALPHAS = (0., .1, .25, .5, .75)
POLICIES = ("logdet", "leverage", "chemistry", "random-971", "random-972", "random-973")
BASELINES = ("B0", "B1", "B2", "B3", "B4")


def positive_semidefinite(matrix, rank=None):
    val, vec = np.linalg.eigh(matrix)
    subset = slice(-rank, None) if rank else slice(None)
    return (vec[:, subset] * np.maximum(val[subset], 0)) @ vec[:, subset].T


def empirical(values):
    """Pairwise observed covariance using each pair's own observed means."""
    valid = np.isfinite(values).astype(float)
    filled = np.nan_to_num(values)
    counts = valid.T @ valid
    sums = filled.T @ valid
    divisor = np.maximum(counts, 1)
    product = filled.T @ filled - sums * sums.T / divisor
    return positive_semidefinite(np.where(counts > 1, product / np.maximum(counts - 1, 1), 0.))


def chemistry(meta):
    from rdkit import Chem, RDLogger
    from vc_build.external import _chemistry

    RDLogger.DisableLog("rdApp.warning")
    bits = _chemistry(meta.reset_index(), fp_size=2048).loc[meta.index].to_numpy()[:, :2048]
    cross = bits @ bits.T
    size = bits.sum(axis=1)
    tanimoto = cross / np.maximum(size[:, None] + size[None, :] - cross, 1)
    targets = [set(re.findall(r"[A-Z][A-Z0-9-]{1,12}", text)) for text in meta.drug_target_source_labels]
    overlap = np.array([[len(a & b) / max(np.sqrt(len(a) * len(b)), 1) for b in targets] for a in targets])
    return np.eye(len(meta)) * .65 + tanimoto * .15 + overlap * .10 + .10, .7 * tanimoto + .3 * overlap


def select_anchors(y, similarity, policy):
    coverage = np.isfinite(y).mean(axis=0)
    scale = np.maximum(np.nanstd(y, axis=0), 1e-8)
    covariance = empirical((y - np.nanmean(y, axis=0)) / scale)
    # Favor repeatably measurable anchors using fit coverage only.
    candidates = np.flatnonzero((coverage >= max(.8, np.sort(coverage)[-8] - .02)) &
                                (np.diag(covariance) > 1e-9))
    if len(candidates) < 8:
        candidates = np.argsort(-coverage, kind="stable")[:min(len(y.T), 16)]
    if len(candidates) < 8:
        raise ValueError("Eight fit-supported anchors are required")
    if policy.startswith("random-"):
        return np.random.default_rng(int(policy.split("-")[-1])).choice(candidates, 8, replace=False)
    if policy == "leverage":
        _, vectors = np.linalg.eigh(covariance)
        score = (vectors[:, -min(8, len(y) - 1):] ** 2).sum(axis=1)
        return candidates[np.argsort(-score[candidates], kind="stable")[:8]]

    ordered = []
    matrix = covariance + np.eye(len(covariance)) * .05
    for _ in range(8):
        remaining = np.setdiff1d(candidates, ordered)
        if policy == "logdet":
            merit = matrix.diagonal()[remaining].copy()
            if ordered:
                cross = matrix[np.ix_(ordered, remaining)]
                merit -= (cross * solve(matrix[np.ix_(ordered, ordered)], cross, assume_a="pos")).sum(axis=0)
        else:
            merit = 1. - (similarity[np.ix_(remaining, ordered)].max(axis=1) if ordered else
                          similarity[np.ix_(remaining, candidates)].mean(axis=1))
        ordered.append(int(remaining[np.argmax(merit)]))
    return np.array(ordered)


def mean_scale(y):
    mean = np.nanmean(y, axis=0)
    sd = np.nanstd(y, axis=0)
    floor = max(float(np.nanmedian(sd)) * .10, 1e-8)
    return mean, np.maximum(sd, floor)


def stand_rna(x, q):
    m, s = x.mean(axis=0), np.maximum(x.std(axis=0), .05)
    return np.clip((x - m) / s, -8, 8), np.clip((q - m) / s, -8, 8)


def rna_kernel(x, q):
    fit, held = stand_rna(x, q)
    norm = max(np.square(fit).sum() / len(fit), 1e-8)
    a = np.square(fit).sum(axis=1) / norm

    def kernel(left, b):
        linear = left @ fit.T / norm
        d = np.maximum(b[:, None] + a[None, :] - 2 * linear, 0)
        return .5 * (linear + np.exp(-d / 2))

    return kernel(fit, a), kernel(held, np.square(held).sum(axis=1) / norm)


def rna_forecast(y, x, q, chem):
    """Existing separable RNA × structure kernel and masked CG solver."""
    import torch
    from vc_build.structured_kernel import solve_residuals

    ck, cq = rna_kernel(x, q)
    mean = np.nanmean(y, axis=0)
    available = np.isfinite(y)
    to_tensor = lambda a: torch.as_tensor(a, dtype=torch.float64)
    coefficient, _, _ = solve_residuals(
        to_tensor(ck), to_tensor(chem), to_tensor(available), to_tensor(np.where(available, y - mean, 0.)), 4.
    )
    return mean + cq @ coefficient.numpy() @ chem, cq, ck


def rna_oof(y, x, q, chem, groups):
    from .inputs import fold_order

    fitted, cross, fit_kernel = rna_forecast(y, x, q, chem)
    oof = np.empty_like(y)
    ordered = np.array(fold_order(groups, "rna-oof"))
    for fold in range(3):
        val = np.isin(groups, ordered[fold::3])
        if not val.any() or val.all():
            raise ValueError("RNA OOF requires three independent held parts")
        oof[val], _, _ = rna_forecast(y[~val], x[~val], x[val], chem)
    return fitted, oof, cross, fit_kernel


def assay_prior(priors, n):
    matrices = []
    for data in priors:
        if len(data) < 5 or data.shape[1] != n:
            continue
        use = np.isfinite(data).sum(axis=0) >= 5
        if use.sum() < 2:
            continue
        matrix = np.eye(n)
        available = data[:, use]
        m, s = mean_scale(available)
        matrix[np.ix_(use, use)] = empirical((available - m) / s)
        diag = np.sqrt(np.maximum(matrix.diagonal(), 1e-8))
        matrices.append(matrix / np.outer(diag, diag))
    return np.mean(matrices, axis=0) if matrices else None


def multirank(values, reference):
    mapped = []
    for j, column in enumerate(values.T):
        source = np.sort(reference[np.isfinite(reference[:, j]), j])
        lower, upper = (np.searchsorted(source, column, side=side) for side in ("left", "right"))
        mapped.append((lower + upper) / (2. * len(source)) - .5)
    return np.column_stack(mapped)


def pair_similarity(first, second, family, width=1.):
    from scipy.spatial.distance import cdist

    if family == "rbf":
        return np.exp(-cdist(first, second, "sqeuclidean") / max(1, first.shape[1]) / width)
    norms = np.maximum(np.linalg.norm(first, axis=1)[:, None] * np.linalg.norm(second, axis=1), 1e-9)
    return first @ second.T / norms


def masked_krr(gram, cross, target, alpha):
    result = np.empty((len(cross), target.shape[1]))
    masks = {}
    for j in range(target.shape[1]):
        mask = np.isfinite(target[:, j])
        masks.setdefault(mask.tobytes(), (mask, []))[1].append(j)
    for mask, columns in masks.values():
        result[:, columns] = 0.
        if mask.any():
            fit = gram[np.ix_(mask, mask)] + alpha * np.eye(mask.sum())
            result[:, columns] = cross[:, mask] @ solve(fit, target[np.ix_(mask, columns)], assume_a="pos")
    return result


def lowrank_als(values, rank, alpha):
    """Observed-entry-only penalized bilinear factorization."""
    _, _, vt = np.linalg.svd(np.nan_to_num(values), full_matrices=False)
    v = vt[:rank].T * np.sqrt(len(vt[:rank]))
    valid = np.isfinite(values)
    fill = np.nan_to_num(values)
    eye = np.eye(v.shape[1]) * alpha

    def update(left, incidence, observed):
        lhs = np.einsum("ij,jk,jl->ikl", incidence, left, left, optimize=True) + eye
        return np.linalg.solve(lhs, (observed @ left)[..., None])[..., 0]

    for _ in range(12):
        u = update(v, valid, fill)
        v = update(u, valid.T, fill.T)
    return v


def covariance(values, method, rank):
    if method == "sample":
        return empirical(values)
    if method == "svd":
        return positive_semidefinite(empirical(values), rank)
    # The estimator's working grid uses fit-centred zero residuals for unobserved entries.
    return {"lw": LedoitWolf, "oas": OAS}[method](assume_centered=False).fit(np.nan_to_num(values)).covariance_


def config(family, **parameters):
    return {"family": family, **parameters}


@dataclass
class Fit:
    y: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    anchors: np.ndarray
    rna_fit: np.ndarray | None = None
    rna_oof: np.ndarray | None = None
    rna_cross: np.ndarray | None = None
    rna_gram: np.ndarray | None = None
    prior_chem: np.ndarray | None = None
    prior_pan: np.ndarray | None = None

    def __post_init__(self):
        self.cache = {}

    def predict(self, obs, k, settings, *, rna_index=None):
        name = settings["family"]
        rna_using = name in {"rna", "rna_cov", "rna_mix"} or settings.get("base") == "rna"
        ix = np.arange(len(obs)) if rna_index is None else rna_index
        base = self.rna_fit[ix].copy() if rna_using else np.broadcast_to(self.mean, (len(obs), len(self.mean))).copy()
        if not k or name in {"mean", "rna"}:
            return base
        a = self.anchors[:k]
        obs = np.asarray(obs)
        if obs.shape != (len(base), k) or not np.isfinite(obs).all():
            raise ValueError("All selected anchors must be measured")
        delta = obs - base[:, a]
        aa = (obs - self.mean[a]) / self.sd[a]
        residual = self.y - self.rna_oof if rna_using else self.y - self.mean
        if name in {"scalar", "affine"}:
            shifts = delta.mean(axis=1, keepdims=True)
            if name == "scalar":
                return base + shifts
            x = base[:, a] - base[:, a].mean(axis=1, keepdims=True)
            penalty = k * np.median(self.sd[a] ** 2)
            scale = np.clip(1 + np.sum(x * delta, axis=1) / (np.sum(x ** 2, axis=1) + penalty), .25, 4.)
            return base + shifts + (scale[:, None] - 1) * (base - base[:, a].mean(axis=1, keepdims=True))
        complete = np.isfinite(self.y[:, a]).all(axis=1)
        train = (self.y[complete][:, a] - self.mean[a]) / self.sd[a]
        if name == "knn":
            from scipy.spatial.distance import cdist

            order = np.argsort(cdist(aa, train, "sqeuclidean"), axis=1, kind="stable")
            nearest = self.y[complete][order[:, :settings["neighbors"]]]
            count = np.isfinite(nearest).sum(axis=1)
            return np.divide(np.nansum(nearest, axis=1), count, out=base, where=count > 0)
        if name in {"rbf", "cosine", "rna_mix"}:
            family = "rbf" if name == "rna_mix" else name
            gram = pair_similarity(train, train, family, settings.get("width", 1.))
            cross = pair_similarity(aa, train, family, settings.get("width", 1.))
            if name == "rna_mix":
                weight = settings["weight"]
                gram = gram * (1 - weight) + self.rna_gram[np.ix_(complete, complete)] * weight
                cross = cross * (1 - weight) + self.rna_cross[ix][:, complete] * weight
            return base + masked_krr(gram, cross, residual[complete], settings["ridge"])
        scale = self.sd if settings.get("scale", True) else np.full(len(self.sd), np.median(self.sd))
        standardized = residual / scale
        if name == "als":
            key = ("als", settings["rank"], settings["ridge"])
            if key not in self.cache:
                self.cache[key] = lowrank_als(standardized, settings["rank"], settings["ridge"])
            v = self.cache[key]
            projection = solve(v[a].T @ v[a] + np.eye(v.shape[1]) * settings["ridge"], v[a].T, assume_a="pos")
            return base + ((delta / scale[a]) @ projection.T @ v.T) * scale
        transform = settings.get("robust", "")
        key = (rna_using, settings.get("estimator", "sample"), settings.get("rank"), bool(settings.get("scale", True)),
               transform, settings.get("clip"), settings.get("prior"), settings.get("alpha"))
        values = standardized
        query = delta / scale[a]
        if transform == "rank":
            values = np.where(np.isfinite(values), multirank(values, values), np.nan)
            query = multirank(query, standardized[:, a])
        elif transform in {"huber", "winsor"}:
            bound = settings["clip"]
            values = np.clip(values, -bound, bound)
            if transform == "huber":
                # Bounded score contribution for any one extreme measured anchor.
                query = np.clip(query, -bound, bound)
            else:
                query = np.clip(query, np.nanquantile(standardized[:, a], .05, axis=0),
                                np.nanquantile(standardized[:, a], .95, axis=0))
        if key not in self.cache:
            k_mat = covariance(values, key[1], key[2])
            prior_name = settings.get("prior")
            if prior_name:
                prior = self.prior_pan if prior_name == "pan" else self.prior_chem
                if prior is None:
                    raise ValueError(f"{prior_name} prior unavailable")
                d = np.sqrt(np.maximum(k_mat.diagonal(), 1e-8))
                alpha = settings["alpha"]
                k_mat = (1 - alpha) * k_mat + alpha * prior * np.outer(d, d)
            self.cache[key] = k_mat
        k_mat = self.cache[key]
        ridge = settings["ridge"] * max(np.median(k_mat.diagonal()), 1e-8)
        weights = solve(k_mat[np.ix_(a, a)] + np.eye(k) * ridge, k_mat[a], assume_a="pos")
        update = query @ weights
        if transform == "rank":
            update *= np.maximum(np.nanstd(standardized, axis=0), 1e-8) / np.maximum(np.nanstd(values, axis=0), 1e-8)
        return base + update * scale
