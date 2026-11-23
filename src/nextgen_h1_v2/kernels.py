"""Family A and the cross-fit first stage for Family D."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh, solve
from scipy.spatial.distance import cdist

from nextgen_h1_anchor.models import masked_krr


def base(ctx, dual, training=False):
    if dual:
        return ctx.bfit if training else ctx.bvalid
    n = len(ctx.train) if training else len(ctx.valid_groups)
    return np.broadcast_to(ctx.mean, (n, len(ctx.mean)))


def residual_packet(ctx, packet, order, prior, clip=None):
    residual = (packet - prior[:, order]) / ctx.sd[order]
    if clip is not None:
        residual = np.clip(residual, -clip, clip)
    return np.nan_to_num(residual)


def eb_predict(ctx, packet, order, dual, config, prior=None):
    prior = base(ctx, dual) if prior is None else prior
    if not len(order):
        return prior.copy()
    local = ctx.cov_rna if dual else ctx.cov
    alpha = config.get("alpha", 0.)
    source = ctx.prior_pan if config.get("prior") == "pan" else ctx.chem
    covariance = (1 - alpha) * local + alpha * source
    weights = solve(covariance[np.ix_(order, order)] + config["ridge"] * np.eye(len(order)),
                    covariance[order], assume_a="pos")
    return prior + (residual_packet(ctx, packet, order, prior, config.get("clip")) @ weights) * ctx.sd



def product_ridge(gram, cross, structure, residual, ridge, rank=16):
    """Separable Nyström ridge on measured assay pairs."""
    def factors(matrix):
        start = max(0, len(matrix) - rank)
        val, vectors = eigh(matrix, subset_by_index=(start, len(matrix) - 1))
        keep = val > 1e-8
        return vectors[:, keep], np.sqrt(val[keep])

    uc, sc = factors(gram)
    ud, sd = factors(structure)
    c, d = uc * sc, ud * sd
    i, j = np.where(np.isfinite(residual))
    design = (c[i, :, None] * d[j, None, :]).reshape(len(i), -1)
    lhs = design.T @ design
    lhs.flat[::len(lhs) + 1] += ridge
    coefficients = solve(lhs, design.T @ residual[i, j], assume_a="pos").reshape(c.shape[1], d.shape[1])
    return ((cross @ (uc / sc)) @ coefficients) @ d.T


@dataclass
class KernelModel:
    ctx: object
    order: np.ndarray
    config: dict
    dual: bool

    def predict(self, packet, k, *, scramble=None, training=False):
        cfg = self.config
        ctx, dual, a = self.ctx, self.dual, self.order[:k]
        prior = base(ctx, dual, training)
        if scramble is not None and dual:
            prior = prior[scramble]
        if cfg["method"] == "eb" or not k and not dual:
            return eb_predict(ctx, packet, a, dual, cfg, prior)
        valid = np.isfinite(ctx.train[:, a]).all(1)
        train_base = base(ctx, dual, True)
        x = residual_packet(ctx, ctx.train[:, a], a, train_base, cfg.get("clip"))[valid]
        z = residual_packet(ctx, packet, a, prior, cfg.get("clip"))
        width = cfg["width"] * max(1, k)
        gram = np.exp(-cdist(x, x, "sqeuclidean") / width)
        cross = np.exp(-cdist(z, x, "sqeuclidean") / width)
        rw = cfg["rna_weight"] if dual else 0.
        if rw:
            rna_q = ctx.rna_fit_kernel if training else ctx.rna_valid_kernel
            if scramble is not None:
                rna_q = rna_q[scramble]
            gram = (1 - rw) * gram + rw * ctx.rna_fit_kernel[np.ix_(valid, valid)]
            cross = (1 - rw) * cross + rw * rna_q[:, valid]
        residual = (ctx.train[valid] - train_base[valid]) / ctx.sd
        if cfg["method"] == "rbf":
            prediction = masked_krr(gram, cross, residual, cfg["ridge"])
        else:
            prediction = product_ridge(gram, cross, ctx.chem, residual, cfg["ridge"])
        return prior + prediction * ctx.sd
