"""Family B direct GBDT rank objectives and Family D cross-fit reranking."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from .data import BUDGETS
from .kernels import base, eb_predict, residual_packet


def row_features(ctx, packet, order, dual, q, *, prior=None, pc=None, stage=None):
    """Emit guide, anchor, structural and RNA features on the requested query drugs."""
    prior = base(ctx, dual) if prior is None else prior
    pc = ctx.pc_valid if pc is None else pc
    n, d, k = len(packet), len(q), len(order)
    z = residual_packet(ctx, packet, order, prior)
    observed = np.isfinite(packet)
    levels = np.where(observed, (packet - ctx.mean[order]) / ctx.sd[order], np.nan)
    anchors = np.zeros((n, 12))
    residuals = np.zeros_like(anchors)
    known = np.zeros_like(anchors)
    anchors[:, :k] = np.nan_to_num(levels)
    residuals[:, :k] = z
    known[:, :k] = observed
    counts = np.maximum(1, observed.sum(1))
    avg = anchors.sum(1) / counts
    spread = np.sqrt(np.maximum(np.square(anchors).sum(1) / counts - avg ** 2, 0.))
    minimum = np.where(observed.any(1), np.min(np.where(observed, levels, np.inf), axis=1, initial=np.inf), 0.)
    maximum = np.where(observed.any(1), np.max(np.where(observed, levels, -np.inf), axis=1, initial=-np.inf), 0.)
    sample = np.column_stack((anchors, residuals, known, avg, spread, minimum, maximum, np.full(n, k),
                              pc if dual else np.zeros_like(pc)))
    cov = ctx.cov_rna if dual else ctx.cov
    def aggregate(matrix, a):
        link = matrix[np.ix_(a, q)]
        return (z @ link) / np.maximum((observed @ np.abs(link)), 1e-8)

    similarity = aggregate(ctx.tanimoto, order)
    targets = aggregate(ctx.targets, order)
    correl = aggregate(cov, order)
    covariance = aggregate(cov * np.outer(ctx.sd, ctx.sd), order)
    fit_score = eb_predict(ctx, packet, order, dual, {"ridge": .3, "alpha": .25, "prior": "pan"}, prior)[:, q]
    mu, sd = np.broadcast_to(ctx.mean[q], (n, d)), np.broadcast_to(ctx.sd[q], (n, d))
    dynamic = [prior[:, q], mu, sd, similarity, targets, correl, covariance, fit_score]
    if stage is not None:
        dynamic.extend([stage[:, q], rankdata(stage[:, q], axis=1) / max(1, d)])
    return np.concatenate([np.broadcast_to(sample[:, None, :], (n, d, sample.shape[1])),
                           np.broadcast_to(ctx.embedding[q][None], (n, d, ctx.embedding.shape[1])),
                           np.stack(dynamic, axis=-1)], axis=-1).astype(np.float32)


def region(order, k, d):
    return np.setdiff1d(np.arange(d), order[:12 if k == 12 else 8])


@dataclass
class Rows:
    features: np.ndarray
    residual: np.ndarray
    relevance: np.ndarray
    group: np.ndarray


def format_rows(f, y, prior):
    valid = np.isfinite(y)
    if not valid.any(axis=1).all():
        raise ValueError("Each training group needs at least one measured query drug")
    # Rank gain on an ordinal 0–30 axis with linear LambdaRank gain.
    rank = rankdata(y, axis=1, nan_policy="omit")
    rel = np.rint(30 * (rank - 1) / np.maximum(valid.sum(1, keepdims=True) - 1, 1))
    return Rows(f[valid], (y - prior)[valid], rel[valid].astype(int), valid.sum(1))


def training_rows(ctx, parts, order, dual, stage_config):
    index = {key: i for i, key in enumerate(ctx.groups)}
    group_buckets = []
    for part in parts:
        pos = np.array([index[g] for g in part.valid_groups])
        for k in BUDGETS:
            # Fit on every eligible development sample at K=5 and a fixed
            # held-out slice on each secondary budget for budget conditioning.
            pos_local = np.arange(len(pos)) if k == 5 else np.arange(len(pos))[int(k) % 3::3]
            a, q = order[:k], region(order, k, len(ctx.mean))
            keep = np.isfinite(ctx.train[pos[pos_local]][:, q]).sum(1) >= 15
            pos_local = pos_local[keep]
            if not len(pos_local):
                continue
            chosen = pos[pos_local]
            obs = ctx.train[chosen][:, a]
            part_prior = base(part, dual)[pos_local]
            p = None
            target = ctx.train[chosen][:, q]
            # Transfer subfold Activity Area has its own local affine fit;
            # bring its fitted feature locations back to the parent fit scale.
            if ctx.levels:
                obs = (obs * ctx.span + ctx.offset - part.offset) / part.span
            if stage_config is not None:
                p = eb_predict(part, obs, a, dual, stage_config, part_prior)
            f = row_features(part, obs, a, dual, q, prior=part_prior, pc=ctx.pc_fit[chosen], stage=p)
            offset = np.zeros_like(target) if p is None else p[:, q]
            if p is None:
                offset = part_prior[:, q]
            if ctx.levels:
                offset = (offset * part.span + part.offset - ctx.offset) / ctx.span
            group_buckets.append(format_rows(f, target, offset))
    return Rows(*(np.concatenate([getattr(r, key) for r in group_buckets])
                  for key in ("features", "residual", "relevance", "group")))


def lightgbm_head(kind, train, valid, cfg):
    import lightgbm as lgb

    rank = kind == "rank"
    label = "relevance" if rank else "residual"
    ds = lgb.Dataset(train.features, label=getattr(train, label), group=train.group if rank else None)
    params = dict(objective="lambdarank" if rank else ("huber" if kind == "huber" else "regression"),
                  metric="ndcg" if rank else "rmse", num_threads=1, verbosity=-1, deterministic=True,
                  force_col_wise=True, seed=3017, learning_rate=cfg["lr"], num_leaves=cfg["leaves"],
                  max_depth=cfg["depth"], min_data_in_leaf=cfg["children"], lambda_l2=cfg["l2"],
                  bagging_fraction=cfg["subsample"], bagging_freq=1, feature_fraction=cfg["cols"])
    if rank:
        params.update(eval_at=[5], lambdarank_truncation_level=5, label_gain=list(range(31)))
    ev = lgb.Dataset(valid.features, label=getattr(valid, label), group=valid.group if rank else None, reference=ds)
    history = {}
    model = lgb.train(params, ds, num_boost_round=cfg["rounds"], valid_sets=[ds, ev],
                      valid_names=["train", "inner_val"],
                      callbacks=[lgb.record_evaluation(history), lgb.early_stopping(12, verbose=False)])
    return model, history


def train_head(kind, train, valid, cfg):
    return lightgbm_head(kind, train, valid, cfg)


def rank_to_scale(scores, baseline, amount):
    centered = scores - scores.mean(1, keepdims=True)
    den = np.maximum(centered.std(1, keepdims=True), 1e-7)
    return baseline.mean(1, keepdims=True) + centered / den * amount


@dataclass
class BoostModel:
    ctx: object
    order: np.ndarray
    dual: bool
    config: dict
    heads: dict
    curves: dict

    def predict(self, packet, k, *, scramble=None, training=False):
        ctx, cfg, a = self.ctx, self.config, self.order[:k]
        q = region(self.order, k, len(ctx.mean))
        prior = base(ctx, self.dual, training)
        px = ctx.pc_fit if training else ctx.pc_valid
        if scramble is not None and self.dual:
            prior, px = prior[scramble], px[scramble]
        stage = None
        if cfg["family"] == "D":
            stage = eb_predict(ctx, packet, a, self.dual, cfg["stage"], prior)
        ff = row_features(ctx, packet, a, self.dual, q, prior=prior, pc=px, stage=stage)
        flat = ff.reshape(-1, ff.shape[-1])
        guess = prior.copy() if stage is None else stage.copy()
        if cfg["family"] == "D":
            residual = self.heads["rank"].predict(flat).reshape(len(packet), len(q))
            # The reranker adjusts the shortlist and protects low-ranked drugs.
            rank = rankdata(guess[:, q], axis=1)
            gate = rank >= len(q) - cfg["top"] + 1
            z = (residual - residual.mean(1, keepdims=True)) / np.maximum(residual.std(1, keepdims=True), 1e-7)
            guess[:, q] += gate * z * cfg["blend"] * np.maximum(.05, guess[:, q].std(1, keepdims=True))
            return guess
        fitted = prior[:, q]
        if "reg" in self.heads:
            fitted = fitted + self.heads["reg"].predict(flat).reshape(fitted.shape)
        if "rank" in self.heads:
            rank = self.heads["rank"].predict(flat).reshape(fitted.shape)
            converted = rank_to_scale(rank, fitted, np.maximum(fitted.std(1, keepdims=True), .05))
            fitted = fitted + cfg["blend"] * (converted - fitted)
        guess[:, q] = fitted
        return guess


def fit_boost(ctx, sub, order, dual, cfg, training_cache):
    family = cfg["family"]
    stage_cfg = cfg["stage"] if family == "D" else None
    # Use a split inside the fit groups for early stopping in both inner CV and final refit.
    first = sub[0]
    val_pos = np.where(np.isin(ctx.groups, first.valid_groups))[0]
    data_key = (tuple(order), dual, str(stage_cfg))
    if data_key not in training_cache:
        training_cache[data_key] = training_rows(ctx, sub, order, dual, stage_cfg)
    all_rows = training_cache[data_key]
    a, q = order[:5], region(order, 5, len(ctx.mean))
    prior = base(ctx, dual)
    # The selection fold is allowed for development callbacks, whereas an
    # outcome-free final forecast reconstructs an internal development split.
    if ctx.valid is not None:
        use = np.isfinite(ctx.valid[:, q]).sum(1) >= 15
        values = ctx.valid[use][:, a]
        p0 = prior[use]
        stage = None if stage_cfg is None else eb_predict(ctx, values, a, dual, stage_cfg, p0)
        f = row_features(ctx, values, a, dual, q, prior=p0, pc=ctx.pc_valid[use], stage=stage)
        ev = format_rows(f, ctx.valid[use][:, q], (p0 if stage is None else stage)[:, q])
    else:
        values = first.valid[:, a]
        p0 = base(first, dual)
        stage = None if stage_cfg is None else eb_predict(first, values, a, dual, stage_cfg, p0)
        f = row_features(first, values, a, dual, q, prior=p0, pc=ctx.pc_fit[val_pos], stage=stage)
        ev = format_rows(f, first.valid[:, q], (p0 if stage is None else stage)[:, q])
        # Round count was fixed from inner curves. Keep all fit donor groups.
    heads, curves = {}, {}
    kinds = ("rank",) if family == "D" or cfg["objective"] == "rank" else (
        ("reg", "rank") if cfg["objective"] == "dual" else ("reg",))
    for name in kinds:
        kind = name if name == "rank" else ("huber" if cfg["objective"] == "huber" else "l2")
        heads[name], curves[name] = train_head(kind, all_rows, ev, cfg)
    return BoostModel(ctx, order, dual, cfg, heads, curves)
