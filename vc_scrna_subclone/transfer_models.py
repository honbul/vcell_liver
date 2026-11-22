"""Source-fitted baselines and inductive bulk-to-single-cell transfer models."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F


def rank_skill(y, prediction, drugs):
    scores = []
    for drug in np.unique(drugs):
        expected = y[drugs == drug]
        found = prediction[drugs == drug]
        if len(expected) >= 8 and np.ptp(expected) > 0:
            value = spearmanr(expected, found).statistic if np.ptp(found) else 0
            scores.append(float(value))
    if not scores:
        raise ValueError("No variable source-only drug outcomes for selection")
    return float(np.mean(scores)), -float(np.sqrt(np.mean(np.square(prediction - y))))


@dataclass
class Result:
    predictions: np.ndarray
    threshold: float
    validation_spearman: float
    config: dict


def regressors(family, seed):
    if family == "ridge":
        for alpha in (5, 50, 300):
            yield Ridge(alpha=alpha), {"alpha": alpha}
    elif family == "elasticnet":
        for alpha in (0.002, 0.01, 0.04):
            params = {"alpha": alpha, "l1_ratio": 0.25, "max_iter": 1200, "random_state": seed}
            yield ElasticNet(**params), params
    else:
        from lightgbm import LGBMRegressor

        for leaves, count in ((9, 180), (25, 220)):
            params = {"num_leaves": leaves, "n_estimators": count, "n_jobs": 4, "max_bin": 63,
                      "learning_rate": 0.05, "verbosity": -1, "random_state": seed,
                      "colsample_bytree": 0.8, "reg_lambda": 3.0, "min_child_samples": 24}
            yield LGBMRegressor(**params), params


def ordinary(family, x, d, y, weight, vx, vd, vy, qx, qd, seed):
    select, estimate, fit_values = np.zeros(len(vy)), np.zeros(len(qd)), np.zeros(len(y))
    settings = {}
    for drug in np.unique(d):
        fit = d == drug
        valid, query = vd == drug, qd == drug
        if fit.sum() < 20 or valid.sum() < 8:
            raise ValueError("One targeted therapy lacks independent training or validation records")
        scale = StandardScaler().fit(x[fit])
        train_x = scale.transform(x[fit])
        valid_x = scale.transform(vx[valid])
        candidates = []
        for candidate, config in regressors(family, seed):
            candidate.fit(train_x, y[fit], sample_weight=weight[fit])
            value = rank_skill(vy[valid], candidate.predict(valid_x), vd[valid])
            candidates.append((value, candidate, config))
        _, best, config = max(candidates, key=lambda row: row[0])
        select[valid] = best.predict(valid_x)
        if query.any():
            estimate[query] = best.predict(scale.transform(qx[query]))
        fit_values[fit] = best.predict(train_x)
        settings[str(drug)] = config
    settings["threshold_rule"] = "Per-therapy source training 75th percentile of fitted scores"
    quantiles = {str(i): float(np.quantile(fit_values[d == i], 0.75)) for i in np.unique(d)}
    settings["thresholds"] = quantiles
    return Result(estimate.astype(np.float32), quantiles["0"], rank_skill(vy, select, vd)[0], settings)


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, weight):
        ctx.weight = weight
        return z.view_as(z)

    @staticmethod
    def backward(ctx, delta):
        return -ctx.weight * delta, None


def encoder(features, width, latent):
    return nn.Sequential(nn.Linear(features, width), nn.LayerNorm(width), nn.GELU(),
                         nn.Dropout(0.13), nn.Linear(width, latent), nn.LayerNorm(latent))


class SharedTransfer(nn.Module):
    def __init__(self, dimension, width, n_drugs):
        super().__init__()
        self.encoder = encoder(dimension, width, 64)
        self.head = nn.Sequential(nn.Linear(64 + n_drugs, 64), nn.GELU(), nn.Dropout(0.15),
                                  nn.Linear(64, 1))
        self.domain = nn.Sequential(nn.Linear(64, 48), nn.GELU(), nn.Linear(48, 1))
        self.n_drugs = n_drugs

    def response(self, z, drug):
        return self.head(torch.cat((z, F.one_hot(drug, self.n_drugs)), dim=1)).flatten().sigmoid()

    def predict(self, x, drug, target=False):
        return self.response(self.encoder(x), drug)


class SeparateTransfer(SharedTransfer):
    def __init__(self, dimension, width, n_drugs):
        super().__init__(dimension, width, n_drugs)
        self.singlecell = encoder(dimension, width, 64)
        self.decode_bulk = nn.Sequential(nn.Linear(64, width), nn.GELU(), nn.Linear(width, dimension))
        self.decode_cells = nn.Sequential(nn.Linear(64, width), nn.GELU(), nn.Linear(width, dimension))

    def predict(self, x, drug, target=False):
        extractor = self.singlecell if target else self.encoder
        return self.response(extractor(x), drug)


def gaussian_mmd(a, b):
    aa, bb, ab = [torch.cdist(x, y).square() / a.shape[1] for x, y in ((a, a), (b, b), (a, b))]
    return sum(((-aa / s).exp().mean() + (-bb / s).exp().mean() - 2 * (-ab / s).exp().mean())
               for s in (0.25, 1.0, 4.0)) / 3


def compact_states(latent, states):
    if len(latent) == 0:
        return latent.sum()
    unique, inverse, support = torch.unique(states, return_inverse=True, return_counts=True)
    barycenter = torch.zeros((len(unique), latent.shape[1]), dtype=latent.dtype, device=latent.device)
    barycenter.index_add_(0, inverse, latent)
    centers = barycenter[inverse] / support[inverse].reshape(-1, 1)
    return (latent - centers).square().mean()


@torch.no_grad()
def replay(model, x, d, target):
    model.eval()
    output = [model.predict(x[i:i + 2048], d[i:i + 2048], target).cpu().numpy()
              for i in range(0, len(x), 2048)]
    return np.concatenate(output)


def run_loss(method, model, train, drugs, expected, weights, cells, states, strength):
    if method == "mmd":
        bulk_hidden = model.encoder(F.dropout(train, p=0.15, training=True))
        cell_hidden = model.singlecell(F.dropout(cells, p=0.15, training=True))
        auxiliary = (0.08 * F.mse_loss(model.decode_bulk(bulk_hidden), train)
                     + 0.08 * F.mse_loss(model.decode_cells(cell_hidden), cells)
                     + strength * gaussian_mmd(bulk_hidden, cell_hidden)
                     + 0.03 * compact_states(cell_hidden, states))
    else:
        bulk_hidden = model.encoder(train)
        auxiliary = torch.zeros((), device=train.device)
        if method == "adversarial":
            target_hidden = model.encoder(cells)
            domain = torch.cat((bulk_hidden, target_hidden))
            domain_outcome = torch.cat((torch.zeros(len(train)), torch.ones(len(cells)))).to(train.device)
            auxiliary = F.binary_cross_entropy_with_logits(
                model.domain(Reverse.apply(domain, strength)).flatten(), domain_outcome,
            )
    effect_loss = F.smooth_l1_loss(model.response(bulk_hidden, drugs), expected, beta=0.2, reduction="none")
    return (effect_loss * weights).mean() + auxiliary


def neural(method, x, d, y, weight, vx, vd, vy, qx, qd, adaptation, device, seed):
    scaled = StandardScaler().fit(x)
    xs, valid, query, unlabeled = [scaled.transform(a).clip(-8, 8).astype(np.float32)
                                    for a in (x, vx, qx, adaptation)]
    cluster = np.zeros(len(unlabeled), np.int64)
    if method == "mmd":
        kmeans = MiniBatchKMeans(n_clusters=16, batch_size=512, n_init=4, random_state=seed)
        cluster = kmeans.fit_predict(unlabeled)
    arrays = (xs, d, y, weight, valid, vd, query, qd, unlabeled, cluster)
    tx, td, ty, tw, tvx, tvd, tqx, tqd, ta, tc = [torch.as_tensor(a, device=device) for a in arrays]
    n_drugs = int(np.max(d)) + 1
    best, solution = (-np.inf, -np.inf), None
    for width, limit in ((128, 0.12), (256, 0.3)):
        torch.manual_seed(seed)
        random = np.random.default_rng(seed)
        cls = SeparateTransfer if method == "mmd" else SharedTransfer
        model = cls(x.shape[1], width, n_drugs).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.015)
        local = (-np.inf, -np.inf)
        stale = 0
        for epoch in range(1, 36):
            model.train()
            order = random.permutation(len(d))
            for start in range(0, len(order), 160):
                source = torch.as_tensor(order[start:start + 160], device=device)
                sample = torch.as_tensor(random.integers(len(ta), size=len(source)), device=device)
                optim.zero_grad(set_to_none=True)
                loss = run_loss(method, model, tx[source], td[source], ty[source], tw[source],
                                ta[sample], tc[sample], limit * min(1, epoch / 6))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3)
                optim.step()
            value = rank_skill(vy, replay(model, tvx, tvd, target=False), vd)
            if value > local:
                local, stale = value, 0
            else:
                stale += 1
            if value > best:
                best = value
                source_scores = replay(model, tx, td, target=False)
                thresholds = {str(i): float(np.quantile(source_scores[d == i], 0.75)) for i in np.unique(d)}
                solution = Result(replay(model, tqx, tqd, target=True), thresholds["0"], value[0],
                                  {"epoch": epoch, "width": width, "alignment_weight": limit, "seed": seed,
                                   "thresholds": thresholds,
                                   "threshold_rule": "Per-therapy source training 75th percentile"})
            if stale == 6:
                break
    return solution
