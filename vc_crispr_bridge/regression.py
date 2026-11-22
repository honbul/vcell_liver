"""Auditable Ridge, ElasticNet, boosted tree and neural scalar heads."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


def regression_metrics(pred, truth, groups):
    if not (np.isfinite(pred).all() and np.isfinite(truth).all()):
        raise ValueError("Nonfinite regression predictions or outcomes")
    pair = pd.DataFrame({"pred": pred, "truth": truth})
    group = pair.groupby(groups, sort=False)
    y_count = group.truth.size()
    variable_truth = (group.truth.nunique() > 1) & (y_count >= 3)
    ranks = group.rank(method="average")
    centered = ranks - ranks.groupby(groups, sort=False).transform("mean")
    dot = (centered.pred * centered.truth).groupby(groups, sort=False).sum()
    norm = centered.pow(2).groupby(groups, sort=False).sum().prod(axis=1).pow(0.5)
    evaluated = (norm > 0) & variable_truth
    correlations = (dot / norm.where(norm > 0)).fillna(0)
    return {
        "within_group_spearman": float(correlations[variable_truth].mean()) if variable_truth.any() else 0.0,
        "constant_prediction_groups": int((variable_truth & ~evaluated).sum()),
        "eligible_groups": int(variable_truth.sum()),
        "rows": len(pair),
        "rmse": float(np.sqrt(np.mean(np.square(pred - truth)))),
        "mae": float(np.abs(pred - truth).mean()),
    }


def validation_value(pred, truth, groups):
    metrics = regression_metrics(pred, truth, groups)
    # Primary criterion is within-group ranking. Absolute fit is a
    # source-only tie-breaker when two predictors induce equal rankings.
    return metrics["within_group_spearman"], -metrics["rmse"]


@dataclass
class Fit:
    score: tuple
    settings: dict
    predictions: list


def make_linear(kind, setting):
    if kind == "ridge":
        return Ridge(alpha=setting), {"alpha": setting}
    params = {"alpha": setting, "l1_ratio": 0.3, "max_iter": 450, "random_state": 41026}
    return ElasticNet(**params), params


def train_scalar(method, data, y, vy, groups, device, seed, modulus=1, weight=None):
    if weight is None:
        weight = np.ones(len(y), dtype=np.float32)
    scaler = StandardScaler().fit(data[0])
    x, vx, *queries = [scaler.transform(a).clip(-15, 15).astype(np.float32) for a in data]
    x_train, y_train, weight_train = x[::modulus], y[::modulus], weight[::modulus]
    best = None
    if method == "mlp":
        return train_mlp(x_train, y_train, weight_train, vx, vy, queries, groups, device, seed)
    if method == "lightgbm":
        from lightgbm import LGBMRegressor
        trials = []
        for leaves, minimum in ((15, 100), (31, 75)):
            params = {"num_leaves": leaves, "min_child_samples": minimum, "learning_rate": 0.045,
                      "n_estimators": 220, "verbosity": -1, "n_jobs": 5, "random_state": seed,
                      "colsample_bytree": 0.8, "reg_lambda": 3.0, "max_bin": 63}
            trials.append((LGBMRegressor(**params), params))
    else:
        values = (10, 100, 1000, 10000) if method == "ridge" else (0.0003, 0.001, 0.003, 0.01)
        trials = [make_linear(method, value) for value in values]
    for model, settings in trials:
        model.fit(x_train, y_train, sample_weight=weight_train)
        value = validation_value(model.predict(vx), vy, groups)
        if best is None or value > best.score:
            best = Fit(value, settings, [model.predict(query) for query in queries])
    return best


@torch.no_grad()
def neural_predict(model, x, device):
    model.eval()
    return np.concatenate([
        model(torch.as_tensor(x[i:i + 8192], device=device)).cpu().numpy().ravel()
        for i in range(0, len(x), 8192)
    ])


def train_mlp(x, y, weight, vx, vy, queries, groups, device, seed):
    tx, ty, tw = [torch.from_numpy(a).to(device) for a in (x, y.astype(np.float32), weight)]
    best = None
    for width in (96, 192):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = nn.Sequential(
            nn.Linear(x.shape[1], width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(width, width), nn.GELU(), nn.Dropout(0.1), nn.Linear(width, 1),
        ).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        no_improvement = 0
        trial_score = (-np.inf, -np.inf)
        for epoch in range(30):
            model.train()
            order = rng.permutation(len(x))
            for off in range(0, len(x), 2048):
                idx = torch.as_tensor(order[off:off + 2048], device=device)
                optim.zero_grad(set_to_none=True)
                estimate = model(tx[idx]).flatten()
                loss = (nn.functional.smooth_l1_loss(estimate, ty[idx], beta=0.2, reduction="none")
                        * tw[idx]).mean()
                loss.backward()
                optim.step()
            val = validation_value(neural_predict(model, vx, device), vy, groups)
            if val > trial_score:
                trial_score, no_improvement = val, 0
            else:
                no_improvement += 1
            if best is None or val > best.score:
                best = Fit(val, {"width": width, "epoch": epoch + 1, "seed": seed},
                           [neural_predict(model, query, device) for query in queries])
            if no_improvement == 5:
                break
    return best
