"""Boosting and GPU FiLM/attention residual estimators on the same OOF kernel."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(frozen=True)
class Candidate:
    family: str
    objective: str
    width: int
    depth: int
    dropout: float
    trees: int = 700
    rate: float = .035


class FilmAttention(nn.Module):
    def __init__(self, cell_dims, drug_dims, joint_dim, width, depth, dropout):
        super().__init__()
        self.cell_token = nn.ModuleList(nn.Linear(n, width) for n in cell_dims)
        self.drug_token = nn.ModuleList(nn.Linear(n, width) for n in drug_dims)
        self.modulators = nn.ModuleList(nn.Linear(width, 2 * width) for _ in range(depth))
        self.attention = nn.ModuleList(nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
                                       for _ in range(depth))
        self.norm = nn.ModuleList(nn.LayerNorm(width) for _ in range(depth))
        self.ffn = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                                nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))
                                 for _ in range(depth))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(width + joint_dim, width), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, cells, drugs, joint):
        c = torch.stack([module(x) for module, x in zip(self.cell_token, cells)], dim=1)
        d = torch.stack([module(x) for module, x in zip(self.drug_token, drugs)], dim=1)
        summary = d.mean(dim=1)
        for film, attention, norm, ffn in zip(self.modulators, self.attention, self.norm, self.ffn):
            gain, shift = film(summary).chunk(2, dim=-1)
            modulated = norm(c) * (1 + .2 * gain.tanh()[:, None]) + .2 * shift[:, None]
            delta = attention(modulated, d, d, need_weights=False)[0]
            c = c + self.dropout(delta)
            c = c + ffn(c)
        return self.head(torch.cat([c.mean(dim=1), joint], dim=1)).flatten()


def group_key(panel):
    return pd.MultiIndex.from_frame(panel[["source_id", "cell_id"]])


def rank_data(panel, x, y):
    label = pd.Series(y).groupby(group_key(panel), sort=False).rank(pct=True, method="average")
    graded = ((label.to_numpy() * 30).clip(0, 30)).astype(np.int32)
    frame = panel[["source_id", "cell_id"]].astype(str).agg("|".join, axis=1).to_numpy()
    order = np.argsort(frame, kind="stable")
    _, counts = np.unique(frame[order], return_counts=True)
    return x[order], graded[order], counts


@dataclass
class Boost:
    estimator: object
    multiplier: float
    center: float
    backend: str

    def predict(self, x):
        if self.backend == "lightgbm":
            result = self.estimator.booster_.predict(x, num_iteration=self.estimator.best_iteration_)
        else:
            result = self.estimator.predict(x)
        return ((result - self.center) * self.multiplier).astype(np.float32)


def fit_boost(train, ft, hold, spec):
    try:
        from lightgbm import LGBMRanker, LGBMRegressor, early_stopping, log_evaluation
    except ImportError:
        if spec.objective == "rank":
            raise RuntimeError("LambdaRank requires the isolated LightGBM installation")
        from sklearn.ensemble import HistGradientBoostingRegressor
        variant = "squared_error" if spec.objective == "l2" else "absolute_error"
        mdl = HistGradientBoostingRegressor(loss=variant, max_iter=spec.trees,
                                            max_leaf_nodes=spec.width, learning_rate=spec.rate,
                                            max_depth=spec.depth, early_stopping=False)
        y = train.aligned.to_numpy() - train.oof.to_numpy()
        mdl.fit(ft[~hold], y[~hold])
        return Boost(mdl, 1., 0., "sklearn-hist"), []
    setup = {"n_estimators": spec.trees, "learning_rate": spec.rate, "num_leaves": spec.width,
             "max_depth": spec.depth, "min_child_samples": 110, "colsample_bytree": .85,
             "reg_lambda": .15, "n_jobs": 6, "verbosity": -1, "deterministic": True,
             "force_col_wise": True, "random_state": 9407}
    callbacks = [early_stopping(45, verbose=False), log_evaluation(0)]
    train_y = train.aligned.to_numpy()
    residual = train_y - train.oof.to_numpy()
    x, x_check = ft[~hold], ft[hold]
    if spec.objective == "rank":
        y_train = train.iloc[np.flatnonzero(~hold)].reset_index(drop=True)
        y_check = train.iloc[np.flatnonzero(hold)].reset_index(drop=True)
        xx, relevance, group = rank_data(y_train, x, train_y[~hold])
        vx, vv, vg = rank_data(y_check, x_check, train_y[hold])
        model = LGBMRanker(objective="lambdarank", lambdarank_truncation_level=8,
                            label_gain=list(range(31)), **setup)
        model.fit(xx, relevance, group=group, eval_set=[(vx, vv)], eval_group=[vg],
                  eval_at=[5], callbacks=callbacks)
        rank = model.booster_.predict(x, num_iteration=model.best_iteration_)
        center = float(np.mean(rank))
        scale = float(np.std(residual[~hold], ddof=0) / max(np.std(rank), 1e-6))
    else:
        model = LGBMRegressor(objective="huber" if spec.objective == "huber" else "regression", **setup)
        model.fit(x, residual[~hold], eval_set=[(x_check, residual[hold])],
                  eval_metric="l2", callbacks=callbacks)
        center, scale = 0., 1.
    return Boost(model, scale, center, "lightgbm"), model.evals_result_


def partition_indices(groups, use, rng, maximum):
    positions = np.flatnonzero(use)
    _, keys = np.unique(groups[positions], return_inverse=True)
    split = pd.Series(positions).groupby(keys).agg(list).tolist()
    rng.shuffle(split)
    pending = []
    for group in split:
        if pending and len(pending) + len(group) > maximum:
            yield np.asarray(pending, np.int64)
            pending = []
        pending.extend(group)
    if pending:
        yield np.asarray(pending, np.int64)


def listnet(estimate, truth, groups):
    slots = int(groups.max().item()) + 1
    maximum = torch.full((slots,), -float("inf"), device=groups.device)
    logsum = maximum.scatter_reduce(0, groups, estimate.detach(), reduce="amax", include_self=True)
    exp = torch.exp(estimate - logsum[groups])
    z = torch.zeros_like(maximum).scatter_add(0, groups, exp)
    truth_max = maximum.scatter_reduce(0, groups, truth.detach(), reduce="amax", include_self=True)
    rel = torch.exp(truth - truth_max[groups])
    zr = torch.zeros_like(maximum).scatter_add(0, groups, rel)
    return -(rel / zr[groups] * (estimate - logsum[groups] - z[groups].log())).sum() / len(groups.unique())


def correlation_loss(pred, target):
    a = pred - pred.mean()
    b = target - target.mean()
    return 1. - (a @ b) / (a.norm() * b.norm()).clamp_min(1e-8)


@dataclass
class NeuralResidual:
    model: FilmAttention
    amplitude: float
    center: float
    scale: float

    def predict(self, features, panel, base, fitting, device):
        ci, di, joint, seen = features.roles(panel, base, fitting)
        cells = [torch.as_tensor(x, device=device) for x in features.cell]
        drugs = [torch.as_tensor(x, device=device) for x in features.drug]
        j = torch.as_tensor(joint, device=device)
        j[:, 0] = (j[:, 0] - self.center) / self.scale
        model = self.model.to(device).eval()
        values = []
        with torch.inference_mode():
            for part in range(0, len(ci), 2048):
                s = slice(part, part + 2048)
                c = torch.as_tensor(ci[s], device=device)
                d = torch.as_tensor(di[s], device=device)
                values.append(model([x[c] for x in cells], [x[d] for x in drugs], j[s]).cpu().numpy())
        return np.where(seen, self.amplitude * np.concatenate(values), 0.)


def fit_neural(train, features, hold, spec, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    base = train.oof.to_numpy(np.float32)
    target = train.aligned.to_numpy(np.float32) - base
    ci, di, raw_joint, mask = features.roles(train, base, True)
    groups = pd.factorize(group_key(train))[0]
    active = mask & ~hold
    checking = mask & hold
    if active.sum() == 0 or checking.sum() == 0:
        raise ValueError("Residual model needs assay support in both role subsets")
    amp = max(float(np.std(target[active])), .05)
    center, scale = float(base[active].mean()), max(float(base[active].std()), .05)
    raw_joint[:, 0] = (raw_joint[:, 0] - center) / scale
    joint = torch.as_tensor(raw_joint, device=device)
    cell = [torch.as_tensor(x, device=device) for x in features.cell]
    drug = [torch.as_tensor(x, device=device) for x in features.drug]
    ci, di, g = [torch.as_tensor(x, device=device) for x in (ci, di, groups)]
    aligned = torch.as_tensor(train.aligned.to_numpy(np.float32) / amp, device=device)
    base_amp = torch.as_tensor(base / amp, device=device)
    y = torch.as_tensor(target / amp, device=device)
    m = FilmAttention([x.shape[1] for x in cell], [x.shape[1] for x in drug], joint.shape[1],
                      spec.width, spec.depth, spec.dropout).to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=spec.rate, weight_decay=.01)
    epochs = spec.trees
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best, state, stale, logs = np.inf, None, 0, []

    def forward(index):
        p = m([x[ci[index]] for x in cell], [x[di[index]] for x in drug], joint[index])
        reg = nn.functional.smooth_l1_loss(p, y[index])
        cor = correlation_loss(p, y[index])
        gid = torch.unique(g[index], return_inverse=True)[1]
        sort = listnet(.5 * (p + base_amp[index]), .5 * aligned[index], gid)
        return reg + .12 * cor + .06 * sort

    for epoch in range(epochs):
        m.train()
        train_loss = []
        for rows in partition_indices(groups, active, rng, 1536):
            n = torch.as_tensor(rows, device=device)
            loss = forward(n)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.)
            optimizer.step()
            train_loss.append(float(loss.detach()))
        lr.step()
        m.eval()
        with torch.no_grad():
            metrics = [float(forward(torch.as_tensor(rows, device=device)))
                       for rows in partition_indices(groups, checking, np.random.default_rng(seed), 2048)]
        value = float(np.mean(metrics))
        logs.append({"epoch": epoch + 1, "fit": float(np.mean(train_loss)), "early_stop": value})
        if value < best:
            state = {key: v.detach().clone().cpu() for key, v in m.state_dict().items()}
            best, stale = value, 0
        else:
            stale += 1
        if stale >= 5 and epoch >= 7:
            break
    if state is None:
        raise RuntimeError("No finite residual network early-stop state")
    m.load_state_dict(state)
    return NeuralResidual(m, amp, center, scale), logs


def search_space():
    return [
        Candidate("gbdt", obj, leaves, depth, 0.)
        for obj in ("l2", "huber", "rank") for leaves, depth in ((31, 5), (63, 7))
    ] + [
        Candidate("neural", "joint", width, depth, drop, trees=24, rate=.0012)
        for width, depth, drop in ((256, 3, .10), (256, 4, .25), (512, 3, .25), (512, 4, .40))
    ]
