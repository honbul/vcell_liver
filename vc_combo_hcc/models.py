"""Matched classical and neural combination models with symmetric pair prediction."""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from .splits import OUTCOMES


FAMILIES = {
    "ridge": (50., 500.),
    "elasticnet": (.005, .025),
    "lightgbm": ((140, 23), (250, 47)),
    "deepsynergy": ((192, 14, .15), (256, 24, .25)),
    "matchmaker": ((192, 14, .15), (256, 24, .25)),
    "cross_attention": ((64, 14, .15), (96, 24, .25)),
    "no_crispr": ((64, 14, .15), (96, 24, .25)),
    "no_targets": ((64, 14, .15), (96, 24, .25)),
    "no_foundation": ((64, 14, .15), (96, 24, .25)),
}
CLASSICAL = ("ridge", "elasticnet", "lightgbm")


def source_weights(source, y):
    _, codes, counts = np.unique(source, return_inverse=True, return_counts=True)
    wt = 1 / np.sqrt(counts[codes])
    wt *= np.sqrt(np.isfinite(y).sum(axis=1) / y.shape[1]).clip(.5)
    return (wt / wt.mean()).astype(np.float32)


def targets(rows):
    values = rows.loc[:, OUTCOMES].to_numpy(dtype=np.float32, copy=True)
    # DrugComb HSA is the mean matched-dose additive increment. The HCC
    # response endpoint here is Sanger's maximum inhibited growth gain.
    values[~rows.source.str.startswith("Sanger:").to_numpy(), OUTCOMES.index("added_inhibition")] = np.nan
    return values


def calibration(y):
    if np.any(np.isfinite(y).sum(axis=0) < 5):
        raise ValueError("An outcome has insufficient measurable training rows")
    return np.nanmean(y, axis=0), np.nanstd(y, axis=0).clip(.25)


def fit_classical(name, config, x, y, weights, seed):
    narrow = name != "lightgbm"
    scaler = StandardScaler().fit(x) if narrow else None
    fit = scaler.transform(x) if narrow else x
    target_center, target_scale = calibration(y)
    result = []
    for j in range(len(OUTCOMES)):
        keep = np.isfinite(y[:, j])
        if name == "ridge":
            model = Ridge(alpha=config)
        elif name == "elasticnet":
            model = ElasticNet(alpha=config, l1_ratio=.28, max_iter=1300, tol=2e-3)
        else:
            from lightgbm import LGBMRegressor

            trees, leaves = config
            model = LGBMRegressor(n_estimators=trees, num_leaves=leaves, max_bin=63,
                                  subsample=.83, subsample_freq=1, colsample_bytree=.75,
                                  min_child_samples=48, learning_rate=.05, reg_lambda=8,
                                  n_jobs=4, random_state=seed + j, verbosity=-1)
        if keep.all():
            sample = fit
        else:
            sample = fit[keep]
        model.fit(sample, (y[keep, j] - target_center[j]) / target_scale[j], sample_weight=weights[keep])
        result.append(model)

    def forecast(values):
        data = scaler.transform(values) if narrow else values
        out = np.column_stack([model.predict(data) for model in result])
        return out * target_scale + target_center

    return forecast


def block(inp, width, dropout):
    return nn.Sequential(nn.Linear(inp, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout))


class DeepSynergy(nn.Module):
    def __init__(self, compound, cell, width, dropout):
        super().__init__()
        self.net = nn.Sequential(block(2 * compound + cell, width, dropout),
                                 block(width, width, dropout), block(width, width // 2, dropout),
                                 nn.Linear(width // 2, len(OUTCOMES)))

    def forward(self, a, b, cell):
        return .5 * (self.net(torch.cat([a, b, cell], dim=-1)) + self.net(torch.cat([b, a, cell], dim=-1)))


class MatchMaker(nn.Module):
    def __init__(self, compound, cell, width, dropout):
        super().__init__()
        self.arm_a = nn.Sequential(block(compound + cell, width, dropout), block(width, width // 2, dropout))
        self.arm_b = nn.Sequential(block(compound + cell, width, dropout), block(width, width // 2, dropout))
        self.head = nn.Sequential(block(width, width // 2, dropout), nn.Linear(width // 2, len(OUTCOMES)))

    def forward(self, a, b, cell):
        ac, bc = torch.cat([a, cell], dim=-1), torch.cat([b, cell], dim=-1)
        ab = self.head(torch.cat([self.arm_a(ac), self.arm_b(bc)], dim=-1))
        ba = self.head(torch.cat([self.arm_a(bc), self.arm_b(ac)], dim=-1))
        return .5 * (ab + ba)


class DualAttention(nn.Module):
    def __init__(self, shapes, width, dropout, features):
        super().__init__()
        self.inputs = nn.ModuleDict({name: block(shapes[name], width, dropout) for name in features})
        self.features = tuple(features)
        self.heads = nn.ModuleList([nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.readout = nn.Sequential(block(4 * width, 2 * width, dropout), block(2 * width, width, dropout),
                                     nn.Linear(width, len(OUTCOMES)))

    def tokens(self, branch, context):
        return torch.stack([self.inputs[k](branch[k] if k in branch else context[k]) for k in self.features], dim=1)

    def forward(self, a, b, context):
        left, right = self.tokens(a, context), self.tokens(b, context)
        for layer, norm in zip(self.heads, self.norm):
            new_a = layer(left, right, right, need_weights=False)[0]
            new_b = layer(right, left, left, need_weights=False)[0]
            left, right = norm(left + new_a), norm(right + new_b)
        x, y = left.mean(dim=1), right.mean(dim=1)
        return self.readout(torch.cat([x + y, torch.abs(x - y), x * y, left[:, -1] + right[:, -1]], dim=-1))


@dataclass
class IndexedTensors:
    left: torch.Tensor
    right: torch.Tensor
    cell: torch.Tensor
    assay: torch.Tensor
    dependency_a: torch.Tensor
    dependency_b: torch.Tensor


class NeuralInputs:
    def __init__(self, library, device):
        self.device = device
        self.mol = {mode: self.tensor(data) for mode, data in (
            ("morgan", library.ecfp), ("foundation", library.molecular), ("targets", library.targets),
        )}
        self.cells = {"rna": self.tensor(library.rna), "crispr": self.tensor(library.crispr)}

    def tensor(self, data, integer=False):
        return torch.as_tensor(data, dtype=torch.long if integer else torch.float32, device=self.device)

    def view(self, feature):
        return IndexedTensors(*(self.tensor(v, True) for v in (feature.left, feature.right, feature.cell)),
                              *(self.tensor(v) for v in (feature.assay, feature.dependency_left, feature.dependency_right)))

    def apply(self, network, name, view, i):
        a, b, cell = view.left[i], view.right[i], view.cell[i]
        if name in {"deepsynergy", "matchmaker"}:
            return network(self.mol["morgan"][a], self.mol["morgan"][b],
                           torch.cat([self.cells["rna"][cell], view.assay[i]], dim=-1))
        first = {k: v[a] for k, v in self.mol.items() if k in network.features}
        other = {k: v[b] for k, v in self.mol.items() if k in network.features}
        if "drug_crispr" in network.features:
            first["drug_crispr"], other["drug_crispr"] = view.dependency_a[i], view.dependency_b[i]
        context = {"rna": self.cells["rna"][cell], "crispr": self.cells["crispr"][cell], "assay": view.assay[i]}
        if network.training and "crispr" in network.features:
            # Support the absence of independent gene effect measurements in
            # organoids and some cell cultures without a query-label transform.
            keep = (torch.rand(len(cell), 1, device=self.device) >= .12)
            context["crispr"] = context["crispr"] * keep
            if "drug_crispr" in network.features:
                first["drug_crispr"] = first["drug_crispr"] * keep
                other["drug_crispr"] = other["drug_crispr"] * keep
        return network(first, other, context)

    def network(self, name, width, dropout):
        compound = self.mol["morgan"].shape[-1]
        cell = self.cells["rna"].shape[-1] + 1
        if name == "deepsynergy":
            return DeepSynergy(compound, cell, width, dropout)
        if name == "matchmaker":
            return MatchMaker(compound, cell, width, dropout)
        shapes = {k: v.shape[-1] for k, v in {**self.mol, **self.cells}.items()}
        shapes.update(drug_crispr=4, assay=1)
        excluded = {"no_crispr": {"crispr", "drug_crispr"}, "no_targets": {"targets", "drug_crispr"},
                    "no_foundation": {"foundation"}}
        fields = [k for k in ("morgan", "foundation", "targets", "drug_crispr", "crispr", "assay", "rna")
                  if k not in excluded.get(name, ())]
        return DualAttention(shapes, width, dropout, fields)


@dataclass
class NeuralFit:
    name: str
    model: nn.Module
    center: np.ndarray
    scale: np.ndarray

    def predict(self, inputs, view):
        self.model.eval()
        tensor = inputs.view(view)
        keys = torch.arange(len(view.left), device=inputs.device)
        parts = []
        with torch.inference_mode():
            for i in keys.split(4096):
                parts.append(inputs.apply(self.model, self.name, tensor, i).cpu().numpy())
        return np.concatenate(parts, axis=0) * self.scale + self.center


def fit_neural(name, config, inputs, data, y, weights, seed):
    width, epochs, dropout = config
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = inputs.network(name, width, dropout).to(inputs.device)
    x = inputs.view(data)
    center, scale = calibration(y)
    y_scale = (y - center) / scale
    target = inputs.tensor(np.nan_to_num(y_scale, nan=0., posinf=0., neginf=0.))
    mask = inputs.tensor(np.isfinite(y_scale))
    w = inputs.tensor(weights)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.015)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.00015)
    net.train()
    for _ in range(epochs):
        shuffled = inputs.tensor(rng.permutation(len(y)), integer=True)
        for take in shuffled.split(1024):
            optimizer.zero_grad(set_to_none=True)
            predicted = inputs.apply(net, name, x, take)
            residual = F.smooth_l1_loss(predicted, target[take], reduction="none")
            mz = w[take, None] * mask[take]
            loss = (residual * mz).sum() / mz.sum().clamp_min(1.)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 4.)
            optimizer.step()
        scheduler.step()
    return NeuralFit(name, net, center, scale)
