"""Low-capacity residual temporal operators with a compound permutation control."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd
from sklearn.model_selection import GroupKFold
import torch
from torch import nn

from .linear import conditions, point_fits
from .data import parent_weights

CONFIGS = {
    "denoising": (24, 0.13, 26, "none"),
    "film": (32, 0.16, 30, "unshuffled"),
    "attention": (32, 0.22, 35, "unshuffled"),
    "film_shuffled": (32, 0.16, 30, "shuffled"),
    "film_no_structure": (32, 0.16, 30, "none"),
}


def feature_conditions(rows, x, mode, seed):
    values = conditions(rows, x, True)
    if mode == "none":
        values[:, 5:] = 0
    if mode == "shuffled":
        codes, parents = rows.parent.factorize(sort=True)
        order = np.random.default_rng(seed).permutation(len(parents))
        reference = np.zeros((len(parents), values.shape[1]-5))
        reference[codes] = values[:, 5:]
        values[:, 5:] = reference[order][codes]
    return values.astype(np.float32)


class ResidualNet(nn.Module):
    def __init__(self, mode, width, dropout, n_input, n_context, n_output):
        super().__init__()
        self.mode = mode
        self.cond = nn.Sequential(nn.Linear(n_context, width), nn.LayerNorm(width), nn.SiLU())
        self.rna = nn.Sequential(nn.Linear(n_input, width), nn.LayerNorm(width), nn.SiLU())
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, n_output))
        if mode == "attention":
            self.token = nn.Linear(n_input // 4, width)
            self.attn = nn.MultiheadAttention(width, num_heads=4, dropout=dropout, batch_first=True)
        else:
            self.film = nn.Linear(width, width*2)

    def forward(self, x, conditions):
        ctx = self.cond(conditions)
        genes = self.rna(x)
        if self.mode == "attention":
            tokens = self.token(x[:, :4 * (x.shape[1] // 4)].reshape(len(x), 4, -1))
            interaction = self.attn(ctx[:, None], tokens, tokens, need_weights=False)[0][:, 0]
            return self.out(genes + interaction)
        scale, add = self.film(ctx).chunk(2, dim=1)
        return self.out(genes * (1. + .25 * scale.tanh()) + add)


def torch_matrix(x, device):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def eigenbasis(values, width):
    _, _, right = svd(values, full_matrices=False, check_finite=False)
    return right[:min(width, len(right))].astype(np.float32)


@dataclass
class Nonlinear:
    mode: str
    config: tuple
    x_mean: np.ndarray
    x_basis: np.ndarray
    y_basis: np.ndarray
    c_mean: np.ndarray
    c_scale: np.ndarray
    x_scale: np.ndarray
    state: dict

    def predict(self, rows, x, *, seed, device="cpu"):
        width, dropout, _, chem = self.config
        model = ResidualNet(self.mode, width, dropout, len(self.x_basis), len(self.c_mean),
                            len(self.y_basis)).to(device)
        model.load_state_dict(self.state)
        model.eval()
        xp = ((x - self.x_mean) @ self.x_basis.T) / self.x_scale
        context = (feature_conditions(rows, x, chem, seed) - self.c_mean) / self.c_scale
        with torch.inference_mode():
            pred = model(torch_matrix(xp, device), torch_matrix(context, device)).cpu().numpy()
        return pred @ self.y_basis


def cross_residual(train, config):
    output = np.empty_like(train.y)
    bare = (*config[:-1], "")
    for fit, check in GroupKFold(3).split(train.x, groups=train.rows.scaffold):
        rows = np.arange(len(train.rows))
        model = train.select(np.isin(rows, fit))
        _, expected, _ = next(point_fits([bare], model, train.x[check], train.rows.iloc[check]))
        output[check] = train.y[check] - expected
    return output


def fit(train, residual, mode, config, seed, device):
    width, dropout, epochs, chem = config
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    weights = parent_weights(train.rows)
    x_mean = np.average(train.x, axis=0, weights=weights)
    inputs = train.x - x_mean
    x_basis = eigenbasis(inputs, width)
    y_basis = eigenbasis(residual, min(20, width))
    values = inputs @ x_basis.T
    x_scale = np.maximum(values.std(axis=0), .6)
    contexts = feature_conditions(train.rows, train.x, chem, seed)
    c_mean = contexts.mean(axis=0)
    c_scale = np.maximum(contexts.std(axis=0), .4)
    x = torch_matrix(values / x_scale, device)
    c = torch_matrix((contexts - c_mean) / c_scale, device)
    y = torch_matrix(residual @ y_basis.T, device)
    w = torch_matrix(weights, device)
    model = ResidualNet(mode, width, dropout, x.shape[1], c.shape[1], y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.06)
    for _ in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(x)), 1+len(x)//64):
            noise = torch.randn_like(x[ids]) * (.10 if mode == "denoising" else .03)
            output = model(x[ids] + noise, c[ids])
            errors = torch.nn.functional.smooth_l1_loss(output, y[ids], beta=.5, reduction="none").mean(dim=1)
            objective = (errors * w[ids]).sum() / w[ids].sum()
            opt.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
    return Nonlinear(mode, config, x_mean, x_basis, y_basis, c_mean, c_scale, x_scale,
                     {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})
