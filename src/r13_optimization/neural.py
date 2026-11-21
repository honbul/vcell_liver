"""Task-aligned residual models for small compound-group training splits."""

from collections import defaultdict
from functools import cached_property
import os

import numpy as np
from sklearn.utils.extmath import randomized_svd
import torch
from torch import nn

from src.r11_chemical_probe.dataset import ENTITY
from src.r11_chemical_probe.models import Interaction
from src.r12_multiaxis.network import center, packs, pool_weights


def candidates():
    """Return paired trials, with three fixed training lengths per setting."""
    common = {
        "head": "concat", "chemical": "ecfp", "context": "stable_pca",
        "basis": "supervised", "rank": 16, "width": 48, "dropout": 0.15,
        "decay": 0.03, "lr": 0.0012, "loss": "contrast", "anchor": 0.1,
        "energy": 0.05, "seed": 42,
    }
    grid = {
        "compact": {},
        "drug_only": {"head": "drug", "context": "none"},
        "context_product": {"head": "product"},
        "film": {"head": "film"},
        "direct": {"basis": "direct", "rank": 0},
        "pca_residual": {"basis": "pca"},
        "pca_rank32": {"basis": "pca", "rank": 32},
        "supervised_rank8": {"rank": 8},
        "supervised_rank32": {"rank": 32},
        "supervised_rank64": {"rank": 64},
        "low_width": {"width": 24},
        "high_width": {"width": 96},
        "light_regularization": {"dropout": 0.03, "decay": 0.001, "energy": 0.0},
        "strong_regularization": {"dropout": 0.35, "decay": 0.15, "energy": 0.2},
        "small_lr": {"lr": 0.0003},
        "large_lr": {"lr": 0.003},
        "no_anchor": {"anchor": 0.0},
        "strong_anchor": {"anchor": 0.5},
        "contrast_mse": {"loss": "contrast_mse"},
        "absolute_rmse": {"loss": "absolute"},
        "direction": {"loss": "direction"},
        "untreated_pca": {"context": "pca"},
        "plate_adjusted": {"context": "nuisance_pca"},
        "chembert": {"chemical": "chembert"},
        "dual_chemical": {"chemical": "dual"},
        "fingerprint_full": {"chemical": "fullfp"},
    }
    return [
        {**common, **changes, "name": f"nn_{label}_{epochs}", "epochs": epochs}
        for label, changes in grid.items() for epochs in (6, 24, 60)
    ]


def _center_numpy(values, pool, weight):
    totals = np.zeros((int(pool.max()) + 1, values.shape[1]), dtype=values.dtype)
    np.add.at(totals, pool, values * weight[:, None])
    return values - totals[pool]


def _train_projection(observed, *blocks, rank):
    pca, scaler = Interaction._fit(observed, rank)
    return tuple(scaler.transform(pca.transform(x)).astype(np.float32) for x in blocks)


class Preparation:
    """Share immutable training transforms across candidates on a split."""

    def __init__(self, fit):
        self.fit = fit
        self.fit_features = fit.train
        self.infer_features = fit.query
        self.pool, self.pool_weight = pool_weights(fit.train.rows)
        self.pool_weight = self.pool_weight.astype(np.float32)
        self.weight = (fit.weight / fit.weight.sum()).astype(np.float32)
        self.anchor = np.asarray(fit.residual / fit.scale, dtype=np.float32)
        # Center the observed training delta directly. The group-specific
        # leave-group-out mean in fit.residual serves only as an absolute anchor.
        delta = np.asarray(fit.train.delta / fit.scale, dtype=np.float32)
        self.contrast = _center_numpy(delta, self.pool, self.pool_weight)
        self.chemical = {}
        self.context = {}

    def chemical_inputs(self, kind):
        if kind in self.chemical:
            return self.chemical[kind]
        if kind == "ecfp":
            pair = tuple(self.fit.drug_blocks)
        elif kind == "dual":
            pair = tuple(np.concatenate((a, b), axis=1) for a, b in
                         zip(self.chemical_inputs("ecfp"), self.chemical_inputs("chembert")))
        else:
            key = "C" if kind == "chembert" else "E"
            train = self.fit_features.chemical[key]
            query = self.infer_features.chemical[key]
            distinct = ~self.fit_features.rows.duplicated(ENTITY)
            unique = train[distinct]
            if kind == "chembert":
                pair = _train_projection(unique, train, query, rank=24)
            elif kind == "fullfp":
                keep = np.flatnonzero(unique.std(axis=0) > 0)
                mean = unique[:, keep].mean(axis=0)
                scale = np.maximum(unique[:, keep].std(axis=0), 0.18)
                pair = tuple(((values[:, keep] - mean) / scale).astype(np.float32)
                             for values in (train, query))
            else:
                raise ValueError(f"Unknown chemical representation {kind}")
        self.chemical[kind] = pair
        return pair

    def context_inputs(self, kind):
        if kind in self.context:
            return self.context[kind]
        if kind == "none":
            t, q = self.fit_features.rows, self.infer_features.rows
            pair = (np.empty((len(t), 0), np.float32), np.empty((len(q), 0), np.float32))
        elif kind == "pca":
            design = self.fit.inputs("pca")
            pair = (design.train[1], design.query[1])
        else:
            names = {"stable_pca": "stable", "nuisance_pca": "nuisance1"}
            bank = self.fit.context_bank(names[kind])
            pair = _train_projection(
                np.stack([bank[k] for k in self.fit.fit_units]),
                np.stack([bank[k] for k in self.fit.train_units]),
                np.stack([bank[k] for k in self.fit.query_units]),
                rank=16,
            )
        self.context[kind] = pair
        return pair

    @cached_property
    def supervised_signal(self):
        # Use fitted, pool-centered linear chemical effects to find directions
        # that the chemical features can predict from the training data.
        root = np.sqrt(self.weight)
        raw = self.fit_features.chemical["E"]
        unique = raw[~self.fit_features.rows.duplicated(ENTITY)]
        projected, = _train_projection(unique, raw, rank=96)
        design = _center_numpy(projected, self.pool, self.pool_weight)
        design = design.astype(np.float64) * root[:, None]
        covariance = design.T @ design
        covariance.flat[::len(covariance) + 1] += 0.5
        cross = design.T @ (self.contrast * root[:, None])
        # The right singular vectors are unchanged by the orthogonal Q.
        upper = np.linalg.qr(design, mode="r")
        return upper @ np.linalg.solve(covariance, cross)

    @cached_property
    def supervised_basis(self):
        rank = min(64, *self.supervised_signal.shape)
        _, singular, basis = randomized_svd(self.supervised_signal, rank, random_state=42)
        keep = singular > 1e-6 * max(singular[0], 1e-12)
        return basis[keep].astype(np.float32)

    @cached_property
    def pca_basis(self):
        weight = np.sqrt(self.weight)
        weighted = self.contrast * weight[:, None]
        _, _, basis = randomized_svd(weighted, min(32, *weighted.shape), random_state=42)
        return basis.astype(np.float32)

    def basis(self, kind, rank):
        if kind == "direct":
            return None
        basis = {"supervised": lambda: self.supervised_basis, "pca": lambda: self.pca_basis}[kind]()
        if not len(basis):
            return self.pca_basis[:rank]
        return basis[:rank]


class ResidualModel(nn.Module):
    """Emit small residual coordinates using shared or context-specific effects."""

    def __init__(self, drug_dim, context_dim, outputs, width, dropout, head):
        super().__init__()
        self.head = head
        self.drug = nn.Sequential(nn.Linear(drug_dim, width), nn.GELU(), nn.Dropout(dropout))
        self.context = None if head == "drug" else nn.Sequential(nn.Linear(context_dim, width), nn.GELU())
        self.transform = None
        if head == "concat":
            self.transform = nn.Linear(width * 2, width)
        elif head == "product":
            self.transform = nn.Linear(width, width, bias=False)
        elif head == "film":
            self.transform = nn.Linear(width, width * 2)
        if head in {"product", "film"}:
            nn.init.zeros_(self.transform.weight)
            if self.transform.bias is not None:
                nn.init.zeros_(self.transform.bias)
        self.readout = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(width, outputs, bias=False))
        # All candidates start at the shared mean prediction.
        nn.init.zeros_(self.readout[-1].weight)

    def forward(self, drug, context):
        d = self.drug(drug)
        if self.head == "drug":
            return self.readout(d)
        c = self.context(context)
        if self.head == "concat":
            x = d + self.transform(torch.cat((d, c), dim=1))
        elif self.head == "product":
            x = d + d * torch.tanh(self.transform(c))
        else:
            gain, bias = self.transform(c).chunk(2, dim=1)
            x = d * (1 + torch.tanh(gain)) + bias
        return self.readout(x)


def _coordinates(values, basis):
    """Preserve all-gene squared error when training on orthogonal coordinates."""
    if basis is None:
        return values, np.zeros(len(values), dtype=np.float32)
    inside = values @ basis.T
    omitted = np.maximum(0, np.square(values).sum(axis=1) - np.square(inside).sum(axis=1))
    return inside.astype(np.float32), omitted.astype(np.float32)


def _run(prep, spec, epochs):
    fit = prep.fit
    seed = spec["seed"]
    device = torch.device(os.environ.get("R13_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def tensor(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=device)

    chemical = tuple(tensor(x) for x in prep.chemical_inputs(spec["chemical"]))
    context = tuple(tensor(x) for x in prep.context_inputs(spec["context"]))
    basis = prep.basis(spec["basis"], spec["rank"])
    contrast, contrast_noise = [tensor(x) for x in _coordinates(prep.contrast, basis)]
    anchor, anchor_noise = [tensor(x) for x in _coordinates(prep.anchor, basis)]
    genes = prep.anchor.shape[1]
    model = ResidualModel(chemical[0].shape[1], context[0].shape[1], contrast.shape[1],
                          spec["width"], spec["dropout"], spec["head"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["decay"])
    schedule = [(tensor(i, torch.long), tensor(p, torch.long), tensor(w), n)
                for i, p, w, n in packs(fit.train.rows, batch_rows=1024)]
    weight = tensor(prep.weight)
    output_map = None if basis is None else tensor(basis)
    mode = spec["loss"]
    episode_count = len(schedule)
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for epoch in range(1, max(epochs) + 1):
            model.train()
            for index in rng.permutation(episode_count):
                ix, pool, pw, n = schedule[index]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(chemical[0][ix], context[0][ix])
                error_anchor = ((prediction - anchor[ix]).square().sum(dim=1) + anchor_noise[ix]) / genes
                pred_contrast = center(prediction, pool, pw, n)
                error_contrast = (
                    (pred_contrast - contrast[ix]).square().sum(dim=1) + contrast_noise[ix]
                ) / genes
                if mode == "absolute":
                    loss = torch.sqrt(error_anchor + 1e-6)
                elif mode == "contrast_mse":
                    loss = error_contrast + spec["anchor"] * error_anchor
                else:
                    loss = torch.sqrt(error_contrast + 1e-6) + spec["anchor"] * error_anchor
                    if mode == "direction":
                        # Keep cosine gradients finite for near-zero predicted
                        # and observed residuals and retain a magnitude loss.
                        dot = (pred_contrast * contrast[ix]).sum(dim=1)
                        c_norm = (contrast[ix].square().sum(dim=1) + contrast_noise[ix] + genes * 0.01).sqrt()
                        p_norm = (pred_contrast.square().sum(dim=1) + genes * 0.01).sqrt()
                        loss = loss - 0.2 * dot / (c_norm * p_norm)
                loss = loss + spec["energy"] * prediction.square().sum(dim=1) / genes
                (episode_count * torch.dot(weight[ix], loss)).backward()
                optimizer.step()
            if epoch in epochs:
                model.eval()
                with torch.inference_mode():
                    pred = model(chemical[1], context[1])
                    if output_map is not None:
                        pred = pred @ output_map
                    prediction = pred.cpu().numpy() * fit.scale + fit.offset
                yield epoch, prediction.astype(np.float32, copy=False)
    finally:
        torch.use_deterministic_algorithms(deterministic)


def predict_many(fit, specs):
    """Yield each spec and prediction, reusing optimizer trajectories."""
    families = defaultdict(list)
    for spec in specs:
        key = tuple(sorted((k, v) for k, v in spec.items() if k not in {"name", "epochs"}))
        families[key].append(spec)
    prep = Preparation(fit)
    for same_run in families.values():
        by_epoch = defaultdict(list)
        for spec in same_run:
            by_epoch[spec["epochs"]].append(spec)
        for epoch, prediction in _run(prep, same_run[0], by_epoch):
            for spec in by_epoch[epoch]:
                yield spec, prediction


def predict(fit, spec):
    """Return one query delta in the original row and gene order."""
    return next(prediction for _, prediction in predict_many(fit, [spec]))
