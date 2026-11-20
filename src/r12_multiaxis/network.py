"""Matched residual heads, exact pool losses, and dual in-development checkpoints."""

import math
import time

import numpy as np
import torch
from torch import nn

from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, POOL, group_weights, weights_for
from src.r11_chemical_probe.network import Concat


class ResidualHead(nn.Module):
    def __init__(self, dims, genes, head, dropout, projection):
        super().__init__()
        n_drug, *n_control = dims
        self.drug = nn.Linear(n_drug, 128)
        self.context = nn.Linear(n_control[0], 128) if n_control else None
        self.head = head
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.GELU()
        if head == "film":
            self.combine = nn.Linear(128, 256)
        else:
            self.combine = nn.Linear(256 if self.context is not None else 128, 128)
        if head == "bilinear":
            self.drug_factor = nn.Linear(128, 16, bias=False)
            self.context_factor = nn.Linear(128, 16, bias=False)
            self.interaction = nn.Linear(16, 128, bias=False)
        if projection is None:
            self.decoder = nn.Linear(128, genes)
        else:
            vectors, center = projection
            self.decoder = nn.Linear(128, len(vectors))
            self.register_buffer("vectors", torch.from_numpy(vectors))
            self.register_buffer("center", torch.from_numpy(center))
        if head == "film":
            nn.init.zeros_(self.combine.weight)
            nn.init.zeros_(self.combine.bias)

    def encode(self, control):
        return self.activate(self.context(control))

    def forward(self, drug, control=None):
        d = self.activate(self.drug(drug))
        if control is None:
            z = self.combine(d)
        else:
            c = self.encode(control)
            if self.head == "film":
                gamma, beta = self.combine(c).chunk(2, dim=1)
                z = (1 + gamma) * d + beta
            else:
                z = self.combine(torch.cat((d, c), dim=1))
                if self.head == "bilinear":
                    z = z + self.interaction(self.drug_factor(d) * self.context_factor(c))
        out = self.decoder(self.dropout(self.activate(z)))
        if hasattr(self, "vectors"):
            out = out @ self.vectors + self.center
        return out


def smooth_rmse(error):
    return torch.sqrt(error.square().mean(dim=1) + 1e-6) - 0.001


def center(error, pool, weight, n):
    totals = error.new_zeros((n, error.shape[1]))
    totals.index_add_(0, pool, error * weight[:, None])
    return error - totals[pool]


def row_loss(error, kind, pool, w_center, n_pools, gene_weight=None):
    square = error.square()
    mse = square.mean(dim=1)
    if kind == "L1":
        return smooth_rmse(error)
    if kind == "L2":
        return smooth_rmse(center(error, pool, w_center, n_pools)) + 0.1 * mse
    if kind == "L3":
        return torch.where(error.abs() <= 1.0, square * 0.5, error.abs() - 0.5).mean(dim=1)
    if kind == "L4":
        return (square * gene_weight).mean(dim=1)
    return mse


def vicreg(a, b):
    inv = (a - b).square().mean()
    centered = [z - z.mean(dim=0) for z in (a, b)]
    std = [torch.sqrt(z.var(dim=0, correction=1) + 0.0001) for z in (a, b)]
    var = sum(torch.relu(1 - s).mean() for s in std) / 2
    cov = [z.T @ z / (len(z) - 1) for z in centered]
    corr = sum((m.square().sum() - m.diagonal().square().sum()) / m.shape[0] for m in cov)
    return 25 * inv + 25 * var + corr


def pool_weights(rows):
    pools = rows.groupby(POOL, sort=False).ngroup().to_numpy()
    weight = weights_for(rows.assign(_pool=pools), ("_pool", GROUP, ENTITY, KEY[0]))
    totals = np.bincount(pools, weights=weight)
    return pools, weight / totals[pools]


def packs(rows, batch_rows, pack_count=None):
    pool, weight = pool_weights(rows)
    members = [np.flatnonzero(pool == p) for p in range(pool.max() + 1)]
    n = min(len(members), pack_count or math.ceil(len(rows) / batch_rows))
    grouped = [[] for _ in range(n)]
    load = np.zeros(n, dtype=int)
    for idx in sorted(members, key=len, reverse=True):
        k = int(np.argmin(load))
        grouped[k].append(idx)
        load[k] += len(idx)
    return [(ix := np.concatenate(group), np.repeat(np.arange(len(group)), [len(i) for i in group]),
             weight[ix], len(group)) for group in grouped]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def affine_macs(model):
    names = {"projector"}
    return sum(layer.in_features * layer.out_features
               for name, layer in model.named_modules() if isinstance(layer, nn.Linear) and name not in names)


def fit_network(fit, design, arch, loss, lr, seed, device, limit, patience=True, vic=0.0,
                replacement=None, pack_count=None):
    start = time.monotonic()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(seed)
    cfg = fit.cfg
    dims = [v.shape[1] for v in design.train]
    n_genes = fit.residual.shape[1]
    if arch == "full":
        model = Concat(dims[0], n_genes, 128)
    else:
        model = ResidualHead(dims, n_genes, arch, cfg["dropout"], design.response_head)
    if loss == "L6":
        model.projector = nn.Linear(128, 16)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])

    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)

    x = tuple(tensor(v) for v in design.train)
    q = tuple(tensor(v) for v in design.query)
    target = tensor(fit.residual / fit.scale)
    offset = tensor(fit.offset)
    weight = tensor(fit.weight)
    schedule = [(tensor(idx, torch.long), tensor(pool, torch.long), tensor(w), n)
                for idx, pool, w, n in packs(fit.train.rows, cfg["batch_rows"], pack_count)]
    w_total = float(fit.weight.sum())
    factor = len(schedule) / w_total
    gene_weight = tensor(fit.noise_weights) if loss == "L4" else None
    two_views = loss in {"L5", "L6"}
    views = None if design.views is None else tuple(tensor(v) for v in design.views)
    view_units = None if design.unit_views is None else tuple(tensor(v) for v in design.unit_views)
    if two_views:
        if views is not None:
            n_units = len(view_units[0])
        else:
            first = ~fit.train.rows.duplicated(["control_well_id", "cell_line_id"]).to_numpy()
            cell_units = x[1][first]
            n_units = len(cell_units)
        if n_units < 2:
            raise ValueError("Two-view regularization requires distinct DMSO units")

    if patience:
        delta = tensor(fit.query.delta)
        qw = tensor(group_weights(fit.query.rows))
        qw = qw / qw.sum()
        pool, pweight = pool_weights(fit.query.rows)
        qp = tensor(pool, torch.long)
        qpw = tensor(pweight)
        qn = int(pool.max()) + 1
    best = {"abs": (float("inf"), 0, None, None), "contrast": (float("inf"), 0, None, None)}
    curves = []
    for epoch in range(1, limit + 1):
        model.train()
        tasks, penalties = [], []
        grad_task = None
        grad_aux = None
        for j, index in enumerate(rng.permutation(len(schedule))):
            idx, p, w_center, n = schedule[index]
            optim.zero_grad(set_to_none=True)
            if two_views:
                c1, c2 = ((v[idx] for v in views) if views is not None else
                          (nn.functional.dropout(x[1][idx], p=0.1, training=True) for _ in range(2)))
                e1 = model(x[0][idx], c1) - target[idx]
                e2 = model(x[0][idx], c2) - target[idx]
                value = 0.5 * (e1.square().mean(dim=1) + e2.square().mean(dim=1))
            else:
                error = model(*(a[idx] for a in x)) - target[idx]
                value = row_loss(error, loss, p, w_center, n, gene_weight)
            task = factor * (weight[idx] * value).sum()
            total = task
            tasks.append(task.detach())
            if loss == "L6":
                chosen = rng.choice(n_units, min(64, n_units), replace=False)
                a, b = ((v[chosen] for v in view_units) if view_units is not None else
                        (nn.functional.dropout(cell_units[chosen], p=0.1, training=True) for _ in range(2)))
                aux = vic * factor * weight[idx].sum() * vicreg(
                    model.projector(model.encode(a)), model.projector(model.encode(b)))
                total = total + aux
                penalties.append(aux.detach())
                if j == 0:
                    encoder = model.context.weight
                    gt = torch.autograd.grad(task, encoder, retain_graph=True)[0]
                    ga = torch.autograd.grad(aux, encoder, retain_graph=True)[0]
                    grad_task, grad_aux = float(gt.norm().item()), float(ga.norm().item())
            total.backward()
            optim.step()
        stats = dict(epoch=epoch, task=float(torch.stack(tasks).mean().item()))
        if penalties:
            stats.update(regularizer=float(torch.stack(penalties).mean().item()),
                         context_task_gradient=grad_task, context_regularizer_gradient=grad_aux)
        if patience:
            model.eval()
            with torch.inference_mode():
                pred = offset + fit.scale * model(*q)
                err = pred - delta
                absolute = torch.dot(qw, err.square().mean(dim=1).sqrt())
                contrast = torch.dot(qw, center(err, qp, qpw, qn).square().mean(dim=1).sqrt())
                stats.update(abs=float(absolute.item()), contrast=float(contrast.item()))
                for key in ("abs", "contrast"):
                    if stats[key] < best[key][0] - 1e-9:
                        best[key] = (stats[key], epoch, pred.cpu().numpy().copy(),
                                     {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        curves.append(stats)
        if patience and epoch >= cfg["min_epochs"] and epoch - max(b[1] for b in best.values()) >= cfg["patience"]:
            break

    diagnostics = {}
    if not patience:
        model.eval()
        with torch.inference_mode():
            base = offset + fit.scale * model(*q)
            last = (0.0, epoch, base.cpu().numpy(), {k: v.detach().cpu() for k, v in model.state_dict().items()})
            best = {"abs": last, "contrast": last}
            for name, design in (replacement or {}).items():
                diagnostics[name] = (offset + fit.scale * model(*(tensor(v) for v in design.query))).cpu().numpy()
            if loss == "L6":
                z = model.projector(model.encode(tensor(design_unit_inputs(fit, x))))
                sv = torch.linalg.svdvals(z - z.mean(dim=0)).cpu().numpy()
                p = sv / max(sv.sum(), 1e-12)
                diagnostics["representation"] = {
                    "mean_variance": float(z.var(dim=0).mean().item()),
                    "effective_rank": float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-12))))),
                }

    runtime = dict(device=torch.cuda.get_device_name() if device == "cuda" else "CPU",
                   seconds=time.monotonic() - start, epochs=epoch, updates=epoch * len(schedule),
                   packs=len(schedule), params=parameter_count(model), dense_macs=affine_macs(model),
                   peak_gpu_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                   seed=seed, learning_rate=lr, scale=fit.scale)
    return best, runtime, curves, diagnostics


def design_unit_inputs(fit, inputs):
    first = ~fit.train.rows.duplicated(["control_well_id", "cell_line_id"]).to_numpy()
    return inputs[1][first]
