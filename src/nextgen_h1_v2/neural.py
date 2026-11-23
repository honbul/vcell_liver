"""Family C permutation-invariant neural-process adaptation."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .boost import region
from .data import BUDGETS, within_drug
from .kernels import base

from nextgen_h1_anchor.metrics import Retrieval


class SetProcess(nn.Module):
    def __init__(self, embed, hidden, heads, dropout):
        super().__init__()
        self.drug = nn.Linear(embed, hidden)
        self.anchor_value = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.null = nn.Parameter(torch.zeros(1, 1, hidden))
        self.self_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden)
        self.cross_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.rna = nn.Sequential(nn.Linear(16, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gate = nn.Linear(2 * hidden, hidden)
        self.film = nn.Linear(hidden, 2 * hidden)
        self.decode = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.skip = nn.Linear(hidden, hidden, bias=False)

    def forward(self, drug, pc, packet, observed, order, dual):
        compound = self.drug(drug)
        anchors = compound[order][None] + self.anchor_value(packet)
        anchors = torch.cat((self.null.expand(len(pc), -1, -1), anchors), dim=1)
        present = F.pad(observed, (1, 0), value=True)
        h, _ = self.self_attention(anchors, anchors, anchors, key_padding_mask=~present, need_weights=False)
        h = self.self_norm(h + anchors)
        func = (h * present[:, :, None]).sum(1) / present.sum(1, keepdim=True)
        expr = self.rna(pc) if dual else torch.zeros_like(func)
        gate = torch.sigmoid(self.gate(torch.cat((func, expr), dim=1)))
        fused = func + gate * expr
        gain, shift = self.film(fused).chunk(2, dim=1)
        query = compound[None].expand(len(pc), -1, -1)
        observed_state, _ = self.cross_attention(query, h, h, key_padding_mask=~present, need_weights=False)
        hidden = (query + observed_state) * (1 + gain[:, None].tanh()) + shift[:, None]
        bilinear = (query * self.skip(fused)[:, None]).sum(-1) / math.sqrt(query.shape[-1])
        return self.decode(hidden).squeeze(-1) + bilinear


def tensors(ctx, packet, order, dual, prior, pc, device, dtype=torch.float32):
    good = np.isfinite(packet)
    standardized = np.nan_to_num((packet - ctx.mean[order]) / ctx.sd[order])
    residual = np.nan_to_num((packet - prior[:, order]) / ctx.sd[order])
    return (torch.as_tensor(ctx.embedding, dtype=dtype, device=device),
            torch.as_tensor(np.clip(pc, -8, 8), dtype=dtype, device=device),
            torch.as_tensor(np.stack((standardized, residual), axis=-1), dtype=dtype, device=device),
            torch.as_tensor(good, device=device), torch.as_tensor(order, device=device), dual)


def joint_loss(pred, truth, native_pred, native_truth, valid, cfg):
    denom = valid.sum().clamp_min(1)
    mse = F.smooth_l1_loss(pred[valid], truth[valid], beta=.5, reduction="sum") / denom
    n = valid.sum(0).clamp_min(1)
    ty = torch.where(valid, native_truth, 0.)
    py = torch.where(valid, native_pred, 0.)
    y_centered = (ty - ty.sum(0) / n) * valid
    p_centered = (py - py.sum(0) / n) * valid
    w = (y_centered ** 2).sum(0)
    rho = (y_centered * p_centered).sum(0) / (w * (p_centered ** 2).sum(0) + 1e-7).sqrt()
    rank_loss = pred.new_zeros(())
    if cfg["rank_weight"]:
        y_order = torch.argsort(torch.argsort(ty.masked_fill(~valid, -float("inf")), dim=1), dim=1)
        t = y_order.masked_fill(~valid, 0).to(pred.dtype)
        gain = t / max(1, truth.shape[1] - 1)
        if cfg["rank_objective"] == "listnet":
            distribution = torch.softmax((6 * gain).masked_fill(~valid, -1e4), dim=1)
            log_probability = torch.log_softmax(native_pred.masked_fill(~valid, -1e4), dim=1)
            rank_loss = -(distribution * log_probability).sum(1).mean()
        else:
            top = (gain >= gain.masked_fill(~valid, -1.).topk(5, dim=1).values[:, -1:]) & valid
            links = top[:, :, None] & valid[:, None, :] & (ty[:, :, None] > ty[:, None, :])
            penalty = F.softplus(native_pred[:, None, :] - native_pred[:, :, None])
            rank_loss = (penalty * links).sum() / links.sum().clamp_min(1)
    return mse - cfg["rho_weight"] * rho.mean() + cfg["rank_weight"] * rank_loss


@dataclass
class NeuralModel:
    ctx: object
    order: np.ndarray
    dual: bool
    config: dict
    state: dict
    curves: list
    best_epoch: int
    _net: object = None

    def predict(self, packet, k, *, scramble=None, training=False, device="cpu"):
        if self._net is None or next(self._net.parameters()).device.type != torch.device(device).type:
            self._net = SetProcess(self.ctx.embedding.shape[1], self.config["hidden"],
                                   self.config["heads"], self.config["dropout"]).to(device)
            self._net.load_state_dict(self.state)
            if torch.device(device).type == "cpu":
                self._net.double()
            self._net.eval()
        prior = base(self.ctx, self.dual, training)
        pc = self.ctx.pc_fit if training else self.ctx.pc_valid
        if scramble is not None and self.dual:
            prior, pc = prior[scramble], pc[scramble]
        with torch.no_grad():
            output = self._net(*tensors(self.ctx, packet, self.order[:k], self.dual, prior, pc, device,
                                       dtype=next(self._net.parameters()).dtype))
        return prior + output.cpu().numpy() * self.ctx.sd


def stats(y, p):
    if len(y) < 2:
        return 0., 0.
    ndcg = float(np.nanmean(Retrieval(y)(p)[:, 0]))
    return ndcg, within_drug(y, p)


def fit_neural(ctx, order, dual, cfg, device, *, epochs=None):
    torch.manual_seed(cfg.get("seed", 90217))
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg.get("seed", 90217))
    net = SetProcess(ctx.embedding.shape[1], cfg["hidden"], cfg["heads"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["decay"])
    count = cfg["epochs"] if epochs is None else epochs
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, count, eta_min=.05 * cfg["lr"])
    priors = base(ctx, dual, True)
    truth = (ctx.train - priors) / ctx.sd
    scale = max(float(np.median(ctx.sd)), 1e-8)
    data = []
    for k in BUDGETS:
        codes, q = order[:k], region(order, k, len(ctx.mean))
        drug, pcs, packet, observed, drug_codes, _ = tensors(
            ctx, ctx.train[:, codes], codes, dual, priors, ctx.pc_fit, device)
        data.append((drug, pcs, packet, observed, drug_codes, q,
                     torch.as_tensor(np.nan_to_num(truth[:, q]), device=device, dtype=torch.float32),
                     torch.as_tensor(np.isfinite(truth[:, q]), device=device),
                     torch.as_tensor(ctx.sd[q] / scale, device=device, dtype=torch.float32),
                     torch.as_tensor(priors[:, q] / scale, device=device, dtype=torch.float32)))
    curves, epoch_best, best_score, best, stale = [], 0, -np.inf, None, 0
    for epoch in range(1, count + 1):
        net.train()
        groups = rng.permutation(len(ctx.train))
        for batch in np.array_split(groups, max(1, math.ceil(len(groups) / 64))):
            d, pc, packet, obs, a, q, yy, mask, drug_scale, intercept = data[
                int(rng.choice(len(data), p=[.05, .08, .12, .48, .17, .10]))]
            i = torch.as_tensor(batch, device=device)
            observed = obs[i]
            values = packet[i]
            if cfg.get("noise", 0.):
                jitter = torch.randn_like(values[:, :, :1]) * cfg["noise"]
                values = values + jitter * observed[:, :, None]
            y, b = yy[i], intercept[i]
            pred = net(d, pc[i], values, observed, a, dual)[:, q]
            loss = joint_loss(pred, y, pred * drug_scale + b, y * drug_scale + b, mask[i], cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.)
            opt.step()
        schedule.step()
        if epoch % 5 and epoch != count:
            continue
        if ctx.valid is None:
            continue
        state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        proxy = NeuralModel(ctx, order, dual, cfg, state, [], epoch, net.eval())
        a, q = order[:5], region(order, 5, len(ctx.mean))
        train_pred = proxy.predict(ctx.train[:, a], 5, training=True, device=device)
        valid_pred = proxy.predict(ctx.valid[:, a], 5, device=device)
        def complete(y):
            return np.isfinite(y[:, order[:8]]).all(1) & (np.isfinite(y[:, q]).sum(1) >= 15)
        tr, va = complete(ctx.train), complete(ctx.valid)
        tn, trho = stats(ctx.train[tr][:, q], train_pred[tr][:, q])
        vn, vrho = stats(ctx.valid[va][:, q], valid_pred[va][:, q])
        curves.append(dict(epoch=epoch, train_ndcg=tn, val_ndcg=vn, train_rho=trho, val_rho=vrho))
        score = vn + (.12 if ctx.study == "LICOB" else .06) * vrho
        if score > best_score + 1e-6:
            best_score, epoch_best, best, stale = score, epoch, state, 0
        else:
            stale += 1
        if epoch >= min(50, count) and stale >= 6:
            break
    if best is None:
        best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        epoch_best = count
    return NeuralModel(ctx, order, dual, cfg, best, curves, epoch_best)
