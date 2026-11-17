"""Public TranSiGen architecture under the R2 label budget: as-is, + support mean-offset, full fine-tuning.

The prior study's adapter around the pinned upstream model class is reused unchanged. Fine-tuning sees the K support
drug groups only; there is no target validation set, so learning rate, BatchNorm handling and the number of updates
are fixed beforehand from source virtual targets.
"""
import copy, json, os
import numpy as np
import pandas as pd
import torch

from .common import B_MAN, sha256
from .build_units import rows_to_units
from src.transigen_adaptation_b1 import tsg_adapter as T

WEIGHT_DECAY = 1e-5


class RowAccess:
    """Plate rows of one target cell. Treated profiles are handed out for support groups only."""

    def __init__(self, rows, x0, x1, fp_rows, group_names):
        self.rows = rows.reset_index(drop=True)        # cell, pert_id, drug_group, det_plate
        self.x0, self._x1, self.fp = x0, x1, fp_rows
        self.group_names = group_names                 # group index -> drug_group string

    def select(self, g_idx):
        names = self.group_names[np.asarray(g_idx)]
        m = self.rows.drug_group.isin(set(names)).to_numpy()
        assert set(self.rows.drug_group[m]) == set(names), "an episode drug group has no plate row"
        return np.where(m)[0], names

    def inputs(self, g_idx):
        i, names = self.select(g_idx)
        return self.rows.iloc[i], self.x0[i], self.fp[i], names

    def support_x1(self, ep, K):
        i, _ = self.select(ep.g_s[:K])                 # never a query group
        return self._x1[i]


def source_row_access(B, src, cell):
    """Row access for a source virtual target (prior-study arrays, degenerate rows removed)."""
    n = B.n_src
    rows = B.rows.iloc[:n]
    ex = pd.read_csv(os.path.join(B_MAN, "source_rows_excluded_degenerate.tsv"), sep="\t").row_id.to_numpy()
    m = (rows.cell.to_numpy() == cell)
    m[ex] = False
    idx = np.where(m)[0]
    return RowAccess(rows.iloc[idx], np.asarray(B.x0[idx]), np.asarray(B.source_x1()[idx]), B.fp[rows.drug_idx.to_numpy()[idx]].astype(np.float32),
                     src.groups.drug_group.to_numpy())


def checkpoints():
    sel = json.load(open(os.path.join(B_MAN, "source_selection.json")))["checkpoints"]
    for e in sel.values():
        assert sha256(e["checkpoint"]) == e["checkpoint_sha256"], "prior-study source checkpoint changed"
    return {int(s): e for s, e in sel.items()}


def units_from_rows(rows, delta_rows, x0_rows, names):
    tab, ud, _ = rows_to_units(rows, delta_rows, x0_rows)
    pos = {g: i for i, g in enumerate(tab.drug_group)}
    return ud[[pos[g] for g in names]]


@torch.no_grad()
def predict_units(model, ra, g_idx, dev):
    rows, x0, fp, names = ra.inputs(g_idx)
    x0t = torch.tensor(x0, device=dev)
    p = T.predict_x1_mean_path(model, x0t, torch.tensor(fp, device=dev))
    return torch.tensor(units_from_rows(rows, (p - x0t).cpu().numpy(), x0, names), device=dev)


def set_bn(model, mode):
    model.train()
    if mode == "frozen_running_stats":
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()                               # running statistics fixed; affine weights still trained
    else:
        assert mode == "train_mode"


def train_step(model, opt, x0, x1, fp, bn):
    """Same calls as the upstream training step; only the BatchNorm mode can differ."""
    set_bn(model, bn)
    opt.zero_grad()
    x1_rec, mu1, logvar1, x2_pert, mu_pred, logvar_pred, _ = model.forward(x0, fp)
    z2, mu2, logvar2 = model.encode_x2(x1)
    x2_rec = model.decode_x2(z2)
    loss = model.loss(x0, x1_rec, mu1, logvar1, x1, x2_rec, mu2, logvar2, x2_pert, mu_pred, logvar_pred)[0]
    loss.backward()
    opt.step()
    return loss.item()


def fine_tune(base_model, ep, K, ra, lr, bn, update_grid, seed, dev):
    """Full fine-tuning on the K support groups. Returns {n_updates: query unit prediction}; diverged steps are missing."""
    model = copy.deepcopy(base_model)
    T.seed_all(seed)
    rows, x0, fp, _ = ra.inputs(ep.g_s[:K])
    x0t, fpt = torch.tensor(x0, device=dev), torch.tensor(fp, device=dev)
    x1t = torch.tensor(ra.support_x1(ep, K), device=dev)
    n = len(rows)
    bs = min(32, n)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out, step, perm, pos, status = {}, 0, None, 0, "ok"
    grid = sorted(update_grid)
    while step < grid[-1]:
        if perm is None or pos >= n:
            perm, pos = torch.randperm(n, generator=gen).to(dev), 0
        b = perm[pos:pos + bs]
        pos += bs
        if len(b) < 2:
            continue
        l = train_step(model, opt, x0t[b], x1t[b], fpt[b], bn)
        step += 1
        if not np.isfinite(l):
            status = f"non_finite_loss_at_{step}"
            break
        if step in grid:
            p = predict_units(model, ra, ep.g_q, dev)
            if not torch.isfinite(p).all():
                status = f"non_finite_prediction_at_{step}"
                break
            out[step] = p
    return out, status, dict(n_support_rows=int(n), batch_size=int(bs))
