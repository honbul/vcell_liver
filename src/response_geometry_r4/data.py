"""Unit store access, training-only statistics (cell means, scale, teacher PCA, reliability) and the sealed test door."""
import json, os
import numpy as np
import pandas as pd

from .common import MAN, proc, HCC

FREEZE = os.path.join(MAN, "freeze_manifest.json")


class Store:
    def __init__(self):
        self.units = pd.read_csv(proc("units.tsv"), sep="\t")
        self.groups = pd.read_csv(proc("groups.tsv"), sep="\t")
        fp = np.load(os.path.join(os.environ["TB_PROC"], "ecfp4.npy"))
        self.group_fp = fp[self.groups.rep_drug_idx.to_numpy()].astype(np.float32)        # (n_groups, 2048)
        self.x0 = np.load(proc("unit_x0.npy"))
        self._open = np.load(proc("unit_delta_open.npy"))
        u = self.units
        self.role = u.role.to_numpy()
        self.cell_idx = u.cell_idx.to_numpy()
        self.n_cells = int(self.cell_idx.max()) + 1
        self.cells = u.drop_duplicates("cell_idx").sort_values("cell_idx").cell.to_numpy()
        self.train = np.where(self.role == "train")[0]
        self.val = np.where(self.role == "validation")[0]
        self.test = np.where(self.role == "test")[0]
        assert len(set(u.drug_group[self.train]) & set(u.drug_group[self.val])) == 0
        assert len(set(u.drug_group[self.train]) & set(u.drug_group[self.test])) == 0
        assert len(set(u.drug_group[self.val]) & set(u.drug_group[self.test])) == 0

    def fp(self, idx):
        return self.group_fp[self.units.group_idx.to_numpy()[idx]]

    def delta(self, idx):
        """Measured unit deltas of train / validation units. Test units are refused here."""
        idx = np.asarray(idx)
        assert not (self.role[idx] == "test").any(), "test deltas are sealed (use test_delta after the freeze)"
        return self._open[self.units.open_idx.to_numpy()[idx]]

    def test_delta(self, idx):
        assert os.environ.get("R4_FINAL_EVAL") == "1", "test deltas only inside the final evaluation job"
        assert os.path.exists(FREEZE), "freeze manifest missing: test stays sealed"
        idx = np.asarray(idx)
        assert (self.role[idx] == "test").all()
        t = np.load(proc("unit_delta_test.npy"), mmap_mode="r")
        return np.asarray(t[self.units.test_idx.to_numpy()[idx]])

    def source_mask(self, idx):
        return self.units.is_hcc.to_numpy()[idx] == 0


class TrainStats:
    """Every statistic below is fitted on training-drug units only (fit_idx must be training units)."""

    def __init__(self, S, fit_idx=None, n_axes=64, floor_q=10):
        fit_idx = S.train if fit_idx is None else np.asarray(fit_idx)
        assert (S.role[fit_idx] == "train").all(), "teacher / scale fit on non-training units"
        y = S.delta(fit_idx).astype(np.float64)
        c = S.cell_idx[fit_idx]
        m = np.zeros((S.n_cells, 978)); n = np.bincount(c, minlength=S.n_cells)
        np.add.at(m, c, y)
        self.has_mean = n > 0
        m[self.has_mean] /= n[self.has_mean, None]
        self.cell_mean = m.astype(np.float32)                  # m_c (zero for a cell without training units)
        cen = y - m[c]
        sd = cen.std(0)
        self.scale = np.maximum(sd, np.percentile(sd, floor_q)).astype(np.float32)    # D
        z = cen / self.scale
        mu = z.mean(0)
        _, s, vt = np.linalg.svd(z - mu, full_matrices=False)
        self.U = vt[:n_axes].T.astype(np.float32)              # (978, 64)
        self.pca_mu = mu.astype(np.float32)
        self.explained = float((s[:n_axes] ** 2).sum() / (s ** 2).sum())
        self.fit_idx = fit_idx

    def teacher(self, S, idx, delta=None):
        y = S.delta(idx) if delta is None else delta
        return (((y - self.cell_mean[S.cell_idx[idx]]) / self.scale - self.pca_mu) @ self.U).astype(np.float32)


def split_half_reliability(S, stats):
    """Split-half Pearson of cell-centred delta over training units with >= 2 detection plates."""
    a = np.load(proc("half_delta_a.npy")); b = np.load(proc("half_delta_b.npy"))
    hi = S.units.half_idx.to_numpy()
    rep = np.where(hi >= 0)[0]
    assert (S.role[rep] == "train").all()
    m = stats.cell_mean[S.cell_idx[rep]]
    ca, cb = a[hi[rep]] - m, b[hi[rep]] - m
    ca -= ca.mean(1, keepdims=True); cb -= cb.mean(1, keepdims=True)
    r = (ca * cb).sum(1) / np.sqrt((ca ** 2).sum(1) * (cb ** 2).sum(1) + 1e-12)
    return rep, r, a[hi[rep]], b[hi[rep]]


def reliability_weights(S, stats, min_cell=10):
    """r in [0, 1] for every training unit; unknown units get the cell median (global median if < min_cell replicated units)."""
    rep, r, _, _ = split_half_reliability(S, stats)
    rr = np.full(len(S.units), np.nan, np.float32)
    rr[rep] = np.clip(r, 0, 1)
    known = np.zeros(len(S.units), bool); known[rep] = True
    glob = float(np.median(rr[rep]))
    med = {}
    for c in range(S.n_cells):
        sel = rep[S.cell_idx[rep] == c]
        med[c] = float(np.median(rr[sel])) if len(sel) >= min_cell else glob
    tr = S.train
    fill = np.array([med[c] for c in S.cell_idx[tr]], np.float32)
    rr[tr] = np.where(known[tr], rr[tr], fill)
    return rr, known, glob
