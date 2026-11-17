"""The fixed metric, the shared non-negative restoration and the drug-block aggregation.

Energy distance.  D_E(P, Q) = 2 E|X-Y| - E|X-X'| - E|Y-Y'| with Euclidean norm. The two
within-sample terms are U-statistics (i != j, divided by n(n-1)); the cross term is the plain mean
over all n*m pairs. The estimator is unbiased, so it can be slightly negative for two samples of
one distribution. Negative values are kept: nothing is clipped anywhere.

Primary space: the 2,000 panel genes divided by the training-only pooled within-well SD.
Sensitivity space: a 50-component PCA fitted on training cells in that standardised space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def energy_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    n, m = A.shape[0], B.shape[0]
    dab = torch.cdist(A, B).mean()
    daa = torch.cdist(A, A).sum() / (n * (n - 1))
    dbb = torch.cdist(B, B).sum() / (m * (m - 1))
    return float(2 * dab - daa - dbb)


def restore_nonneg(V: torch.Tensor, mu: torch.Tensor, iters: int = 50) -> torch.Tensor:
    """Shift each gene by a_g and clamp at zero so that mean_j max(v_jg + a_g, 0) = max(mu_g, 0).

    One monotone 1-D search per gene (bisection, all genes at once). A gene whose target mean is
    zero becomes all zeros. This is the single restoration rule applied to every mean-anchored
    method; it reads the predicted mean `mu` only, never an observed treated statistic.
    """
    target = mu.clamp(min=0)
    lo = -V.max(0).values                    # everything clamps to zero
    hi = target - V.min(0).values            # nothing clamps, mean >= target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = (V + mid).clamp(min=0).mean(0) - target
        hi = torch.where(f > 0, mid, hi)
        lo = torch.where(f > 0, lo, mid)
    out = (V + 0.5 * (lo + hi)).clamp(min=0)
    return torch.where(target > 0, out, torch.zeros_like(out))


def restore_zero_masked(V: torch.Tensor, mu: torch.Tensor, pi0: torch.Tensor, iters: int = 50) -> torch.Tensor:
    """Restoration v2 (cycle 1): take the zero fraction from the control population.

    Per gene, the round(pi0 * n) smallest generated values are set to zero (pi0 = zero fraction of
    the same plate's control cells of that line, an allowed input), and the remaining cells get
    the v1 rule: one shift a_g with mean_j max(v_jg + a_g, 0) = max(mu_g, 0) over all n cells.
    When mu_g > 0 at least one cell is kept. No treated statistic is read.
    """
    n = V.shape[0]
    target = mu.clamp(min=0)
    k = torch.round(pi0 * n).long().clamp(min=0, max=n - 1)
    rank = V.argsort(0).argsort(0)
    keep = (rank >= k[None]).to(V.dtype)
    nk = keep.sum(0)
    lo = -V.max(0).values
    hi = target * n / nk - V.min(0).values
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = ((V + mid).clamp(min=0) * keep).mean(0) - target
        hi = torch.where(f > 0, mid, hi)
        lo = torch.where(f > 0, lo, mid)
    out = (V + 0.5 * (lo + hi)).clamp(min=0) * keep
    return torch.where(target > 0, out, torch.zeros_like(out))


def shift_to_mean(V: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return V - V.mean(0, keepdim=True) + mu


class Scorer:
    """Scores one generated population against one observed population, in fixed spaces."""

    def __init__(self, sd_within, pca_mean=None, pca_comp=None, kmeans=None, device="cpu"):
        self.dev = device
        self.sd = torch.as_tensor(sd_within, dtype=torch.float32, device=device)
        self.pm = None if pca_mean is None else torch.as_tensor(pca_mean, dtype=torch.float32, device=device)
        self.pc = None if pca_comp is None else torch.as_tensor(pca_comp, dtype=torch.float32, device=device)
        self.kmeans = kmeans or {}             # line -> [k, 50] centroids in PCA space

    def t(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)

    def pca(self, Z):
        return (Z - self.pm) @ self.pc.T

    def ed(self, gen, obs) -> float:
        return energy_distance(gen / self.sd, obs / self.sd)

    def score(self, gen, obs, line=None, full=True) -> dict:
        """gen, obs: tensors [n, G] in log1p(CP10k) units, same n."""
        Zg, Zo = gen / self.sd, obs / self.sd
        out = dict(ed=energy_distance(Zg, Zo))
        if not full:
            return out
        mg, mo = gen.mean(0), obs.mean(0)
        out["ed_shape"] = energy_distance(Zg - Zg.mean(0), Zo - Zo.mean(0))
        out["mean_rmse"] = float(((mg - mo) ** 2).mean().sqrt())
        vg, vo = Zg.var(0), Zo.var(0)
        out["var_ratio"] = float(vg.sum() / vo.sum())
        out["var_logabs"] = float((torch.log((vg + 0.05) / (vo + 0.05))).abs().mean())
        zg, zo = (gen <= 0).float().mean(0), (obs <= 0).float().mean(0)
        out["zero_frac_gen"] = float(zg.mean()); out["zero_frac_obs"] = float(zo.mean())
        out["zero_frac_mae"] = float((zg - zo).abs().mean())
        out["neg_frac"] = float((gen < 0).float().mean())
        if self.pc is not None:
            Pg, Po = self.pca(Zg), self.pca(Zo)
            out["ed_pca50"] = energy_distance(Pg, Po)
            Cg, Co = torch.cov(Pg.T), torch.cov(Po.T)
            out["cov_relfro"] = float(torch.linalg.norm(Cg - Co) / torch.linalg.norm(Co))
            if line is not None and line in self.kmeans:
                C = self.kmeans[line]
                k = C.shape[0]
                pg = torch.bincount(torch.cdist(Pg, C).argmin(1), minlength=k).float() / len(Pg)
                po = torch.bincount(torch.cdist(Po, C).argmin(1), minlength=k).float() / len(Po)
                out["cluster_tv"] = float(0.5 * (pg - po).abs().sum())
        return out


def fit_pca_kmeans(d, train_wells: pd.DataFrame, sd_within, n_comp=50, k=4, per_well=8, seed=0):
    """Training-only PCA (standardised space) and per-line k-means states in that PCA space."""
    from sklearn.cluster import KMeans
    rows, lines = [], []
    for r in train_wells.itertuples():
        x = np.asarray(d.X[r.start:min(r.end, r.start + per_well)])
        rows.append(x); lines.append(np.full(len(x), r.line))
    for (p, l), sl in d.ctrl_slices.items():
        for s, e in sl:
            x = np.asarray(d.X[s:min(e, s + 4 * per_well)])
            rows.append(x); lines.append(np.full(len(x), l))
    Z = np.concatenate(rows) / sd_within
    lines = np.concatenate(lines)
    mean = Z.mean(0)
    Zt = torch.as_tensor(Z - mean)
    _, _, V = torch.pca_lowrank(Zt, q=n_comp, center=False, niter=6)
    comp = V.T.numpy().astype(np.float32)
    P = (Z - mean) @ comp.T
    km = {}
    for l in np.unique(lines):
        m = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(P[lines == l])
        km[int(l)] = m.cluster_centers_.astype(np.float32)
    return mean.astype(np.float32), comp, km


# ---------------------------------------------------------------- aggregation and uncertainty
def to_condition(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Well-first: average wells inside a (line, drug) condition."""
    return df.groupby(["line", "drug"], as_index=False)[list(cols)].mean()


def drug_table(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Wells -> condition -> drug (cell lines weighted equally inside a drug)."""
    return to_condition(df, cols).groupby("drug")[list(cols)].mean()


def drug_block_bootstrap(dt: pd.DataFrame, fn=None, B: int = 2000, seed: int = 20261117):
    """Resample drug groups (all their cell lines together). `dt` is indexed by drug.

    `fn` maps a resampled frame to a scalar; default is the equal-drug mean of the single column.
    Cells and wells are never the resampling unit.
    """
    fn = fn or (lambda t: float(t.iloc[:, 0].mean()))
    rng = np.random.default_rng(seed)
    n = len(dt)
    est = fn(dt)
    bs = np.array([fn(dt.iloc[rng.integers(0, n, n)]) for _ in range(B)])
    return dict(est=est, lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)), n_drugs=int(n))
