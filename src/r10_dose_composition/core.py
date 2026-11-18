"""Shared data access, fixed-atlas re-reading, and distribution metrics for R10.

Two functions are adapted from the R3 study (read-only reuse, not merged):
  origin branch : research/hcc-anchored-distribution-r3
  origin commit : 0ace11c4cc84f587d2aa791a5afaf872fbab59bf (last commit touching the file; branch tip 90abe72)
  origin file   : src/anchored_distribution_r3/metrics.py
  file sha256   : dd6475e5465fb5dfa19a854f0bf043e807de7f97708ca55619ca230d2216422b
  - energy_distance: generalised here to weighted point clouds made of independent groups
    (`cloud_stat`). For one uniform group it is exactly the R3 estimator (U-statistic within a
    sample, plain mean across samples; nothing clipped, may be slightly negative).
  - drug_block_bootstrap: copied (paired use: the column is a per-drug difference).

A predicted population is a `Cloud`: points X [N, d] in the fixed train-only PCA space,
weights w (sum 1) and group labels g. Points of one group are treated as one i.i.d. sample
(weighted U-statistic inside a group, plain weighted mean across groups), so an empirical
mixture of two endpoint wells is scored as the mixture of two samples, not as one sample with
duplicated points.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

LINES = ("A549", "K562", "MCF7")


# ------------------------------------------------------------------ clouds and metrics
@dataclass
class Cloud:
    X: np.ndarray
    w: np.ndarray
    g: np.ndarray
    declared_pi: np.ndarray | None = None        # head / interpolation composition (state-wise methods)
    declared_means: np.ndarray | None = None     # [K, d] declared state means
    meta: dict = field(default_factory=dict)

    @staticmethod
    def uniform(X, group=0):
        X = np.asarray(X, dtype=np.float64)
        return Cloud(X, np.full(len(X), 1.0 / len(X)), np.full(len(X), group))

    def mean(self):
        return self.w @ self.X


def _k(D, kind, bw):
    return D if kind == "ed" else torch.exp(-D ** 2 / (2 * bw ** 2))


def _block(A, wa, B, wb, same, kind, bw):
    K = _k(torch.cdist(A, B), kind, bw)
    W = wa[:, None] * wb[None, :]
    if same:
        W = W.clone()
        W.fill_diagonal_(0)
    s = W.sum()
    return (W * K).sum() / s if s > 0 else torch.zeros((), dtype=A.dtype)


def _self(X, w, g, kind, bw):
    gs = np.unique(g)
    if len(gs) == 1:
        return _block(X, w, X, w, True, kind, bw)
    out = torch.zeros((), dtype=X.dtype)
    tot = w.sum()
    for a in gs:
        ma = torch.as_tensor(g == a)
        for b in gs:
            mb = torch.as_tensor(g == b)
            if b < a:
                continue
            v = _block(X[ma], w[ma], X[mb], w[mb], a == b, kind, bw)
            f = (w[ma].sum() * w[mb].sum()) / tot ** 2
            out = out + (1 if a == b else 2) * f * v
    return out


def cloud_stat(P: Cloud, Q: Cloud, kind="ed", bw=1.0, wP=None, wQ=None) -> float:
    """Energy distance (kind='ed') or squared MMD with an RBF kernel (kind='mmd').

    wP / wQ optionally replace the weights (used for state-conditional measures)."""
    X = torch.as_tensor(P.X, dtype=torch.float64)
    Y = torch.as_tensor(Q.X, dtype=torch.float64)
    a = torch.as_tensor(P.w if wP is None else wP, dtype=torch.float64)
    b = torch.as_tensor(Q.w if wQ is None else wQ, dtype=torch.float64)
    a, b = a / a.sum(), b / b.sum()
    sxx = _self(X, a, P.g, kind, bw)
    syy = _self(Y, b, Q.g, kind, bw)
    sxy = _block(X, a, Y, b, False, kind, bw)
    if kind == "ed":
        return float(2 * sxy - sxx - syy)
    return float(sxx + syy - 2 * sxy)


def energy_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    """Verbatim R3 estimator (kept for the equivalence check in the B1 positive controls)."""
    n, m = A.shape[0], B.shape[0]
    dab = torch.cdist(A, B).mean()
    daa = torch.cdist(A, A).sum() / (n * (n - 1))
    dbb = torch.cdist(B, B).sum() / (m * (m - 1))
    return float(2 * dab - daa - dbb)


def drug_block_bootstrap(dt: pd.DataFrame, fn=None, B: int = 2000, seed: int = 20261117):
    """Copied from R3: resample drugs (all their cell lines together). `dt` is indexed by drug."""
    fn = fn or (lambda t: float(t.iloc[:, 0].mean()))
    rng = np.random.default_rng(seed)
    n = len(dt)
    est = fn(dt)
    bs = np.array([fn(dt.iloc[rng.integers(0, n, n)]) for _ in range(B)])
    return dict(est=est, lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)), n_drugs=int(n))


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def js(p, q, eps=1e-12):
    p, q = np.asarray(p) + eps, np.asarray(q) + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def neff(w):
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    return float(s * s / (w * w).sum()) if s > 0 else 0.0


# ------------------------------------------------------------------ the fixed atlas
class Atlas:
    """Fixed train-only GMM states; s_k(z) is the posterior, read with numpy (no refit)."""

    def __init__(self, path, K):
        g = np.load(f"{path}/gmm_K{K}.npz")
        self.K = K
        self.w = g["weights"].astype(np.float64)
        self.mu = g["means"].astype(np.float64)
        self.cov = g["covariances"].astype(np.float64)
        self.L = np.linalg.cholesky(self.cov)
        self.logdet = 2 * np.log(np.diagonal(self.L, axis1=1, axis2=2)).sum(1)

    def maha2(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        out = np.empty((len(Z), self.K))
        for k in range(self.K):
            y = np.linalg.solve(self.L[k], (Z - self.mu[k]).T)
            out[:, k] = (y * y).sum(0)
        return out

    def read(self, Z):
        m = self.maha2(Z)
        ll = np.log(self.w) - 0.5 * (m + self.logdet)
        ll -= ll.max(1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(1, keepdims=True)


# ------------------------------------------------------------------ per-line data
class LineData:
    def __init__(self, cfg, line, K):
        root = f"{cfg['cache_dir']}/atlas_{line}"
        self.line, self.K = line, K
        self.Z = np.load(f"{root}/Z.npy").astype(np.float64)
        self.S = np.load(f"{root}/S_K{K}.npy").astype(np.float64)
        self.obs = pd.read_parquet(f"{root}/obs.parquet")
        self.wells = pd.read_csv(f"{root}/wells.tsv", sep="\t")
        self.G = np.load(f"{root}/well_gene_means.npy").astype(np.float64)
        dec = np.load(f"{root}/decoder.npz", allow_pickle=True)
        self.dec_mu, self.dec_sd, self.dec_comp = dec["mu"].astype(np.float64), dec["sd"].astype(
            np.float64), dec["comp"].astype(np.float64)
        info = json.load(open(f"{root}/fit_info.json"))
        self.bw = info["mmd_bandwidth"]
        self.atlas = Atlas(root, K)
        fit = np.load(f"{root}/fit_idx.npy")
        self.maha_thr = float(np.quantile(self.atlas.maha2(self.Z[fit]).min(1), 0.99))
        self.idx = {c: np.asarray(v) for c, v in self.obs.groupby("Combo").groups.items()}
        self.well_row = {c: i for i, c in enumerate(self.wells.Combo)}
        w = self.wells
        self.combo = {(r.drug, float(r.dose), r.replicate): r.Combo for r in w.itertuples()}
        self.ctrl = {rep: np.concatenate([self.idx[c] for c in w[(w.dose == 0) & (w.replicate == rep)].Combo])
                     for rep in ("rep1", "rep2")}

    def cells(self, drug, dose, rep):
        c = self.combo.get((drug, float(dose), rep))
        return None if c is None else self.idx[c]

    def decode_mean(self, zbar):
        return self.dec_mu + self.dec_sd * (np.asarray(zbar) @ self.dec_comp)


# ------------------------------------------------------------------ endpoint context (no query!)
@dataclass
class Context:
    """Everything a predictor may see. The query well is NOT part of this object."""
    line: str
    drug: str
    episode: str
    rep_end: str
    t: float
    ZL: np.ndarray
    SL: np.ndarray
    ZH: np.ndarray
    SH: np.ndarray
    ZC: np.ndarray
    SC: np.ndarray
    atlas: Atlas


@dataclass
class Query:
    rep: str
    Z: np.ndarray
    S: np.ndarray
    gene_mean: np.ndarray


def state_moments(Z, S, atlas: Atlas, kappa_mean=5.0, kappa_cov=20.0):
    """Per-state occupancy, raw and shrunk conditional moments and effective sizes."""
    n = len(Z)
    nk = S.sum(0)
    pi = nk / n
    m = (S.T @ Z) / np.maximum(nk[:, None], 1e-12)
    ne = np.array([neff(S[:, k]) for k in range(S.shape[1])])
    lam = ne / (ne + kappa_mean)
    ms = lam[:, None] * m + (1 - lam[:, None]) * atlas.mu
    covs = []
    for k in range(S.shape[1]):
        D = Z - m[k]
        C = (S[:, k, None] * D).T @ D / max(nk[k], 1e-12)
        lc = ne[k] / (ne[k] + kappa_cov)
        covs.append(lc * C + (1 - lc) * atlas.cov[k])
    return dict(pi=pi, m=m, m_shrunk=ms, cov_shrunk=np.stack(covs), neff=ne, n=n)


def interp_state(ctx: Context, kappa_mean=5.0, kappa_cov=20.0):
    """Mixture-consistent state-wise interpolation (plan section 7)."""
    t = ctx.t
    L = state_moments(ctx.ZL, ctx.SL, ctx.atlas, kappa_mean, kappa_cov)
    H = state_moments(ctx.ZH, ctx.SH, ctx.atlas, kappa_mean, kappa_cov)
    pi = (1 - t) * L["pi"] + t * H["pi"]
    a = (1 - t) * L["pi"] / pi
    b = t * H["pi"] / pi
    m_raw = a[:, None] * L["m"] + b[:, None] * H["m"]
    m_s = a[:, None] * L["m_shrunk"] + b[:, None] * H["m_shrunk"]
    dm = H["m_shrunk"] - L["m_shrunk"]
    cov = (a[:, None, None] * L["cov_shrunk"] + b[:, None, None] * H["cov_shrunk"]
           + (a * b)[:, None, None] * dm[:, :, None] * dm[:, None, :])
    return dict(pi=pi, a=a, b=b, m_raw=m_raw, m_shrunk=m_s, cov=cov, L=L, H=H)


def build_context(D: LineData, drug, episode, ep_def, rep_end):
    ZL = D.cells(drug, ep_def["low"], rep_end)
    ZH = D.cells(drug, ep_def["high"], rep_end)
    t = (np.log(ep_def["query"]) - np.log(ep_def["low"])) / (np.log(ep_def["high"]) - np.log(ep_def["low"]))
    c = D.ctrl[rep_end]
    return Context(D.line, drug, episode, rep_end, float(t), D.Z[ZL], D.S[ZL], D.Z[ZH], D.S[ZH], D.Z[c], D.S[c],
                   D.atlas)


def build_query(D: LineData, drug, dose, rep):
    q = D.cells(drug, dose, rep)
    return Query(rep, D.Z[q], D.S[q], D.G[D.well_row[D.combo[(drug, float(dose), rep)]]])


# ------------------------------------------------------------------ scoring
def score(D: LineData, P: Cloud, Q: Query, min_state_neff=5.0, ED_only=False) -> dict:
    Qc = Cloud.uniform(Q.Z)
    out = dict(ed=cloud_stat(P, Qc, "ed"))
    if ED_only:
        return out
    out["mmd"] = cloud_stat(P, Qc, "mmd", bw=D.bw)
    S_p = D.atlas.read(P.X)
    pi_p = P.w @ S_p
    pi_q = Q.S.mean(0)
    out.update(comp_tv=tv(pi_p, pi_q), comp_js=js(pi_p, pi_q), comp_brier=float(((pi_p - pi_q) ** 2).sum()))
    for k in range(D.K):
        out[f"pi_pred_{k}"] = float(pi_p[k]); out[f"pi_obs_{k}"] = float(pi_q[k])
    mp, mq = P.mean(), Q.Z.mean(0)
    out["mean_rmse_latent"] = float(np.sqrt(((mp - mq) ** 2).mean()))
    gp = D.decode_mean(mp)
    out["dmean_rmse_gene"] = float(np.sqrt(((gp - Q.gene_mean) ** 2).mean()))
    out["decoder_recon_rmse_gene"] = float(np.sqrt(((D.decode_mean(mq) - Q.gene_mean) ** 2).mean()))
    # within-state (re-read with the fixed atlas on both sides)
    num_m = num_e = den = 0.0
    n_ok = 0
    for k in range(D.K):
        wp = P.w * S_p[:, k]
        wq = Q.S[:, k]
        nq = neff(wq)
        out[f"neff_q_{k}"] = nq
        out[f"neff_p_{k}"] = neff(wp)
        if nq < min_state_neff or wp.sum() < 1e-8:
            out[f"within_mean_rmse_{k}"] = np.nan; out[f"within_ed_{k}"] = np.nan
            continue
        cm_p = (wp @ P.X) / wp.sum()
        cm_q = (wq @ Q.Z) / wq.sum()
        e_m = float(np.sqrt(((cm_p - cm_q) ** 2).mean()))
        e_d = cloud_stat(P, Qc, "ed", wP=wp, wQ=wq)
        out[f"within_mean_rmse_{k}"] = e_m; out[f"within_ed_{k}"] = e_d
        num_m += pi_q[k] * e_m; num_e += pi_q[k] * e_d; den += pi_q[k]; n_ok += 1
    out["within_mean_rmse"] = num_m / den if den > 0 else np.nan
    out["within_ed"] = num_e / den if den > 0 else np.nan
    out["within_n_states"] = n_ok
    out["within_mass_scored"] = den
    # assembly consistency for state-wise outputs
    if P.declared_pi is not None:
        out["head_comp_tv"] = tv(P.declared_pi, pi_q)
        out["assembly_tv_head_vs_reread"] = tv(P.declared_pi, pi_p)
        if P.declared_means is not None:
            rr = []
            for k in range(D.K):
                wp = P.w * S_p[:, k]
                if neff(wp) >= min_state_neff:
                    rr.append(np.sqrt((((wp @ P.X) / wp.sum() - P.declared_means[k]) ** 2).mean()))
            out["assembly_mean_rmse_head_vs_reread"] = float(np.mean(rr)) if rr else np.nan
    return out


def query_diagnostics(D: LineData, Q: Query) -> dict:
    m2 = D.atlas.maha2(Q.Z).min(1)
    return dict(n_query=len(Q.Z), query_frac_out_of_atlas=float((m2 > D.maha_thr).mean()),
                query_mean_max_membership=float(Q.S.max(1).mean()))
