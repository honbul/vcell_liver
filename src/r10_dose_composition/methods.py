"""Predictors. Every function receives a `Context` (endpoints, control, dose position) and
never the query well. Learned heads are small MLPs on endpoint summaries (see `features`).

Non-learned baselines (plan section 8)
  control          vehicle cells of the endpoint replicate
  nearest_low      the low-dose endpoint cloud as is (tie: both sides reported, none chosen)
  nearest_high     the high-dose endpoint cloud as is
  mixture          log-dose empirical mixture (1-t) P_L + t P_H  (== D00 of the 2x2)
  global_moment    one Gaussian with the mixture mean and mixture covariance (Ledoit-Wolf
                   shrunk endpoint covariances), n_gen fixed samples
  state_moment     fixed state-wise mixture-consistent moment interpolation (shrunk state
                   moments), Gaussian per state, n_gen fixed samples
  ot_displacement  exact OT (squared Euclidean, uniform weights) between the two endpoint
                   clouds in the train-only PCA space; McCann displacement at t
Learned (same endpoint information)
  D10 / D01 / D11  2x2 heads on top of the mixture (empirical-residual assembly)
  W                compact whole-distribution model: global affine (shift + per-axis scale)
                   of the mixture, trained with the energy distance
"""
from __future__ import annotations

import numpy as np
import ot
import torch
from sklearn.covariance import LedoitWolf

from .core import Cloud, Context, interp_state, state_moments

EPS = 1e-6


def _mix(ctx: Context):
    nL, nH, t = len(ctx.ZL), len(ctx.ZH), ctx.t
    X = np.concatenate([ctx.ZL, ctx.ZH])
    w = np.concatenate([np.full(nL, (1 - t) / nL), np.full(nH, t / nH)])
    g = np.concatenate([np.zeros(nL, int), np.ones(nH, int)])
    S = np.concatenate([ctx.SL, ctx.SH])
    return X, w, g, S


def baseline(name: str, ctx: Context, cfg: dict, seed: int) -> Cloud:
    rng = np.random.default_rng(seed)
    sh = cfg["shrink"]
    if name == "control":
        return Cloud.uniform(ctx.ZC)
    if name == "nearest_low":
        return Cloud.uniform(ctx.ZL)
    if name == "nearest_high":
        return Cloud.uniform(ctx.ZH)
    if name == "mixture":
        X, w, g, _ = _mix(ctx)
        I = interp_state(ctx, sh["kappa_mean"], sh["kappa_cov"])
        return Cloud(X, w, g, declared_pi=I["pi"], declared_means=I["m_raw"])
    if name == "global_moment":
        t = ctx.t
        mL, mH = ctx.ZL.mean(0), ctx.ZH.mean(0)
        CL = LedoitWolf().fit(ctx.ZL).covariance_
        CH = LedoitWolf().fit(ctx.ZH).covariance_
        d = mH - mL
        C = (1 - t) * CL + t * CH + t * (1 - t) * np.outer(d, d)
        X = rng.multivariate_normal((1 - t) * mL + t * mH, C, size=cfg["n_gen_gaussian"], method="cholesky")
        return Cloud.uniform(X)
    if name == "state_moment":
        I = interp_state(ctx, sh["kappa_mean"], sh["kappa_cov"])
        n = cfg["n_gen_gaussian"]
        cnt = np.floor(I["pi"] * n).astype(int)
        rem = n - cnt.sum()
        cnt[np.argsort(-(I["pi"] * n - cnt))[:rem]] += 1
        X = np.concatenate([rng.multivariate_normal(I["m_shrunk"][k], I["cov"][k], size=cnt[k], method="cholesky")
                            for k in range(len(cnt)) if cnt[k] > 0])
        return Cloud(X, np.full(n, 1.0 / n), np.zeros(n, int), declared_pi=I["pi"], declared_means=I["m_shrunk"])
    if name == "ot_displacement":
        nL, nH, t = len(ctx.ZL), len(ctx.ZH), ctx.t
        M = ot.dist(ctx.ZL, ctx.ZH)                               # squared Euclidean
        G = ot.emd(np.full(nL, 1 / nL), np.full(nH, 1 / nH), M, numItermax=2_000_000)
        i, j = np.nonzero(G > 1e-12)
        X = (1 - t) * ctx.ZL[i] + t * ctx.ZH[j]
        w = G[i, j] / G[i, j].sum()
        return Cloud(X, w, np.zeros(len(X), int))
    raise KeyError(name)


# ------------------------------------------------------------------ 2x2 assembly
def assemble(ctx: Context, cfg: dict, r_pi=None, r_m=None) -> Cloud:
    """Empirical-residual assembly on the mixture. r_pi: [K] logit residual; r_m: [K, d].

    D00 (both None) reproduces the empirical mixture exactly. The pi-head reweights every cell
    by sum_k s_k(z) pi_hat_k / pi_int_k (total mass stays 1); the Q-head shifts every cell by
    sum_k s_k(z) r_k. Memberships s_k are the fixed-atlas memberships of the endpoint cells."""
    sh = cfg["shrink"]
    X, w, g, S = _mix(ctx)
    I = interp_state(ctx, sh["kappa_mean"], sh["kappa_cov"])
    pi_int = I["pi"]
    pi_hat = pi_int
    if r_pi is not None:
        lg = np.log(pi_int + EPS) + r_pi
        pi_hat = np.exp(lg - lg.max())
        pi_hat /= pi_hat.sum()
        w = w * (S @ (pi_hat / pi_int))
        w = w / w.sum()
    means = I["m_raw"]
    if r_m is not None:
        X = X + S @ r_m
        means = means + r_m
    return Cloud(X, w, g, declared_pi=pi_hat, declared_means=means)


# ------------------------------------------------------------------ features (endpoint summaries)
LINE_IDX = {"A549": 0, "K562": 1, "MCF7": 2}
EP_IDX = {"E100": 0, "E1000": 1}


def features(ctx: Context, cfg: dict) -> tuple[np.ndarray, dict]:
    sh = cfg["shrink"]
    I = interp_state(ctx, sh["kappa_mean"], sh["kappa_cov"])
    L, H = I["L"], I["H"]
    C = state_moments(ctx.ZC, ctx.SC, ctx.atlas, sh["kappa_mean"], sh["kappa_cov"])
    oh = np.zeros(3); oh[LINE_IDX[ctx.line]] = 1
    oe = np.zeros(2); oe[EP_IDX[ctx.episode]] = 1
    mc = ctx.ZC.mean(0)
    f = np.concatenate([oh, oe, L["pi"], H["pi"], C["pi"], np.log((H["pi"] + 1e-3) / (L["pi"] + 1e-3)),
                        (L["m_shrunk"] - ctx.atlas.mu).ravel(), (H["m_shrunk"] - L["m_shrunk"]).ravel(),
                        ctx.ZL.mean(0) - mc, ctx.ZH.mean(0) - mc, [np.log(L["n"]), np.log(H["n"])],
                        np.log1p(L["neff"]), np.log1p(H["neff"])])
    return f.astype(np.float32), I


class MLP(torch.nn.Module):
    def __init__(self, n_in, n_out, hidden=64, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                                       torch.nn.Linear(hidden, n_out))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)          # start exactly at the interpolation (D00)

    def forward(self, x):
        return self.net(x)


class Head:
    """A trained head with its feature standardisation, usable on any Context."""

    def __init__(self, kind, state_dict, f_mean, f_sd, n_in, n_out, K, d, hidden=64):
        self.kind, self.K, self.d = kind, K, d
        self.net = MLP(n_in, n_out, hidden)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.f_mean, self.f_sd = f_mean, f_sd

    def __call__(self, f):
        with torch.no_grad():
            x = torch.as_tensor((f - self.f_mean) / self.f_sd, dtype=torch.float32)[None]
            return self.net(x)[0].numpy().astype(np.float64)

    @staticmethod
    def load(path):
        z = torch.load(path, weights_only=False)
        return Head(z["kind"], z["state_dict"], z["f_mean"], z["f_sd"], z["n_in"], z["n_out"], z["K"], z["d"],
                    z.get("hidden", 64))


def predict_learned(arm: str, ctx: Context, cfg: dict, heads: dict) -> Cloud:
    f, _ = features(ctx, cfg)
    if arm == "W":
        o = heads["W"](f)
        d = ctx.ZL.shape[1]
        X, w, g, _ = _mix(ctx)
        mu = w @ X
        X = mu + (X - mu) * np.exp(o[d:]) + o[:d]
        return Cloud(X, w, g)
    r_pi = heads["pi"](f) if arm in ("D10", "D11") else None
    r_m = heads["Q"](f).reshape(ctx.SL.shape[1], -1) if arm in ("D01", "D11") else None
    return assemble(ctx, cfg, r_pi, r_m)
