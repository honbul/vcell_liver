"""Readout (fixed for every policy) and the Gaussian-process acquisition model in the whitened residual space z."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .data import tanimoto

KERNELS = ["tanimoto", "f0_cosine", "combined"]
NOISE_MULT = [1.0, 2.0]
HYPOTHESES = [(k, m) for k in KERNELS for m in NOISE_MULT]


# ------------------------------------------------------------------ episode-level kernels and features
class EpisodeKernels:
    def __init__(self, ep, bases):
        self.tan = tanimoto(ep.fp, ep.fp).astype(np.float64)
        f = ep.feat.astype(np.float64)
        f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
        self.cos = f @ f.T
        self.U = bases["z_U"].astype(np.float64)
        self.sq = np.sqrt(bases["z_lam"].astype(np.float64))
        self.f0z = (ep.f0.astype(np.float64) @ self.U) / self.sq
        self.f0z_norm = np.linalg.norm(self.f0z, axis=1) / np.sqrt(self.U.shape[1])

    def K(self, name):
        if name == "tanimoto":
            return self.tan
        if name == "f0_cosine":
            return self.cos
        if name == "combined":
            return 0.5 * (self.tan + self.cos)
        raise ValueError(name)

    def readout_K(self, alpha):
        return alpha * self.tan + (1 - alpha) * self.cos

    def to_z(self, ep, idx, y):
        """Residual of revealed well deltas against f0, projected and scaled (acquisition space only)."""
        return ((y.astype(np.float64) - ep.f0[idx].astype(np.float64)) @ self.U) / self.sq


def noise_features(ep, ek):
    r = ep.rep
    return np.stack([np.ones(len(r)), np.log(r.ctl_sd.to_numpy(np.float64)), np.log(r.n_ctl.to_numpy(np.float64)),
                     np.log(np.maximum(ek.f0z_norm, 1e-6)), np.log1p(ep.n_f0.astype(np.float64))], 1)


def nu_vector(ep, ek, hp):
    """Per-group observation noise (z units per axis) of the representative row."""
    if hp.get("hetero") and hp.get("beta") is not None:
        x = noise_features(ep, ek)
        x = np.nan_to_num(x, nan=np.nanmedian(x[:, 1]) if np.isnan(x[:, 1]).any() else 0.0)
        return hp["noise_hetero"] * np.exp(x @ np.asarray(hp["beta"]))
    return np.full(len(ep.groups), hp["noise_homo"])


# ------------------------------------------------------------------ readout
class Readout:
    def __init__(self, alpha, lam, kind="main"):
        self.alpha, self.lam, self.kind = alpha, lam, kind

    def predict(self, ep, ek, S, Y, query):
        """S: revealed group indices, Y: their well deltas (978). Returns predictions for `query` indices."""
        S = np.asarray(S)
        e = Y.astype(np.float64) - ep.f0[S].astype(np.float64)
        b = e.mean(0)
        pred = ep.f0[query].astype(np.float64) + b
        if self.kind == "mean_offset" or len(S) < 2:
            return pred
        K = ek.readout_K(self.alpha)
        Kss = K[np.ix_(S, S)] + self.lam * np.eye(len(S))
        coef = cho_solve(cho_factor(Kss), e - b)
        return pred + K[np.ix_(query, S)] @ coef


# ------------------------------------------------------------------ GP in z space
class GP:
    """One hypothesis: prior cov sb2 + s2 K, noise diag(nu * mult). Same covariance on every z axis (axes independent)."""

    def __init__(self, Kfull, nu, sb2, s2, mult=1.0):
        self.C = sb2 + s2 * Kfull
        self.nu = nu * mult

    def posterior(self, S, Z, P):
        """Posterior mean (|P| x d) and covariance (|P| x |P|) of the latent at P given (S, Z)."""
        S = np.asarray(S, int)
        P = np.asarray(P, int)
        Cpp = self.C[np.ix_(P, P)]
        if len(S) == 0:
            return np.zeros((len(P), Z.shape[1] if Z is not None and Z.ndim == 2 else 0)), Cpp.copy()
        A = self.C[np.ix_(S, S)] + np.diag(self.nu[S])
        cf = cho_factor(A)
        Cps = self.C[np.ix_(P, S)]
        mu = Cps @ cho_solve(cf, Z)
        Sig = Cpp - Cps @ cho_solve(cf, Cps.T)
        return mu, Sig

    def loglik(self, S, Z):
        S = np.asarray(S, int)
        A = self.C[np.ix_(S, S)] + np.diag(self.nu[S])
        cf = cho_factor(A)
        alpha = cho_solve(cf, Z)
        logdet = 2 * np.log(np.diag(cf[0])).sum()
        n, d = Z.shape
        return float(-0.5 * (Z * alpha).sum() - 0.5 * d * logdet - 0.5 * n * d * np.log(2 * np.pi))


def make_gps(ek, nu, hp, which="mixture"):
    """List of (name, GP) for the mixture, or the single fixed combined homoscedastic model."""
    if which == "fixed":
        return [("combined", GP(ek.K("combined"), nu, hp["sb2"], hp["s2"]["combined"], 1.0))]
    return [(f"{k}_x{m:g}", GP(ek.K(k), nu, hp["sb2"], hp["s2"][k], m)) for k, m in HYPOTHESES]


def mixture_weights(gps, S, Z, T):
    if len(S) == 0:
        return np.full(len(gps), 1.0 / len(gps))
    ll = np.array([g.loglik(S, Z) for _, g in gps]) * T
    w = np.exp(ll - ll.max())
    return w / w.sum()
