"""Distribution baselines that share the mean predictor: control, shifted control, Gaussians.

Each generator returns, for one evaluation well, `n` cells in log1p(CP10k) units BEFORE the shared
anchoring (centre -> add mu_hat -> non-negative restoration), which evaluate.py applies identically
to every mean-anchored method.

Gaussian covariances are fitted on the setting's TRAINING wells only, per cell line, from
within-well residuals (cell minus its own well mean):
    diag      per-gene pooled residual variance of the line
    lowrank   top-k eigenvectors of the line's pooled residual covariance, eigenvalues shrunk
              towards the average remaining variance, plus a diagonal remainder chosen so that
              every gene keeps its pooled marginal variance (k fixed in development)
"""
from __future__ import annotations

import numpy as np
import torch


def control_cells(st, r, n, seed, dev):
    C = st.d.ctrl_cells(r.plate_index, r.line)
    rng = np.random.default_rng([seed, int(r.gid)])
    idx = rng.permutation(len(C))[:n]
    return torch.as_tensor(C[np.sort(idx)], dtype=torch.float32, device=dev)


class TreatedPool:
    """Real TRAINING-drug treated cells of the same (plate, line), each minus its own well mean.

    A non-parametric shape baseline suggested by diagnostic R3-0: no model, no held-out cell.
    """

    def __init__(self, st):
        self.st = st
        self.pool = {k: g[["start", "end"]].to_numpy() for k, g in st.fit_wells.groupby(["plate_index", "line"])}

    def sample(self, r, n, seed, dev):
        se = self.pool[(r.plate_index, r.line)]
        rng = np.random.default_rng([seed, int(r.gid), 1])
        sizes = se[:, 1] - se[:, 0]
        w = rng.choice(len(se), size=n, p=sizes / sizes.sum())
        X = self.st.d.X
        out = np.empty((n, X.shape[1]), np.float32)
        for j, wi in enumerate(w):
            block = np.asarray(X[se[wi, 0]:se[wi, 1]])
            m = len(block)
            out[j] = (block[rng.integers(m)] - block.mean(0)) * np.sqrt(m / (m - 1))
        return torch.as_tensor(out, device=dev)


class GaussianBank:
    def __init__(self, st, k_max=50, dev="cpu"):
        d = st.d
        G = d.X.shape[1]
        self.var, self.U, self.lam, self.dev = {}, {}, {}, dev
        for line, g in st.fit_wells.groupby("line"):
            S = torch.zeros(G, G, dtype=torch.float64, device=dev); dof = 0
            for r in g.itertuples():
                x = torch.as_tensor(np.asarray(d.X[r.start:r.end]), dtype=torch.float64, device=dev)
                x = x - x.mean(0)
                S += x.T @ x; dof += len(x) - 1
            S /= dof
            ev, V = torch.linalg.eigh(S)
            ev, V = ev.flip(0)[:k_max], V.flip(1)[:, :k_max]
            self.var[int(line)] = torch.diag(S).float()
            self.U[int(line)] = V.float(); self.lam[int(line)] = ev.float()
            self.rest = None
        self.G = G

    def sample(self, line, n, seed, gid, kind="diag", k=50):
        g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(gid))
        var = self.var[line]
        z = torch.randn(n, self.G, generator=g).to(self.dev)
        if kind == "diag":
            return z * var.sqrt()
        U, lam = self.U[line][:, :k], self.lam[line][:k]
        resid_mean = (var.sum() - lam.sum()) / self.G           # average variance left outside the factors
        lam_f = (lam - resid_mean).clamp(min=0)                 # shrink factor variances
        diag = (var - (U ** 2) @ lam_f).clamp(min=0.05 * var)   # keep each gene's marginal variance
        f = torch.randn(n, k, generator=g).to(self.dev)
        return (f * lam_f.sqrt()) @ U.T + z * diag.sqrt()
