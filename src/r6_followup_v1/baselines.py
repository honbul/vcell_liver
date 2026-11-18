"""Response-mean baselines that use measured responses of FITTING drugs only (plan 5.1, R6-A1).

    global mean   mean response over all fitting conditions
    line mean     b_c    : mean over the fitting drugs measured in the cell line
    pool mean     b_{c,p}: mean over the fitting drugs whose primary plate is the pool's plate
    shrunk        ( n b_{c,p} + kappa b_c ) / ( n + kappa ),  n = number of fitting drug groups in the pool
                  kappa = 0 -> pool mean, kappa = inf -> line mean
Every drug group contributes one condition per (cell line, pool), so drug-level and condition-level
means coincide. A pool without fitting drugs falls back to the line mean, a line without fitting
drugs to the global mean; such conditions are kept and flagged, never dropped.

`exclude_self=True` gives the leave-one-drug-out version for conditions that are themselves fitting
conditions (training anchors): the condition's own response is removed from its pool mean and from
its line mean before the estimate is formed.
"""
from __future__ import annotations

import numpy as np

KAPPAS = (0.0, 2.0, 8.0, 32.0, float("inf"))


class MeanBaseline:
    def __init__(self, Y: np.ndarray, Lf: np.ndarray, Df: np.ndarray, pf: np.ndarray, n_lines: int, n_pools: int):
        """Y [L, D, G] truth; (Lf, Df, pf) the fitting conditions and their pool ids."""
        G = Y.shape[2]
        self.Y = Y
        yf = Y[Lf, Df].astype(np.float64)
        self.S_pool = np.zeros((n_pools, G)); self.n_pool = np.zeros(n_pools)
        self.S_line = np.zeros((n_lines, G)); self.n_line = np.zeros(n_lines)
        np.add.at(self.S_pool, pf, yf); np.add.at(self.n_pool, pf, 1.0)
        np.add.at(self.S_line, Lf, yf); np.add.at(self.n_line, Lf, 1.0)
        self.glob = yf.mean(0)
        self.fit_key = set(zip(Lf.tolist(), Df.tolist()))
        self.fit_drugs = set(Df.tolist())

    def predict(self, Lq, Dq, pq, kappa: float, exclude_self: bool = False):
        """Returns (P [n, G] float32, level [n] in {'pool', 'line', 'global'}, n_pool_drugs [n])."""
        Lq, Dq, pq = np.asarray(Lq), np.asarray(Dq), np.asarray(pq)
        Sp, npl = self.S_pool[pq].copy(), self.n_pool[pq].copy()
        Sl, nl = self.S_line[Lq].copy(), self.n_line[Lq].copy()
        if not exclude_self:
            assert not (set(Dq.tolist()) & self.fit_drugs), "query drugs are fitting drugs: use exclude_self=True"
        else:
            own = np.array([(l, g) in self.fit_key for l, g in zip(Lq.tolist(), Dq.tolist())])
            assert own.all(), "exclude_self is for fitting conditions only"
            y = self.Y[Lq, Dq].astype(np.float64)
            Sp -= y; npl -= 1; Sl -= y; nl -= 1
        b_line = np.where(nl[:, None] > 0, Sl / np.maximum(nl, 1)[:, None], self.glob[None])
        b_pool = Sp / np.maximum(npl, 1)[:, None]
        if np.isinf(kappa):
            P = b_line
        else:
            w = np.where(npl > 0, npl / np.maximum(npl + kappa, 1e-12), 0.0)[:, None]
            P = w * b_pool + (1 - w) * b_line
        level = np.where((npl > 0) & ~np.isinf(kappa), "pool", np.where(nl > 0, "line", "global"))
        return P.astype(np.float32), level, npl.astype(int)

    def global_predict(self, n: int) -> np.ndarray:
        return np.repeat(self.glob[None].astype(np.float32), n, 0)
