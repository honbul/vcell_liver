"""Training-set preparation and configuration grids for the R1 model family.

A `Prep` holds everything fitted on one training set (feature scaler, label scaler, pathway basis,
kernels, reduced basis). Rows to predict are passed as features only; their labels never enter.

Configuration keys (tuples):
    ("ridge", kernel, lam)            kernel in {"whole", "par", "perp", "r1", "r10", "r100"}
                                      whole = eta 0, par = eta -> inf, perp = pathway part removed
                                      (diagnostic only), rK = eta / lam = K
    ("nuc", ratio, lam, tau_frac)     eta = ratio * lam, tau = tau_frac * tau_max(training set)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_solvers as S  # noqa: E402
from r1_data import a1core  # noqa: E402

CONST_SD = 1e-8
KERNEL_RATIO = {"whole": 0.0, "r1": 1.0, "r10": 10.0, "r100": 100.0, "par": np.inf}


class Prep:
    def __init__(self, X_tr, Y_tr, fam_tr, X_te, sets, need_reduced=True):
        self.wt = S.norm_weights(a1core.family_weights(np.asarray(fam_tr)))
        self.x_mu, self.x_sd = S.wmean_sd(X_tr, self.wt)
        Xs = S.standardise(X_tr, self.x_mu, self.x_sd)
        Xt = S.standardise(X_te, self.x_mu, self.x_sd)
        self.y_mu, self.y_sd = S.wmean_sd(Y_tr, self.wt)
        self.constant = self.y_sd < CONST_SD
        self.y_sd_safe = np.where(self.constant, 1.0, self.y_sd)
        self.Z = (Y_tr - self.y_mu) / self.y_sd_safe
        self.D = Y_tr.shape[1]
        self.n_te = X_te.shape[0]
        self.K = Xs @ Xs.T
        self.K_te = Xt @ Xs.T
        self.has_prior = sets is not None
        if self.has_prior:
            Q = S.orth_basis(S.membership_std(sets, self.x_sd))
            self.r = Q.shape[1]
            XQ, XtQ = Xs @ Q, Xt @ Q
            self.K_par, self.K_te_par = XQ @ XQ.T, XtQ @ XQ.T
            self.K_perp, self.K_te_perp = self.K - self.K_par, self.K_te - self.K_te_par
            # share of the standardised training feature variance that lies in the pathway subspace
            self.par_fraction = float(np.trace(self.wt[:, None] * self.K_par) / np.trace(self.wt[:, None] * self.K))
        else:
            Q = np.zeros((Xs.shape[1], 0))
            self.r = 0
        if need_reduced:
            B = S.reduced_basis(Xs, Q)
            self.XB, self.XtB = Xs @ B, Xt @ B
            self.m = B.shape[1]
            self.tau_max = S.tau_max(self.XB, self.wt, self.Z)
        self._dual = {}
        self._nuc_warm = {}

    # ---- tau = 0 ----
    def kernels(self, kernel: str) -> tuple:
        if kernel == "whole":
            return self.K, self.K_te
        if kernel == "perp":
            return self.K_perp, self.K_te_perp
        ratio = KERNEL_RATIO[kernel]
        return S.soft_kernel(self.K_par, self.K_perp, ratio), S.soft_kernel(self.K_te_par, self.K_te_perp, ratio)

    def predict_ridge_z(self, kernel: str, lam: float) -> np.ndarray:
        if kernel not in self._dual:
            Ktr, Kte = self.kernels(kernel)
            self._dual[kernel] = (S.DualRidge(Ktr, self.wt, self.Z), Kte)
        dual, Kte = self._dual[kernel]
        return Kte @ dual.alpha(lam)

    # ---- tau > 0 ----
    def predict_nuc_z(self, ratio: float, lam: float, tau_frac: float, tol: float = 1e-7) -> tuple:
        eta = ratio * lam
        H, Rm = S.smooth_terms(self.XB, self.wt, self.Z, self.r, lam, eta)
        if tau_frac == 0:
            C = np.linalg.solve(H, Rm)
            return self.XtB @ C, dict(converged=True, iterations=0, rank=min(C.shape))
        warm = self._nuc_warm.get((ratio, lam))
        C, info = S.admm_nuclear(H, Rm, tau_frac * self.tau_max, C0=warm, tol=tol)
        self._nuc_warm[(ratio, lam)] = C
        return self.XtB @ C, info

    def predict_z(self, cfg_key: tuple) -> tuple:
        if cfg_key[0] == "ridge":
            return self.predict_ridge_z(cfg_key[1], cfg_key[2]), None
        if cfg_key[0] == "nuc":
            return self.predict_nuc_z(cfg_key[1], cfg_key[2], cfg_key[3])
        raise ValueError(cfg_key)

    def to_raw(self, Pz: np.ndarray) -> np.ndarray:
        return Pz * self.y_sd_safe + self.y_mu

    def z_of(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.y_mu) / self.y_sd_safe


def ridge_configs(lams, kernels) -> list:
    return [("ridge", k, float(l)) for k in kernels for l in lams]


def nuc_configs(lams, ratios, tau_fracs) -> list:
    # tau descending inside each (ratio, lam) so that warm starts follow the regularisation path
    return [("nuc", float(r), float(l), float(t)) for r in ratios for l in lams
            for t in sorted(tau_fracs, reverse=True)]


def key_str(k: tuple) -> str:
    return "|".join(str(x) for x in k)


def inner_losses(X, Y, fam, sets, configs, n_inner, seed, return_family=False):
    """Track A inner loss for every configuration: family-first squared error in inner-training-SD
    units, pooled over the inner test families. Returns loss [config, drug] and solver records
    (and, on request, the per-family terms [config, family, drug] with the family ids)."""
    fam = np.asarray(fam)
    inner = a1core.group_folds(fam, n_inner, seed)
    fams = np.unique(fam)
    f_pos = {f: i for i, f in enumerate(fams)}
    fam_loss = np.full((len(configs), len(fams), Y.shape[1]), np.nan)
    records = []
    need_reduced = any(c[0] == "nuc" for c in configs)
    for k in range(n_inner):
        tr, te = inner != k, inner == k
        assert not set(fam[tr]) & set(fam[te])
        prep = Prep(X[tr], Y[tr], fam[tr], X[te], sets, need_reduced=need_reduced)
        Zte = prep.z_of(Y[te])
        f_te, inv = np.unique(fam[te], return_inverse=True)
        cnt = np.bincount(inv)
        pos = np.array([f_pos[f] for f in f_te])
        for c, key in enumerate(configs):
            Pz, info = prep.predict_z(key)
            se = (Pz - Zte) ** 2
            agg = np.zeros((len(f_te), Y.shape[1]))
            np.add.at(agg, inv, se)
            fam_loss[c, pos] = agg / cnt[:, None]
            if info is not None:
                records.append(dict(inner_fold=k, config=key_str(key), converged=info["converged"],
                                    iterations=info["iterations"], rank=info["rank"]))
    assert np.isfinite(fam_loss).all()
    loss = fam_loss.mean(axis=1)
    if return_family:
        return loss, records, fam_loss, fams
    return loss, records


def select_per_drug(loss: np.ndarray, order: list) -> np.ndarray:
    """Index (into the rows of loss) of the best configuration per drug. `order` lists the candidate
    rows from weakest to strongest regularisation; ties go to the strongest."""
    sub = loss[order]
    best = len(order) - 1 - np.argmin(sub[::-1], axis=0)
    return np.asarray(order)[best]


def select_global(loss: np.ndarray, order: list) -> int:
    """One configuration for all drugs: smallest mean over drugs of sqrt(inner loss); ties -> strongest."""
    score = np.sqrt(loss[order]).mean(axis=1)
    best = len(order) - 1 - int(np.argmin(score[::-1]))
    return int(order[best])
