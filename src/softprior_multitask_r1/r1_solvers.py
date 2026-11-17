"""Solvers for the R1 objective.

    f(W) = (1/D) * sum_p sum_i wt_i (x_i W_p - z_ip)^2 + lam ||W||_F^2 + eta ||(I - QQ^T) W||_F^2 + tau ||W||_*

X: N x G standardised features (training-only weighted mean/SD), Z: N x D standardised labels,
wt: family weights normalised to sum to one, Q: G x r orthonormal basis of the pathway subspace
(column space of diag(s_gene) M), W: G x D.

tau = 0  -> generalised Ridge, solved in the dual (N x N) through the kernel
            K(lam, eta) = X_par X_par^T / lam + X_perp X_perp^T / (lam + eta).
tau > 0  -> the minimiser lies in span([X^T, Q]) (dimension m <= N + r): projecting W onto that span
            leaves the loss unchanged and cannot increase either penalty. The problem is solved for the
            m x D coefficient matrix C (W = B C, B orthonormal) by ADMM with singular-value
            soft-thresholding. A FISTA solver in the same reduced space and a direct full-space
            proximal-gradient solver exist only as references for the tests.
"""
from __future__ import annotations

import numpy as np

ZERO_SD = 1e-12


# ----------------------------------------------------------------------------------------------
# standardisation and the pathway subspace
# ----------------------------------------------------------------------------------------------

def norm_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    return w / w.sum()


def wmean_sd(A: np.ndarray, wt: np.ndarray) -> tuple:
    mu = wt @ A
    sd = np.sqrt(wt @ (A - mu) ** 2)
    return mu, sd


def standardise(A: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(A - mu) / sd; columns whose training SD is zero become exactly zero (they carry no weight)."""
    ok = sd > ZERO_SD
    out = np.zeros_like(A, dtype=float)
    out[:, ok] = (A[:, ok] - mu[ok]) / sd[ok]
    return out


def membership_std(sets: list, sd: np.ndarray) -> np.ndarray:
    """M_std = diag(s_gene) M with M the G x K averaging matrix of the gene sets.
    A linear model on the K set means of the *unstandardised* features has, in standardised
    coordinates, a weight vector in the column space of this matrix."""
    G = len(sd)
    s = np.where(sd > ZERO_SD, sd, 0.0)
    M = np.zeros((G, len(sets)))
    for k, idx in enumerate(sets):
        M[idx, k] = s[idx] / len(idx)
    return M


def orth_basis(M: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    U, sv, _ = np.linalg.svd(M, full_matrices=False)
    r = int((sv > rtol * sv[0]).sum())
    return U[:, :r]


# ----------------------------------------------------------------------------------------------
# tau = 0: dual generalised Ridge
# ----------------------------------------------------------------------------------------------

class DualRidge:
    """Weighted kernel Ridge for many lam at once: alpha(lam) = (K + lam*D*S^-1)^-1 Z."""

    def __init__(self, K: np.ndarray, wt: np.ndarray, Z: np.ndarray):
        self.sq = np.sqrt(wt)
        Ks = self.sq[:, None] * K * self.sq[None, :]
        self.e, self.V = np.linalg.eigh((Ks + Ks.T) / 2)
        self.e = np.clip(self.e, 0.0, None)
        self.VtZ = self.V.T @ (self.sq[:, None] * Z)
        self.D = Z.shape[1]

    def alpha(self, lam: float) -> np.ndarray:
        return self.sq[:, None] * (self.V @ (self.VtZ / (self.e[:, None] + lam * self.D)))


def soft_kernel(K_par: np.ndarray, K_perp: np.ndarray, ratio: float) -> np.ndarray:
    """lam * K(lam, eta) with eta = ratio * lam. ratio = inf keeps the pathway part only."""
    if np.isinf(ratio):
        return K_par
    return K_par + K_perp / (1.0 + ratio)


# ----------------------------------------------------------------------------------------------
# tau > 0: reduced-space ADMM
# ----------------------------------------------------------------------------------------------

def svt(A: np.ndarray, thr: float) -> tuple:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s2 = np.clip(s - thr, 0.0, None)
    return (U * s2) @ Vt, s2


def reduced_basis(Xs: np.ndarray, Q: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """Orthonormal basis B (G x m) of span([Xs^T, Q]) whose first r columns are Q."""
    R = Xs.T - Q @ (Q.T @ Xs.T)
    U, sv, _ = np.linalg.svd(R, full_matrices=False)
    keep = sv > rtol * max(sv[0], 1e-300)
    return np.hstack([Q, U[:, keep]])


def smooth_terms(XB: np.ndarray, wt: np.ndarray, Z: np.ndarray, r: int, lam: float, eta: float) -> tuple:
    """H and R of g(C) = tr(C^T H C) - 2 tr(C^T R) + const, in a basis whose first r columns span Q."""
    D = Z.shape[1]
    m = XB.shape[1]
    pen = np.full(m, lam + eta)
    pen[:r] = lam
    H = XB.T @ (wt[:, None] * XB) / D + np.diag(pen)
    Rm = XB.T @ (wt[:, None] * Z) / D
    return H, Rm


def tau_max(XB: np.ndarray, wt: np.ndarray, Z: np.ndarray) -> float:
    """Smallest tau for which W = 0 is optimal: operator norm of the gradient of the smooth part at 0."""
    D = Z.shape[1]
    return float(np.linalg.norm(2.0 * XB.T @ (wt[:, None] * Z) / D, 2))


def objective(C, H, Rm, tau, const=0.0) -> float:
    nuc = np.linalg.svd(C, compute_uv=False).sum() if tau > 0 else 0.0
    return float(np.sum(C * (H @ C)) - 2.0 * np.sum(C * Rm) + const + tau * nuc)


def admm_nuclear(H: np.ndarray, Rm: np.ndarray, tau: float, C0: np.ndarray | None = None,
                 tol: float = 1e-7, max_iter: int = 20000, eig: tuple | None = None) -> tuple:
    """min_C tr(C^T H C) - 2 tr(C^T R) + tau ||C||_*  (H symmetric positive definite).
    Returns (E, info); E is the thresholded (exactly low-rank) iterate."""
    e, V = eig if eig is not None else np.linalg.eigh(H)
    m, D = Rm.shape
    rho = 2.0 * float(np.sqrt(e[0] * e[-1]))
    E = np.zeros((m, D)) if C0 is None else C0.copy()
    U = np.zeros((m, D))
    VtR2 = 2.0 * (V.T @ Rm)
    scale = np.sqrt(m * D)
    info = dict(converged=False, iterations=max_iter)
    for it in range(1, max_iter + 1):
        C = V @ ((VtR2 + rho * (V.T @ (E - U))) / (2.0 * e[:, None] + rho))
        E_old = E
        E, sv = svt(C + U, tau / rho)
        U = U + C - E
        r_pri = np.linalg.norm(C - E)
        r_dual = rho * np.linalg.norm(E - E_old)
        eps_pri = scale * 1e-12 + tol * max(np.linalg.norm(C), np.linalg.norm(E))
        eps_dual = scale * 1e-12 + tol * rho * np.linalg.norm(U)
        if r_pri <= eps_pri and r_dual <= eps_dual:
            info.update(converged=True, iterations=it)
            break
        if it % 25 == 0:                       # residual balancing
            if r_pri > 10 * r_dual:
                rho *= 2.0
                U /= 2.0
            elif r_dual > 10 * r_pri:
                rho /= 2.0
                U *= 2.0
    info["rank"] = int((sv > 0).sum())
    info["top_singular_values"] = sv[:5].tolist()
    return E, info


def fista_nuclear(H: np.ndarray, Rm: np.ndarray, tau: float, max_iter: int = 200000, tol: float = 1e-12) -> np.ndarray:
    """Reference solver (accelerated proximal gradient with restart) for the tests."""
    L = 2.0 * float(np.linalg.eigvalsh(H)[-1])
    C = np.zeros_like(Rm)
    Yk, t = C.copy(), 1.0
    for _ in range(max_iter):
        grad = 2.0 * (H @ Yk - Rm)
        C_new, _ = svt(Yk - grad / L, tau / L)
        if np.sum((Yk - C_new) * (C_new - C)) > 0:      # gradient restart
            t = 1.0
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        Yk = C_new + ((t - 1.0) / t_new) * (C_new - C)
        done = np.linalg.norm(C_new - C) <= tol * max(1.0, np.linalg.norm(C_new))
        C, t = C_new, t_new
        if done:
            break
    return C


def full_space_reference(Xs, wt, Z, Q, lam, eta, tau, max_iter=200000, tol=1e-13) -> np.ndarray:
    """Proximal gradient directly on the G x D matrix W of the stated objective (tests only; small G)."""
    N, G = Xs.shape
    D = Z.shape[1]
    A = Xs.T @ (wt[:, None] * Xs) / D
    P = (lam + eta) * np.eye(G) - eta * Q @ Q.T
    Hf = A + P
    Rf = Xs.T @ (wt[:, None] * Z) / D
    if tau == 0:
        return np.linalg.solve(Hf, Rf)
    return fista_nuclear(Hf, Rf, tau, max_iter=max_iter, tol=tol)


def full_objective(W, Xs, wt, Z, Q, lam, eta, tau) -> float:
    """The objective exactly as written in the plan (used to compare solvers)."""
    D = Z.shape[1]
    res = Xs @ W - Z
    loss = float(np.sum(wt[:, None] * res ** 2) / D)
    perp = W - Q @ (Q.T @ W)
    nuc = np.linalg.svd(W, compute_uv=False).sum() if tau > 0 else 0.0
    return loss + lam * float(np.sum(W ** 2)) + eta * float(np.sum(perp ** 2)) + tau * nuc
