"""Follow-up metrics (numpy only). Sign convention everywhere: candidate - comparator, negative = better.

Total error (unchanged primary of R6)
    per-condition RMSE over the genes -> mean over the conditions of a drug group -> unweighted
    mean over drug groups.

Within-pool contrast RMSE  L_within  (plan 5.2; the specificity metric)
    group = the evaluated drugs that share one control pool (and one source fold). Truth and
    prediction are each centred by THEIR OWN mean over the drugs of the group (equal weight per
    drug), then
        l_i = sqrt( mean_g ( pred_c[i, g] - truth_c[i, g] )^2 )
        L_within = mean over drug groups of the mean of l_i over the drug's conditions.
    A group with fewer than two distinct drugs contributes nothing. With e = pred - truth the
    centred error is e_i - mean_group(e), so only the Gram matrix K = E E^T / G of each group is
    needed; a drug-level bootstrap replicate with multiplicities w re-centres with weights w:
        l_i(w)^2 = K_ii - 2 (K w)_i / W + w^T K w / W^2 ,   W = sum(w).
    Subtracting one and the same vector from both sides (e.g. a training pool mean) does not change
    e and is therefore NOT a specificity metric; it is never reported as one.

Retrieval with ties (plan appendix B)
    a = candidates scoring strictly higher than the right one, t = candidates tied with it
    (itself included), N = candidates:
        top1 = [a == 0] / t ;  topk = max(0, min(t, k - a)) / t ;  mrr = mean_{j=1..t} 1 / (a + j)
    chance: 1/N, min(k, N)/N, H_N / N. A zero-norm (constant) centred prediction ties with every
    candidate. The original R6 definitions are kept under their own names:
        legacy_rank_score = 1 - (a + (t - 1)/2) / (N - 1) ;  legacy_unique_top1 = [a + (t - 1)/2 == 0].

Uncertainty: paired bootstrap over drug structure groups, each carrying all of its cell lines.
Per-condition values are averaged over training seeds and thinning seeds first.
"""
from __future__ import annotations

import numpy as np

OPERATING_REL = 0.01      # the pre-existing R6 operating criterion (1% relative); not a biological threshold


def cond_rmse(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((P.astype(np.float64) - T) ** 2, axis=-1))


def drug_macro(v: np.ndarray, drug: np.ndarray) -> float:
    ok = ~np.isnan(v)
    u, inv = np.unique(drug[ok], return_inverse=True)
    return float((np.bincount(inv, weights=v[ok]) / np.bincount(inv)).mean())


def groups_of(*keys) -> list[np.ndarray]:
    """Index arrays of the rows sharing the same tuple of keys."""
    k = np.stack([np.asarray(x) for x in keys], 1)
    _, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    cut = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, cut)


def centre_by(A: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Each row minus the mean of its group (rows of a group are distinct drugs: equal drug weight)."""
    out = A.astype(np.float64).copy()
    for idx in groups:
        out[idx] -= out[idx].mean(0)
    return out


# ------------------------------------------------------------------------------------------------
# within-pool contrast
# ------------------------------------------------------------------------------------------------
def grams(E: np.ndarray, groups: list[np.ndarray]) -> list[np.ndarray]:
    G = E.shape[1]
    out = []
    for idx in groups:
        e = E[idx].astype(np.float64)
        out.append(e @ e.T / G)
    return out


def within_point(gr: list[np.ndarray], groups: list[np.ndarray], n: int) -> np.ndarray:
    """l_i for every condition (nan where the group has fewer than two drugs)."""
    out = np.full(n, np.nan)
    for K, idx in zip(gr, groups):
        m = len(idx)
        if m < 2:
            continue
        a = np.diag(K) - 2.0 * K.sum(1) / m + K.sum() / m ** 2
        out[idx] = np.sqrt(np.clip(a, 0.0, None))
    return out


def within_boot(gr_list: list[list[np.ndarray]], groups: list[np.ndarray], dinv: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-centred l_i under drug-level resampling.

    gr_list: Gram lists, one per (training seed, thinning seed); M [B, n_drug] multiplicities.
    Returns (Lacc [B, n] mean l over the seed-views, 0 where invalid;  V [B, n] validity 0/1).
    A group is valid in a replicate when at least two distinct drugs of it were drawn.
    """
    B, n = M.shape[0], len(dinv)
    Lacc = np.zeros((B, n), np.float32); V = np.zeros((B, n), np.float32)
    for gi, idx in enumerate(groups):
        if len(idx) < 2:
            continue
        W = M[:, dinv[idx]].astype(np.float64)                      # [B, m]
        ok = (W > 0).sum(1) >= 2
        Ws = np.maximum(W.sum(1), 1.0)[:, None]
        acc = np.zeros_like(W)
        for gr in gr_list:
            K = gr[gi]
            KW = W @ K
            a = np.diag(K)[None] - 2.0 * KW / Ws + ((KW * W).sum(1)[:, None]) / Ws ** 2
            acc += np.sqrt(np.clip(a, 0.0, None))
        acc /= len(gr_list)
        Lacc[:, idx] = (acc * ok[:, None]).astype(np.float32)
        V[:, idx] = ok[:, None].astype(np.float32)
    return Lacc, V


def pair_contrast_from_gram(K: np.ndarray) -> float:
    """sqrt of the mean over unordered pairs of |e_d - e_e|^2 / G  (appendix A identity)."""
    m = K.shape[0]
    cen = (np.trace(K) - K.sum() / m) / m                           # mean_d |e_d - e_bar|^2 / G
    return float(np.sqrt(max(2.0 * m / (m - 1) * cen, 0.0)))


# ------------------------------------------------------------------------------------------------
# retrieval
# ------------------------------------------------------------------------------------------------
def _unit(A: np.ndarray):
    A = A - A.mean(1, keepdims=True)
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    return np.where(nrm > 1e-12, A / np.maximum(nrm, 1e-12), 0.0), nrm[:, 0]


def retrieval_counts(Pc: np.ndarray, Tc: np.ndarray, groups: list[np.ndarray]):
    """Per query: (a, t, N). Candidates of a query = the measured responses Tc of its group (fixed)."""
    n = len(Pc)
    a = np.zeros(n, int); t = np.ones(n, int); N = np.ones(n, int)
    for idx in groups:
        m = len(idx)
        N[idx] = m
        if m < 2:
            continue
        q, _ = _unit(Pc[idx].astype(np.float64))
        c, _ = _unit(Tc[idx].astype(np.float64))
        S = q @ c.T
        dg = np.diag(S)[:, None]
        a[idx] = (S > dg).sum(1)
        t[idx] = (S == dg).sum(1)
    return a, t, N


_H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, 100001))])     # harmonic numbers H_0..H_100000


def retrieval_scores(a: np.ndarray, t: np.ndarray, N: np.ndarray, k: int = 5) -> dict:
    """Tie-corrected and legacy scores per query; nan where the candidate set has fewer than two."""
    ok = N >= 2
    f = lambda x: np.where(ok, x, np.nan)
    r = a + 0.5 * (t - 1)
    return dict(
        top1=f((a == 0) / t), topk=f(np.clip(np.minimum(t, k - a), 0, None) / t), mrr=f((_H[a + t] - _H[a]) / t),
        chance_top1=f(1.0 / N), chance_topk=f(np.minimum(k, N) / N), chance_mrr=f(_H[N] / N),
        legacy_rank_score=f(1.0 - r / np.maximum(N - 1, 1)), legacy_unique_top1=f((r == 0).astype(float)),
        n_candidates=f(N.astype(float)))


# ------------------------------------------------------------------------------------------------
# drug-level bootstrap
# ------------------------------------------------------------------------------------------------
def multiplicities(n_drug: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_drug, size=(n_boot, n_drug))
    M = np.zeros((n_boot, n_drug), np.int16)
    for b in range(n_boot):
        M[b] = np.bincount(idx[b], minlength=n_drug)
    return M


def drug_indicator(dinv: np.ndarray, n_drug: int, mask: np.ndarray | None = None) -> np.ndarray:
    A = np.zeros((len(dinv), n_drug), np.float32)
    A[np.arange(len(dinv)), dinv] = 1.0
    if mask is not None:
        A *= mask[:, None]
    return A


def macro_point(v: np.ndarray, A: np.ndarray) -> tuple[float, int, int]:
    """Drug-macro mean of per-condition values under indicator A (nan values are skipped)."""
    ok = (~np.isnan(v)).astype(np.float32)
    num = np.nan_to_num(v).astype(np.float64) @ A
    cnt = ok @ A
    has = cnt > 0
    return float((num[has] / cnt[has]).mean()) if has.any() else float("nan"), int(has.sum()), int(cnt.sum())


def macro_boot_fixed(v: np.ndarray, A: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Bootstrap of the drug-macro mean of fixed per-condition values."""
    ok = (~np.isnan(v)).astype(np.float32)
    num = np.nan_to_num(v).astype(np.float64) @ A
    cnt = ok @ A
    has = cnt > 0
    dm = np.where(has, num / np.maximum(cnt, 1), 0.0)
    Mf = M.astype(np.float64)
    den = Mf @ has.astype(np.float64)
    return np.where(den > 0, (Mf @ dm) / np.maximum(den, 1), np.nan)


def macro_boot_varying(Lacc: np.ndarray, V: np.ndarray, A: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Bootstrap of the drug-macro mean when the per-condition values depend on the replicate."""
    num = Lacc @ A                                                  # [B, n_drug]
    cnt = V @ A
    has = cnt > 0
    dm = np.where(has, num / np.maximum(cnt, 1), 0.0)
    Mf = M.astype(np.float64)
    den = (Mf * has).sum(1)
    return np.where(den > 0, (Mf * dm).sum(1) / np.maximum(den, 1), np.nan)


def ci(bs: np.ndarray) -> tuple[float, float]:
    bs = bs[~np.isnan(bs)]
    return (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))) if len(bs) else (float("nan"), float("nan"))


def decision_category(diff: float, lo: float, hi: float, comparator: float, rel_op: float = OPERATING_REL) -> str:
    """Plan 3.4 categories for candidate - comparator (negative = better); 1% is the operating value."""
    if not np.isfinite(diff) or comparator == 0:
        return "undefined"
    rel, lo_rel = diff / comparator, lo / comparator
    if hi < 0:
        return "improved_operational" if rel <= -rel_op else "improved_small"
    if lo > 0:
        return "worse_operational" if rel >= rel_op else "worse_small"
    return "uncertain" if lo_rel <= -rel_op else "no_large_gain"
