"""Scoring of condition-level delta predictions against the fixed full-observation truth.

Primary: per-condition RMSE over the 2,000 genes -> mean over the cell lines of a drug group ->
unweighted mean over drug groups (every drug group has the same weight). Uncertainty: paired
bootstrap over drug groups, each carrying all of its cell lines (the cluster unit). Training seeds
and thinning seeds are averaged per condition before resampling; they are not independent samples.

Secondary read-outs (never used to select anything): own-centred delta Pearson, top-20
large-change direction, predicted magnitude, fixed-candidate retrieval.
"""
from __future__ import annotations

import numpy as np

SNU423 = "CVCL_0366"
HEPG2 = "CVCL_1098"


def cond_rmse(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((P.astype(np.float64) - T) ** 2, axis=-1))


def drug_macro(v: np.ndarray, drug: np.ndarray) -> float:
    """Mean within drug group, then unweighted mean over drug groups."""
    u, inv = np.unique(drug, return_inverse=True)
    s = np.bincount(inv, weights=v) / np.bincount(inv)
    return float(s.mean())


def per_drug(v: np.ndarray, drug: np.ndarray):
    u, inv = np.unique(drug, return_inverse=True)
    return u, np.bincount(inv, weights=v) / np.bincount(inv)


def paired_bootstrap(va: np.ndarray, vb: np.ndarray, drug: np.ndarray, n: int = 5000, seed: int = 0) -> dict:
    """a - b on the primary statistic, resampling drug groups with replacement."""
    _, da = per_drug(va, drug)
    _, db = per_drug(vb, drug)
    diff = da - db
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    bs = diff[idx].mean(1)
    return dict(diff=float(diff.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                rel=float(diff.mean() / db.mean()), n_drugs=int(len(diff)),
                frac_drugs_a_better=float((diff < 0).mean()))


def bootstrap_mean(v: np.ndarray, drug: np.ndarray, n: int = 5000, seed: int = 0) -> dict:
    _, d = per_drug(v, drug)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return dict(mean=float(d.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)))


def _centre_by_line(A: np.ndarray, line: np.ndarray) -> np.ndarray:
    out = A.astype(np.float64).copy()
    for l in np.unique(line):
        m = line == l
        out[m] -= out[m].mean(0)
    return out


def _rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.where(den > 1e-12, (a * b).sum(1) / np.maximum(den, 1e-12), 0.0)


def secondaries(P: np.ndarray, T: np.ndarray, line: np.ndarray, drug: np.ndarray) -> dict:
    """Per-condition secondary vectors (dict of arrays aligned with the conditions)."""
    Pc, Tc = _centre_by_line(P, line), _centre_by_line(T, line)   # each side minus its OWN line mean
    out = dict(pearson=_rowcorr(P.astype(np.float64), T.astype(np.float64)), centred_pearson=_rowcorr(Pc, Tc))
    k = np.argsort(-np.abs(T), axis=1)[:, :20]                     # scoring-only gene set
    out["top20_direction"] = (np.sign(np.take_along_axis(P, k, 1)) == np.sign(np.take_along_axis(T, k, 1))).mean(1)
    out["magnitude_ratio"] = np.sqrt((P.astype(np.float64) ** 2).mean(1)) / np.maximum(np.sqrt((T.astype(np.float64) ** 2).mean(1)), 1e-12)
    # fixed-candidate retrieval: within a cell line, candidates are the measured centred responses
    # of all evaluated drugs; the rank of the right one by correlation with the centred prediction
    rr = np.zeros(len(P)); top1 = np.zeros(len(P))
    for l in np.unique(line):
        m = np.flatnonzero(line == l)
        if len(m) < 2:
            rr[m] = np.nan; top1[m] = np.nan
            continue
        a = Pc[m] - Pc[m].mean(1, keepdims=True); b = Tc[m] - Tc[m].mean(1, keepdims=True)
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        S = a @ b.T
        r = (S > np.diag(S)[:, None]).sum(1)                       # 0 = best
        rr[m] = 1.0 - r / (len(m) - 1)                             # 1 = right candidate ranked first, 0.5 = chance
        top1[m] = (r == 0)
    out["retrieval_rank_score"] = rr
    out["retrieval_top1"] = top1
    return out


def summarise(P: np.ndarray, T: np.ndarray, line: np.ndarray, drug: np.ndarray) -> dict:
    v = cond_rmse(P, T)
    s = secondaries(P, T, line, drug)
    out = dict(delta_rmse=drug_macro(v, drug), n_conditions=int(len(v)), n_drugs=int(len(np.unique(drug))))
    for k, a in s.items():
        ok = ~np.isnan(a)
        out[k] = drug_macro(a[ok], drug[ok])
    return out
