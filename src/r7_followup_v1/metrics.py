"""Retrieval with explicit tie handling (plan Appendix B) next to the legacy definition of the original evaluator.

Query = a prediction; candidates = the measured (cell-centred) responses of the scored units of the same cell
(the candidate pool is identical for every model). a = number of candidates scoring strictly higher than the correct
one, t = number of candidates tied with it (the correct one included). Possible ranks a+1 .. a+t, each equally likely:
    top1 = 1[a == 0] / t          topk = max(0, min(t, k - a)) / t          mrr = (1/t) * sum_{j=1..t} 1/(a+j)
All N candidates tied: top1 = 1/N, topk = min(k, N)/N, mrr = H_N / N.
The legacy value `legacy_midrank_mrr` = 1 / (a + (t - 1)/2 + 1) is kept under its own name; it is NOT an expected MRR.
A zero-norm centred prediction scores 0 against every candidate and is therefore an all-tie query in both definitions.
numpy only."""
import numpy as np


def _harm(n):
    return np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 1))])


def rank_counts(sim, correct=None):
    """sim: (n_query, n_cand). correct[i] = column of the correct candidate (default: the diagonal)."""
    sim = np.asarray(sim, np.float64)
    c = np.arange(sim.shape[0]) if correct is None else np.asarray(correct)
    d = sim[np.arange(sim.shape[0]), c][:, None]
    a = (sim > d).sum(1)
    t = (sim == d).sum(1)
    return a, t


def tie_corrected(a, t, ks=(1, 5)):
    a = np.asarray(a); t = np.asarray(t)
    H = _harm(int((a + t).max()) if len(a) else 1)
    out = {"mrr_tie_expected": (H[a + t] - H[a]) / t, "legacy_midrank_mrr": 1.0 / (a + 0.5 * (t - 1) + 1)}
    for k in ks:
        out[f"top{k}_tie_expected"] = np.maximum(0, np.minimum(t, k - a)) / t
    out["legacy_unique_top1"] = ((a == 0) & (t == 1)).astype(float)
    return out


def chance(n, ks=(1, 5)):
    out = {"mrr_tie_expected": _harm(n)[n] / n, "legacy_midrank_mrr": 1.0 / (0.5 * (n - 1) + 1)}
    for k in ks:
        out[f"top{k}_tie_expected"] = min(k, n) / n
    return out


def centred_cosine(pred, true):
    """Rows: predictions, columns: candidate truths; both centred by their own mean over the pool (scoring only)."""
    p = pred - pred.mean(0); t = true - true.mean(0)
    a = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)
    b = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def retrieval_by_pool(pred, true, pool, ks=(1, 5)):
    """Per-unit retrieval values; pool = label defining the candidate set (cell line, or cell line x fold)."""
    pred = np.asarray(pred, np.float64); true = np.asarray(true, np.float64); pool = np.asarray(pool)
    keys = ["mrr_tie_expected", "legacy_midrank_mrr", "legacy_unique_top1"] + [f"top{k}_tie_expected" for k in ks]
    out = {k: np.zeros(len(pred)) for k in keys}
    out["n_candidates"] = np.zeros(len(pred), int); out["n_tied"] = np.zeros(len(pred), int)
    for c in np.unique(pool):
        s = np.where(pool == c)[0]
        a, t = rank_counts(centred_cosine(pred[s], true[s]))
        r = tie_corrected(a, t, ks)
        for k in keys:
            out[k][s] = r[k]
        out["n_candidates"][s] = len(s); out["n_tied"][s] = t
    return out
