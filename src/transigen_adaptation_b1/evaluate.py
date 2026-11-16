"""The single evaluator. Every model and baseline is scored here from condition-level deltas."""
import numpy as np
import pandas as pd


def _rowcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(1) / den
    r[den < 1e-12] = np.nan          # constant or zero prediction: correlation undefined, not zero
    return r


def retrieval_ranks(pred, truth):
    """Rank of each condition's own truth among all truths by cosine to its prediction (1 = best).

    Ties receive the mean rank. A zero-norm prediction ties with everything.
    """
    pn = np.linalg.norm(pred, axis=1, keepdims=True)
    tn = np.linalg.norm(truth, axis=1, keepdims=True)
    cos = (pred @ truth.T) / np.maximum(pn, 1e-12) / np.maximum(tn.T, 1e-12)
    cos[pn[:, 0] < 1e-12] = 0.0
    cos = np.round(cos, 10)
    own = np.diag(cos)[:, None]
    better = (cos > own).sum(1)
    equal = (cos == own).sum(1)       # includes itself
    return better + (equal + 1) / 2.0


def per_condition(pred, truth, top=20):
    """pred, truth: (n_conditions, 978) deltas of one target and one scored set."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    assert pred.shape == truth.shape and pred.shape[1] == 978 and np.isfinite(pred).all()
    n = len(truth)
    out = dict(delta_rmse=np.sqrt(((pred - truth) ** 2).mean(1)), delta_pearson=_rowcorr(pred, truth))
    c = truth.mean(0, keepdims=True)                       # scoring-only centring vector
    out["centred_pearson"] = _rowcorr(pred - c, truth - c)
    ti = np.argsort(-np.abs(truth), axis=1, kind="stable")[:, :top]
    pi = np.argsort(-np.abs(pred), axis=1, kind="stable")[:, :top]
    st = np.sign(np.take_along_axis(truth, ti, 1))
    sp = np.sign(np.take_along_axis(pred, ti, 1))
    out["top20_direction"] = (st == sp).mean(1)
    out["top20_recall"] = np.array([len(set(ti[i]) & set(pi[i])) / top for i in range(n)])
    rk = retrieval_ranks(pred, truth)
    out["retrieval_rank"] = rk
    out["retrieval_candidates"] = np.full(n, n)
    out["truth_rms"] = np.sqrt((truth ** 2).mean(1))
    out["pred_rms"] = np.sqrt((pred ** 2).mean(1))
    return pd.DataFrame(out)


def magnitude_ratio(pred, truth):
    """Per-gene RMS of predicted response over per-gene RMS of measured response (median over genes), and the global ratio."""
    pg = np.sqrt((np.asarray(pred, float) ** 2).mean(0))
    tg = np.sqrt((np.asarray(truth, float) ** 2).mean(0))
    return float(np.median(pg / np.maximum(tg, 1e-12))), float(np.sqrt((np.asarray(pred, float) ** 2).mean()) / np.sqrt((np.asarray(truth, float) ** 2).mean()))


def shuffled_retrieval_reference(pred, truth, n_perm=200, seed=20261116):
    """Retrieval when the drug labels of the predictions are shuffled (fixed seed)."""
    rs = np.random.RandomState(seed)
    mrr, top5 = [], []
    for _ in range(n_perm):
        p = rs.permutation(len(pred))
        rk = retrieval_ranks(np.asarray(pred)[p], truth)
        mrr.append((1 / rk).mean())
        top5.append((rk <= 5).mean())
    return float(np.mean(mrr)), float(np.mean(top5))


def summarise(pc, weights=None):
    """Mean of per-condition scores (optionally bootstrap-weighted)."""
    w = np.ones(len(pc)) if weights is None else np.asarray(weights, float)
    def m(v):
        v = np.asarray(v, float)
        ok = np.isfinite(v) & (w > 0)
        return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else float("nan")
    return dict(delta_rmse=m(pc.delta_rmse), delta_pearson=m(pc.delta_pearson), centred_pearson=m(pc.centred_pearson),
                top20_direction=m(pc.top20_direction), top20_recall=m(pc.top20_recall),
                retrieval_mrr=m(1 / pc.retrieval_rank), retrieval_top5=m((pc.retrieval_rank <= 5).astype(float)),
                n_conditions=int(len(pc)), n_pearson_defined=int(np.isfinite(pc.delta_pearson).sum()))
