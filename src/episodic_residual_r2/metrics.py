"""R2 evaluator. One row per scored unit (target, drug_group); all methods are scored here."""
import numpy as np
import pandas as pd

from src.transigen_adaptation_b1.evaluate import _rowcorr, retrieval_ranks


def per_unit(pred, truth, top=20):
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    assert pred.shape == truth.shape and pred.shape[1] == 978 and np.isfinite(pred).all()
    out = dict(delta_rmse=np.sqrt(((pred - truth) ** 2).mean(1)), delta_pearson=_rowcorr(pred, truth))
    pc, tc = pred - pred.mean(0, keepdims=True), truth - truth.mean(0, keepdims=True)   # each side loses its OWN mean over the scored units
    out["centred_rmse"] = np.sqrt(((pc - tc) ** 2).mean(1))
    out["centred_pearson"] = _rowcorr(pc, tc)
    ti = np.argsort(-np.abs(truth), axis=1, kind="stable")[:, :top]
    out["top20_direction"] = (np.sign(np.take_along_axis(truth, ti, 1)) == np.sign(np.take_along_axis(pred, ti, 1))).mean(1)
    out["retrieval_rank"] = retrieval_ranks(pred, truth)
    out["retrieval_rank_centred"] = retrieval_ranks(pc, tc)
    out["truth_rms"] = np.sqrt((truth ** 2).mean(1))
    out["pred_rms"] = np.sqrt((pred ** 2).mean(1))
    return pd.DataFrame(out)


SUMMARY = ["delta_rmse", "delta_pearson", "centred_rmse", "centred_pearson", "top20_direction", "retrieval_mrr", "retrieval_mrr_centred", "magnitude_ratio"]


def summarise(pu, w=None):
    w = np.ones(len(pu)) if w is None else np.asarray(w, float)

    def m(v):
        v = np.asarray(v, float)
        ok = np.isfinite(v) & (w > 0)
        return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
    return dict(delta_rmse=m(pu.delta_rmse), delta_pearson=m(pu.delta_pearson), centred_rmse=m(pu.centred_rmse), centred_pearson=m(pu.centred_pearson),
                top20_direction=m(pu.top20_direction), retrieval_mrr=m(1 / pu.retrieval_rank), retrieval_mrr_centred=m(1 / pu.retrieval_rank_centred),
                magnitude_ratio=m(pu.pred_rms) / max(m(pu.truth_rms), 1e-12), n_units=int(len(pu)))


def expected_random_mrr(n):
    """Expected reciprocal rank of a uniformly random rank among n candidates."""
    return float(np.mean(1.0 / np.arange(1, n + 1)))
