"""Metrics. One primary number plus the three secondaries the plan allows; nothing is fitted here.

macro_delta_rmse : per-condition RMSE over genes -> mean over that line's drugs -> mean over lines.
drug_centred_r   : within a cell line, the mean response over the evaluated drugs is removed from
                   truth and prediction alike, then the per-drug gene vectors are correlated. This
                   separates a model that only reproduces the shared average response.
top20            : on the 20 genes with the largest observed |Delta|, sign agreement and the overlap
                   with the model's own top 20. These are large-change genes, not tested DEGs.
"""
from __future__ import annotations

import numpy as np


def per_condition_rmse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(((pred - truth) ** 2).mean(-1))


def macro_from_rmse(rmse: np.ndarray, line: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean over drugs inside each line, then an unweighted mean over the lines present."""
    ls = np.unique(line)
    per_line = np.array([rmse[line == l].mean() for l in ls])
    return float(per_line.mean()), ls, per_line


def macro_delta_rmse(pred, truth, line) -> float:
    return macro_from_rmse(per_condition_rmse(pred, truth), line)[0]


def drug_centred_correlation(pred, truth, line):
    """Returns (mean r, per-condition r with NaN where undefined, n_undefined)."""
    r = np.full(len(line), np.nan)
    for l in np.unique(line):
        m = line == l
        if m.sum() < 2:
            continue                      # centring needs at least two drugs in the line
        p = pred[m] - pred[m].mean(0)
        t = truth[m] - truth[m].mean(0)
        ps, ts = p.std(1), t.std(1)
        good = (ps > 0) & (ts > 0)
        rr = np.full(m.sum(), np.nan)
        if good.any():
            pc = (p[good] - p[good].mean(1, keepdims=True))
            tc = (t[good] - t[good].mean(1, keepdims=True))
            num = (pc * tc).sum(1)
            den = np.sqrt((pc ** 2).sum(1) * (tc ** 2).sum(1))
            rr[good] = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
        r[m] = rr
    return float(np.nanmean(r)) if np.isfinite(r).any() else float("nan"), r, int((~np.isfinite(r)).sum())


def top20_direction_and_recall(pred, truth, k: int = 20):
    idx_t = np.argsort(-np.abs(truth), axis=1)[:, :k]
    idx_p = np.argsort(-np.abs(pred), axis=1)[:, :k]
    rows = np.arange(len(truth))[:, None]
    st = np.sign(truth[rows, idx_t])
    sp = np.sign(pred[rows, idx_t])
    direction = (st == sp).mean(1)
    recall = np.array([len(np.intersect1d(a, b)) / k for a, b in zip(idx_t, idx_p)])
    return direction, recall


def line_macro_table(pred, truth, line, line_names):
    rmse = per_condition_rmse(pred, truth)
    _, r, _ = drug_centred_correlation(pred, truth, line)
    d, rc = top20_direction_and_recall(pred, truth)
    out = []
    for l in np.unique(line):
        m = line == l
        out.append(dict(line_index=int(l), cell_line=str(line_names[l]), n_drugs=int(m.sum()),
                        delta_rmse=float(rmse[m].mean()),
                        drug_centred_r=float(np.nanmean(r[m])) if np.isfinite(r[m]).any() else float("nan"),
                        top20_direction=float(d[m].mean()), top20_recall=float(rc[m].mean())))
    return out


def all_metrics(pred, truth, line, line_names, snu423: int | None):
    rmse = per_condition_rmse(pred, truth)
    macro, ls, per_line = macro_from_rmse(rmse, line)
    rmean, r, n_undef = drug_centred_correlation(pred, truth, line)
    d, rc = top20_direction_and_recall(pred, truth)
    out = dict(macro_delta_rmse=macro, n_conditions=int(len(line)), n_lines=int(len(ls)),
               drug_centred_r=rmean, drug_centred_r_undefined=n_undef,
               top20_direction=float(d.mean()), top20_recall=float(rc.mean()))
    if snu423 is not None and (line == snu423).any():
        m = line == snu423
        out.update(snu423_delta_rmse=float(rmse[m].mean()),
                   snu423_drug_centred_r=float(np.nanmean(r[m])) if np.isfinite(r[m]).any() else float("nan"),
                   snu423_top20_direction=float(d[m].mean()), snu423_top20_recall=float(rc[m].mean()),
                   snu423_n_drugs=int(m.sum()))
    return out, rmse


def paired_bootstrap(rmse_a, rmse_b, drug, n: int = 2000, seed: int = 0, line=None):
    """Resample drug groups, carrying every cell line of a drug together (plan s8.6).

    Conditional on the trained models: it describes sampling of drugs, not training randomness.
    Returns the macro difference (a - b) and its percentile interval.
    """
    rng = np.random.default_rng(seed)
    drugs = np.unique(drug)
    by_drug = {d: np.where(drug == d)[0] for d in drugs}
    lines = np.unique(line) if line is not None else None

    def macro(sel_idx, lin):
        if lin is None:
            return rmse_a[sel_idx].mean(), rmse_b[sel_idx].mean()
        a = [rmse_a[sel_idx][lin == l].mean() for l in np.unique(lin)]
        b = [rmse_b[sel_idx][lin == l].mean() for l in np.unique(lin)]
        return np.mean(a), np.mean(b)

    obs_a, obs_b = macro(np.arange(len(rmse_a)), line)
    diffs = np.empty(n)
    for i in range(n):
        pick = rng.choice(drugs, len(drugs), replace=True)
        idx = np.concatenate([by_drug[d] for d in pick])
        lin = line[idx] if line is not None else None
        a, b = macro(idx, lin)
        diffs[i] = a - b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rel = np.percentile(diffs / np.maximum(obs_a, 1e-12) * 100, [2.5, 97.5])
    return dict(macro_a=float(obs_a), macro_b=float(obs_b), diff=float(obs_a - obs_b),
                diff_ci=[float(lo), float(hi)], rel_improvement_pct=float((obs_a - obs_b) / obs_a * 100),
                rel_improvement_ci_pct=[float(rel[0]), float(rel[1])], n_boot=n, n_drugs=int(len(drugs)))
