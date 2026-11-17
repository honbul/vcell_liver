"""Scoring copied unchanged from the R4 evaluator (per-unit metrics, family-equal aggregation, paired drug-group bootstrap)."""
import numpy as np
import pandas as pd

from .common import HCC


def unit_metrics(pred, true, cells):
    """pred, true: (n, 978). cells: cell label per row. Returns a frame of per-unit metrics."""
    pred = np.asarray(pred, np.float64); true = np.asarray(true, np.float64)
    rmse = np.sqrt(((pred - true) ** 2).mean(1))
    pc = pred - pred.mean(1, keepdims=True); tc = true - true.mean(1, keepdims=True)
    pear = (pc * tc).sum(1) / np.sqrt((pc ** 2).sum(1) * (tc ** 2).sum(1) + 1e-12)
    # centred by each side's own cell mean over the scored units (scoring only)
    cells = np.asarray(cells)
    pcen = pred.copy(); tcen = true.copy()
    mrr = np.zeros(len(pred)); ncand = np.zeros(len(pred), int)
    for c in np.unique(cells):
        s = cells == c
        pcen[s] -= pred[s].mean(0); tcen[s] -= true[s].mean(0)
        a = pcen[s] / (np.linalg.norm(pcen[s], axis=1, keepdims=True) + 1e-12)
        b = tcen[s] / (np.linalg.norm(tcen[s], axis=1, keepdims=True) + 1e-12)
        sim = a @ b.T                                   # row: prediction, column: candidate truth
        d = np.diag(sim)[:, None]
        # mid-rank for ties: a prediction that is constant within the cell (all candidates tie) gets the middle rank, not rank 1
        rank = (sim > d).sum(1) + 0.5 * ((sim == d).sum(1) - 1) + 1
        mrr[s] = 1.0 / rank; ncand[s] = s.sum()
    a = pcen - pcen.mean(1, keepdims=True); b = tcen - tcen.mean(1, keepdims=True)
    cpear = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1) + 1e-12)
    top = np.argsort(-np.abs(true), 1)[:, :20]
    dirn = (np.sign(np.take_along_axis(pred, top, 1)) == np.sign(np.take_along_axis(true, top, 1))).mean(1)
    rms_p = np.sqrt((pred ** 2).mean(1)); rms_t = np.sqrt((true ** 2).mean(1))
    return pd.DataFrame(dict(rmse=rmse, pearson=pear, centred_pearson=cpear, mrr=mrr, n_candidates=ncand, top20_direction=dirn,
                             rms_pred=rms_p, rms_true=rms_t))


def family_mean(values, families, weights=None):
    """Mean over units within family -> equal mean over families (families with zero weight dropped)."""
    df = pd.DataFrame(dict(v=values, f=families, w=1.0 if weights is None else weights))
    df = df[df.w > 0]
    g = df.assign(vw=df.v * df.w).groupby("f")
    return float((g.vw.sum() / g.w.sum()).mean())


def summarise(metrics, units, panel="source"):
    """metrics aligned with units (frame with family, cell, is_hcc). panel: source (non-HCC families) / an HCC cell / hcc_mean."""
    out = {}
    if panel == "source":
        s = units.is_hcc.to_numpy() == 0
        for k in ["rmse", "pearson", "centred_pearson", "mrr", "top20_direction", "rms_pred", "rms_true"]:
            out[k] = family_mean(metrics[k].to_numpy()[s], units.family.to_numpy()[s])
        out["n_units"] = int(s.sum()); out["n_families"] = int(units.family[s].nunique())
    elif panel == "hcc_mean":
        for k in ["rmse", "pearson", "centred_pearson", "mrr", "top20_direction"]:
            out[k] = float(np.mean([metrics[k][units.cell.to_numpy() == h].mean() for h in HCC]))
    else:
        s = units.cell.to_numpy() == panel
        for k in ["rmse", "pearson", "centred_pearson", "mrr", "top20_direction", "rms_pred", "rms_true"]:
            out[k] = float(metrics[k].to_numpy()[s].mean())
        out["n_units"] = int(s.sum()); out["n_candidates"] = int(s.sum())
    return out


def primary(pred, true, units):
    m = unit_metrics(pred, true, units.cell.to_numpy())
    return summarise(m, units, "source")["rmse"], m


def paired_bootstrap(rmse_a, rmse_b, units, panel="source", n=2000, seed=20261117):
    """Difference a - b of the panel statistic; drug groups resampled (all cells move together)."""
    groups = units.drug_group.to_numpy()
    ug, gi = np.unique(groups, return_inverse=True)
    rng = np.random.RandomState(seed)
    d = np.asarray(rmse_a) - np.asarray(rmse_b)
    fam = units.family.to_numpy(); cell = units.cell.to_numpy(); hcc = units.is_hcc.to_numpy() == 1

    def stat(w):
        if panel == "source":
            return family_mean(d[~hcc], fam[~hcc], w[~hcc])
        if panel == "hcc_mean":
            vals = []
            for h in HCC:
                s = (cell == h) & (w > 0)
                vals.append((d[s] * w[s]).sum() / w[s].sum() if s.any() else np.nan)
            return float(np.nanmean(vals))
        s = (cell == panel) & (w > 0)
        return float((d[s] * w[s]).sum() / w[s].sum())

    point = stat(np.ones(len(d)))
    bs = np.empty(n)
    for b in range(n):
        cnt = np.bincount(rng.randint(0, len(ug), len(ug)), minlength=len(ug))
        bs[b] = stat(cnt[gi].astype(float))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(diff=point, lo=float(lo), hi=float(hi))
