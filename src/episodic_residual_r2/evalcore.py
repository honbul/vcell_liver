"""Scoring of episodes and the family-equal aggregation used for every source-side table."""
import numpy as np
import pandas as pd

from .metrics import per_unit, summarise


def shared_plate_mask(ep, K):
    """True for query units that share at least one detection plate with a support unit (at support size K)."""
    sp = set()
    for p in ep.plates_s[:K]:
        sp |= set(p.split("|"))
    return np.array([bool(set(p.split("|")) & sp) for p in ep.plates_q])


def score(ep, K, pred, extra=None):
    """One result row for (episode, K, method). `ep.y_q` is touched here and only here."""
    pu = per_unit(pred.detach().cpu().numpy(), ep.y_q.cpu().numpy())
    s = summarise(pu)
    m = shared_plate_mask(ep, K) if ep.plates_q else np.zeros(len(pu), bool)
    s.update(n_query_shared_plate=int(m.sum()), n_query_disjoint_plate=int((~m).sum()),
             delta_rmse_shared_plate=float(pu.delta_rmse[m].mean()) if m.any() else np.nan,
             delta_rmse_disjoint_plate=float(pu.delta_rmse[~m].mean()) if (~m).any() else np.nan)
    row = dict(cell=ep.name, family=ep.family, support_seed=ep.seed, K=K, **s)
    if extra:
        row.update(extra)
    return row, pu


def family_mean(df, value="delta_rmse", by=("K", "method")):
    """episodes -> mean over support seeds per cell -> equal mean over cells of a family -> equal mean over families."""
    by = list(by)
    c = df.groupby(by + ["family", "cell"], dropna=False)[value].mean().reset_index()
    f = c.groupby(by + ["family"], dropna=False)[value].mean().reset_index()
    return f.groupby(by, dropna=False)[value].agg(["mean", "count"]).reset_index().rename(columns={"mean": value, "count": "n_families"}), f


def paired_family(fam_tab, a, b, value="delta_rmse", n_boot=2000, seed=20261117):
    """Family-level paired difference a - b with a family bootstrap interval. fam_tab: rows (K, method, family, value)."""
    out = []
    for K, g in fam_tab.groupby("K"):
        p = g.pivot(index="family", columns="method", values=value)
        if a not in p or b not in p:
            continue
        d = (p[a] - p[b]).dropna().to_numpy()
        rs = np.random.RandomState(seed)
        bs = np.array([d[rs.randint(len(d), size=len(d))].mean() for _ in range(n_boot)])
        out.append(dict(K=K, a=a, b=b, diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                        n_families=len(d), n_families_a_better=int((d < 0).sum())))
    return pd.DataFrame(out)
