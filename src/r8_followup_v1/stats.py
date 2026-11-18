"""Uncertainty and decision helpers (plan sections 11.3-11.5).

source / matched plate : family cluster bootstrap (a family carries all its cells, seeds, pairs and A<->B orders)
HCC                    : synchronised drug-ID bootstrap: ONE multinomial weight per drug ID over the union of query drug groups of
                         all targets and splits, so a drug shared by several targets is drawn together; the equal mean of the
                         three targets is computed inside every draw. A per-target conditional bootstrap uses independent
                         SeedSequence streams and is labelled as such.
Decision rule          : 0.5% relative delta-RMSE operating criterion, direction in >= 2/3 outer folds, 95% CI; equivalence
                         only if the 90% CI lies inside +-0.5% of the reference mean.
"""
import numpy as np
import pandas as pd

from .common import BOOT_SEED, N_BOOT

REL = 0.005


def boot_mean(d, n_boot=N_BOOT, seed=BOOT_SEED):
    d = np.asarray(d, float)
    rs = np.random.RandomState(seed)
    return d[rs.randint(len(d), size=(n_boot, len(d)))].mean(1)


def summarise_diff(d, ref_mean, draws=None):
    d = np.asarray(d, float)
    bs = boot_mean(d) if draws is None else draws
    m = float(d.mean()) if draws is None else float(np.mean(d))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    lo9, hi9 = np.percentile(bs, [5, 95])
    tol = REL * ref_mean
    return dict(diff=m, lo=float(lo), hi=float(hi), lo90=float(lo9), hi90=float(hi9), rel_diff=m / ref_mean, ref_mean=float(ref_mean),
                ci_below_0=bool(hi < 0), ci_above_0=bool(lo > 0), meets_0p5pct=bool(m <= -tol),
                equivalent_0p5pct=bool(lo9 > -tol and hi9 < tol))


def family_table(df, value="delta_rmse", by=("K", "method")):
    """episodes -> support seeds -> cells of a family -> family (the original aggregation order)."""
    by = list(by)
    c = df.groupby(by + ["family", "cell"], dropna=False)[value].mean().reset_index()
    return c.groupby(by + ["family"], dropna=False)[value].mean().reset_index()


def paired(fam, a, b, value="delta_rmse", fold_of=None, extra_by=()):
    """Paired family contrast a - b (rows of fam: extra_by..., method, family, value)."""
    out = []
    keys = list(extra_by)
    groups = fam.groupby(keys) if keys else [((), fam)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        p = g.pivot(index="family", columns="method", values=value)
        if a not in p or b not in p:
            continue
        p = p[[a, b]].dropna()
        d = (p[a] - p[b]).to_numpy()
        if len(d) < 2:
            continue
        r = dict(zip(keys, key), a=a, b=b, metric=value, n_families=len(d), n_families_a_lower=int((d < 0).sum()), mean_a=float(p[a].mean()), mean_b=float(p[b].mean()))
        r.update(summarise_diff(d, float(p[b].mean())))
        if fold_of is not None:
            f = pd.Series(d, index=p.index).groupby(p.index.map(fold_of)).mean()
            r.update(fold_diffs=";".join(f"{k}:{v:+.5f}" for k, v in f.items()), n_folds_a_lower=int((f < 0).sum()), n_folds=int(len(f)))
            r["adoption_rule_met"] = bool(r["ci_below_0"] and r["n_folds_a_lower"] >= 2)
        out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- HCC
def sync_drug_weights(drugs_by_unit, n_boot=N_BOOT, seed=BOOT_SEED):
    """drugs_by_unit: dict key -> list of drug IDs. One multinomial draw over the union of drug IDs; each key receives the
    columns of its own drugs (a drug shared by several keys gets the SAME weight in every key within a draw)."""
    union = sorted(set().union(*[set(v) for v in drugs_by_unit.values()]))
    pos = {g: i for i, g in enumerate(union)}
    rs = np.random.RandomState(seed)
    W = rs.multinomial(len(union), np.ones(len(union)) / len(union), size=n_boot).astype(np.float64)
    return {k: W[:, [pos[g] for g in v]] for k, v in drugs_by_unit.items()}, len(union)


def independent_streams(keys, seed=BOOT_SEED):
    """Per-key conditional bootstrap generators from independent SeedSequence children (never a re-started common seed)."""
    ch = np.random.SeedSequence(seed).spawn(len(keys))
    return {k: np.random.default_rng(c) for k, c in zip(keys, ch)}


def weighted_means(x, W):
    ok = np.isfinite(x)
    w = W[:, ok]
    den = w.sum(1)
    return np.where(den > 0, (w * x[ok]).sum(1) / np.maximum(den, 1e-12), np.nan)
