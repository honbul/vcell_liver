"""Aggregation: triplet rows -> (line, drug, episode) -> drug (lines weighted equally) -> drug-level
paired bootstrap. Learned arms are first averaged over seeds inside a triplet (seed spread is
reported separately). Cross-replicate directions are the primary condition; the two cross
directions of one drug are averaged, never counted as two drugs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import drug_block_bootstrap

METRICS = ["comp_tv", "comp_js", "comp_brier", "within_mean_rmse", "within_ed", "ed", "mmd", "dmean_rmse_gene",
           "mean_rmse_latent"]
KEYS = ["role", "line", "drug", "episode", "direction", "method"]


def alias_d00(df):
    """The 2x2 D00 arm is the empirical mixture (identical clouds); add it under its arm name."""
    m = df[df.method == "mixture"].copy()
    m["method"] = "D00"
    return pd.concat([df, m], ignore_index=True)


def per_condition(df, metrics=METRICS):
    df = df[df.is_oracle == 0] if "is_oracle" in df else df
    a = df.groupby(KEYS + ["rep_endpoint", "rep_query"], as_index=False)[metrics].mean()   # seeds
    return a.groupby(KEYS, as_index=False)[metrics].mean()                                 # directions


def per_drug(cond, metrics=METRICS):
    return cond.groupby(["role", "drug", "episode", "direction", "method"], as_index=False)[metrics].mean()


def summary(drug, metric, methods, B=2000, seed=20261117):
    rows = []
    for (role, ep, di), sub in drug.groupby(["role", "episode", "direction"]):
        for m in methods:
            s = sub[sub.method == m].set_index("drug")[[metric]].dropna()
            if len(s) == 0:
                continue
            r = drug_block_bootstrap(s, B=B, seed=seed)
            rows.append(dict(role=role, episode=ep, direction=di, method=m, metric=metric, mean=r["est"],
                             lo=r["lo"], hi=r["hi"], n_drugs=r["n_drugs"]))
    return pd.DataFrame(rows)


def paired(drug, metric, pairs, B=2000, seed=20261117):
    """pairs: list of (method, reference). Difference method - reference per drug (negative = better)."""
    rows = []
    for (role, ep, di), sub in drug.groupby(["role", "episode", "direction"]):
        w = sub.pivot_table(index="drug", columns="method", values=metric)
        for m, ref in pairs:
            if m not in w or ref not in w:
                continue
            dif = (w[m] - w[ref]).dropna().to_frame("d")
            if len(dif) == 0:
                continue
            r = drug_block_bootstrap(dif, B=B, seed=seed)
            rows.append(dict(role=role, episode=ep, direction=di, method=m, reference=ref, metric=metric,
                             diff=r["est"], lo=r["lo"], hi=r["hi"], n_drugs=r["n_drugs"],
                             frac_drugs_better=float((dif.d < 0).mean())))
        if all(x in w for x in ("D00", "D10", "D01", "D11")):
            dif = (w["D11"] - w["D10"] - w["D01"] + w["D00"]).dropna().to_frame("d")
            r = drug_block_bootstrap(dif, B=B, seed=seed)
            rows.append(dict(role=role, episode=ep, direction=di, method="interaction", reference="D11-D10-D01+D00",
                             metric=metric, diff=r["est"], lo=r["lo"], hi=r["hi"], n_drugs=r["n_drugs"],
                             frac_drugs_better=np.nan))
    return pd.DataFrame(rows)


def per_line_summary(cond, metrics, methods):
    c = cond[cond.method.isin(methods)]
    return c.groupby(["role", "episode", "direction", "line", "method"], as_index=False)[metrics].mean()
