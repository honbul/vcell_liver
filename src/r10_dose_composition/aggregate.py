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
# v2 (R10-E1): the training contract and the atlas / assembly versions are part of every evaluation,
# aggregation and pairing key, so heads trained under different contracts are never averaged together.
VERSION_KEYS = ["train_contract", "assembly_version", "atlas_version"]
KEYS = ["role", "line", "drug", "episode", "direction", "method"] + VERSION_KEYS
LEGACY = "v1_unlabelled"


def ensure_keys(df):
    """v1 run files have no contract/version columns: label them explicitly instead of guessing."""
    df = df.copy()
    for k in VERSION_KEYS:
        if k not in df:
            df[k] = LEGACY
    return df


def alias_d00(df):
    """The 2x2 D00 arm is the empirical mixture (identical clouds); add it under its arm name."""
    df = ensure_keys(df)
    m = df[df.method == "mixture"].copy()
    m["method"] = "D00"
    return pd.concat([df, m], ignore_index=True)


def per_condition(df, metrics=METRICS):
    df = ensure_keys(df)
    df = df[df.is_oracle == 0] if "is_oracle" in df else df
    a = df.groupby(KEYS + ["rep_endpoint", "rep_query"], as_index=False)[metrics].mean()   # seeds
    return a.groupby(KEYS, as_index=False)[metrics].mean()                                 # directions


def per_drug(cond, metrics=METRICS):
    return cond.groupby(["role", "drug", "episode", "direction", "method"] + VERSION_KEYS,
                        as_index=False)[metrics].mean()


def arm_label(df):
    """v2 arm id: method plus training contract (D00 and other non-learned arms keep their name)."""
    df = ensure_keys(df)
    tc = df.train_contract.astype(str)
    lab = df.method.where(tc.isin(["none", LEGACY]), df.method + "[" + tc + "]")
    return df.assign(arm=lab)


def _unique_methods(sub):
    """Guard (v2): one row per drug and method, i.e. no silent averaging across training contracts."""
    dup = sub.duplicated(["drug", "method"]).sum()
    assert dup == 0, f"{dup} drug x method rows differ only in train_contract/version keys; use arm_label()"


def summary(drug, metric, methods, B=2000, seed=20261117):
    rows = []
    for (role, ep, di), sub in drug.groupby(["role", "episode", "direction"]):
        _unique_methods(sub)
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
        _unique_methods(sub)
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
    return c.groupby(["role", "episode", "direction", "line", "method"] + VERSION_KEYS, as_index=False)[metrics].mean()
