"""Scoring for Track A. Observed responses are joined to prediction tables by (model_id, drug_id)."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_oof(run_dir: str, name: str, n_repeats: int) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(run_dir, "oof", f"{name}__rep*.tsv.gz")))
    if len(files) != n_repeats:
        raise FileNotFoundError(f"{name}: expected {n_repeats} repeat tables in {run_dir}, found {len(files)}")
    return pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)


def attach_truth(pred: pd.DataFrame, long: pd.DataFrame, family_of: dict, s_ref: pd.DataFrame | None = None):
    """Join observed y and family by explicit keys. `s_ref` (repeat, outer_fold, drug_id, s_train)
    replaces the table's own s_train so that every run is normalised by the same full-training SD."""
    n = len(pred)
    df = pred.merge(long[["model_id", "drug_id", "y"]], on=["model_id", "drug_id"], how="left", validate="many_to_one")
    assert len(df) == n
    df["family_id"] = df.model_id.map(family_of)
    assert df.family_id.notna().all()
    if s_ref is not None:
        df = df.drop(columns="s_train").merge(s_ref, on=["repeat", "outer_fold", "drug_id"], how="left",
                                              validate="many_to_one")
        assert df.s_train.notna().all() and len(df) == n
    df = df[df.y.notna() & df.pred.notna()].copy()       # unlabeled pairs are not scored
    df["z_err2"] = ((df.y - df.pred) / df.s_train) ** 2
    return df


def error_cube(df: pd.DataFrame, repeats: list, drugs: list, families: list) -> np.ndarray:
    """F[repeat, drug, family] = within-family mean of squared normalised error (NaN if absent)."""
    g = df.groupby(["repeat", "drug_id", "family_id"]).z_err2.mean()
    ri = {r: i for i, r in enumerate(repeats)}
    di = {d: i for i, d in enumerate(drugs)}
    fi = {f: i for i, f in enumerate(families)}
    F = np.full((len(repeats), len(drugs), len(families)), np.nan)
    for (r, d, f), v in g.items():
        F[ri[r], di[d], fi[f]] = v
    return F


def drug_nrmse(F: np.ndarray) -> np.ndarray:
    """[repeat, drug] normalised RMSE: sqrt of the mean over families."""
    return np.sqrt(np.nanmean(F, axis=2))


def macro_nrmse(F: np.ndarray) -> float:
    """Primary metric: per-repeat macro average over drugs, then mean of repeats."""
    return float(np.nanmean(drug_nrmse(F), axis=1).mean())


def bootstrap_weights(n_families: int, n_boot: int, seed: int) -> np.ndarray:
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, n_families, size=(n_boot, n_families))
    W = np.zeros((n_boot, n_families))
    for b in range(n_boot):
        np.add.at(W[b], idx[b], 1.0)
    return W


def boot_drug_nrmse(F: np.ndarray, W: np.ndarray) -> np.ndarray:
    """[boot, repeat, drug] normalised RMSE under family resampling (whole families move together)."""
    valid = ~np.isnan(F)
    num = np.einsum("bf,rdf->brd", W, np.where(valid, F, 0.0))
    den = np.einsum("bf,rdf->brd", W, valid.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(num / den)


def boot_macro(F: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.nanmean(boot_drug_nrmse(F, W), axis=2).mean(axis=1)


def ci(x: np.ndarray, lo: float = 0.025, hi: float = 0.975) -> tuple:
    return float(np.nanquantile(x, lo)), float(np.nanquantile(x, hi))


def family_level(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df.groupby(["repeat", "drug_id", "family_id"])[cols].mean().reset_index()


def safe_spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def per_drug_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per repeat and drug: normalised RMSE, raw-unit RMSE and MAE (family-first) and Spearman."""
    df = df.assign(err2=(df.y - df.pred) ** 2, abserr=(df.y - df.pred).abs())
    fam = family_level(df, ["y", "pred", "z_err2", "err2", "abserr"])
    rows = []
    for (r, d), t in fam.groupby(["repeat", "drug_id"]):
        rows.append(dict(repeat=r, drug_id=d, n_families=len(t), nrmse=float(np.sqrt(t.z_err2.mean())),
                         rmse_raw=float(np.sqrt(t.err2.mean())), mae_raw=float(t.abserr.mean()),
                         spearman=safe_spearman(t.y, t.pred)))
    return pd.DataFrame(rows)
