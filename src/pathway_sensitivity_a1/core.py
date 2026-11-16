"""Shared helpers for Track A: config, hashing, cached data access, splits, features.

All tables are joined through explicit ID dictionaries; positional alignment is asserted, never assumed.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CONFIG = os.path.join(REPO, "configs", "pathway_sensitivity_a1", "analysis.yaml")
MANIFESTS = os.path.join(REPO, "manifests", "pathway_sensitivity_a1")
REPORTS = os.path.join(REPO, "reports", "pathway_sensitivity_a1")


def load_config(path: str = CONFIG) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def rpath(p: str) -> str:
    """Resolve a config path (absolute, or relative to the repository root)."""
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_cr_table(path: str) -> pd.DataFrame:
    """LCCL downloads use bare CR line endings."""
    with open(path, newline="") as fh:
        txt = fh.read().replace("\r\n", "\n").replace("\r", "\n")
    return pd.read_csv(io.StringIO(txt), sep="\t")


def name_key(name: str) -> str:
    """Punctuation-free key used only to line up spellings of the same source name
    (e.g. 'PLC/PRF5' in the response table and 'PLC.PRF5' in the RNA header of the same study)."""
    return re.sub(r"[^A-Za-z0-9]", "", str(name))


def model_id(study: str, source_name: str) -> str:
    return f"{study}__{name_key(source_name)}"


def write_new(path: str, writer) -> None:
    """Write through a temp file; refuse to overwrite an existing output."""
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    writer(tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------
# data cache
# ----------------------------------------------------------------------------------------------

def cache_key(cfg: dict) -> str:
    """Key over input file hashes, gene-axis rule and Hallmark version."""
    parts = {
        "limore_rna": sha256_file(rpath(cfg["paths"]["limore_rna"])),
        "limore_response": sha256_file(rpath(cfg["paths"]["limore_response"])),
        "lccl_rna": sha256_file(rpath(cfg["paths"]["lccl_rna"])),
        "lccl_auc": sha256_file(rpath(cfg["paths"]["lccl_auc"])),
        "lccl_gi50": sha256_file(rpath(cfg["paths"]["lccl_gi50"])),
        "hallmark_gmt": sha256_file(rpath(cfg["paths"]["hallmark_gmt"])),
        "aliases": sha256_file(rpath(cfg["paths"]["model_aliases"])),
        "cellosaurus": sha256_file(rpath(cfg["paths"]["cellosaurus"])),
        "gene_axis_rule": cfg["rna"]["gene_axis"],
        "rank_rule": cfg["rna"]["rank"],
        "hallmark": [cfg["hallmark"]["release"], cfg["hallmark"]["min_genes"], cfg["hallmark"]["min_fraction"]],
        "a4_seeds": cfg["a4"]["seeds"],
    }
    return sha256_text(json.dumps(parts, sort_keys=True))[:12]


def cache_dir(cfg: dict) -> str:
    return os.path.join(rpath(cfg["paths"]["assets"]), "cache", f"data_{cache_key(cfg)}")


@dataclass
class Study:
    name: str
    models: list            # model_id, row order of every matrix below
    expr: np.ndarray        # models x genes, log2 scale
    rank: np.ndarray        # models x genes, relative within-sample rank
    hallmark: np.ndarray    # models x K
    sets: list              # K set names


def _read_ids(path: str) -> list:
    with open(path) as fh:
        return [l.rstrip("\n") for l in fh if l.strip()]


def _score_table(path: str, models: list, sets: list | None = None) -> tuple:
    df = pd.read_csv(path, sep="\t", index_col=0)
    assert set(df.index) == set(models) and len(df.index) == len(models), f"model mismatch in {path}"
    df = df.loc[models]
    if sets is not None:
        assert list(df.columns) == list(sets), f"set order mismatch in {path}"
    return df.values.astype(float), list(df.columns)


def load_study(cfg: dict, study: str) -> Study:
    d = cache_dir(cfg)
    s = study.lower()
    models = _read_ids(os.path.join(d, f"{s}__models.txt"))
    expr = np.load(os.path.join(d, f"{s}__expr_log2.npy"))
    rank = np.load(os.path.join(d, f"{s}__rank.npy"))
    hm, sets = _score_table(os.path.join(d, f"{s}__hallmark_real.tsv"), models)
    assert expr.shape[0] == len(models) == rank.shape[0]
    return Study(study, models, expr, rank, hm, sets)


def load_perm_scores(cfg: dict, study: str, seed: int, models: list, sets: list) -> np.ndarray:
    path = os.path.join(cache_dir(cfg), f"{study.lower()}__hallmark_perm{seed:03d}.tsv")
    return _score_table(path, models, sets)[0]


def load_registry() -> pd.DataFrame:
    reg = pd.read_csv(os.path.join(MANIFESTS, "model_registry.tsv"), sep="\t")
    assert reg.model_id.is_unique
    return reg


def load_limore_response(cfg: dict) -> pd.DataFrame:
    """Long table model_id, drug_id, y (primary endpoint) for all LIMORE models, joined by ID."""
    df = pd.read_csv(os.path.join(cache_dir(cfg), "limore__response_long.tsv"), sep="\t")
    assert not df.duplicated(["model_id", "drug_id"]).any()
    return df


def response_matrix(long: pd.DataFrame, models: list, drugs: list, col: str) -> np.ndarray:
    """models x drugs matrix through explicit ID lookup; missing pairs stay NaN."""
    mi = {m: i for i, m in enumerate(models)}
    di = {d: j for j, d in enumerate(drugs)}
    Y = np.full((len(models), len(drugs)), np.nan)
    for m, d, v in zip(long["model_id"], long["drug_id"], long[col]):
        if m in mi and d in di:
            Y[mi[m], di[d]] = v
    # round trip
    back = {(models[i], drugs[j]): Y[i, j] for i, j in zip(*np.where(~np.isnan(Y)))}
    src = {(m, d): v for m, d, v in zip(long["model_id"], long["drug_id"], long[col])
           if m in mi and d in di and not pd.isna(v)}
    assert back == src, "response matrix round trip failed"
    return Y


# ----------------------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------------------

def min_rank(x: np.ndarray) -> np.ndarray:
    """Within-row rank with ties='min' (1 = lowest), as singscore::rankGenes."""
    from scipy.stats import rankdata
    return rankdata(x, method="min", axis=1).astype(float)


def relative_rank(x: np.ndarray) -> np.ndarray:
    return min_rank(x) / x.shape[1]


def singscore_up(ranks: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Numpy transcription of singscore's centred single-direction score.
    ranks: samples x genes (ties='min'); idx: column indices of the gene set."""
    n_genes = ranks.shape[1]
    n = len(idx)
    score = ranks[:, idx].mean(axis=1) / n_genes
    low = (n + 1) / (2 * n_genes)
    up = (2 * n_genes - n + 1) / (2 * n_genes)
    return (score - low) / (up - low) - 0.5


def read_gmt(path: str) -> dict:
    sets = {}
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) > 2:
                sets[p[0]] = [g for g in p[2:] if g]
    return sets


# ----------------------------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------------------------

def family_weights(families: np.ndarray) -> np.ndarray:
    """1 / number of rows of the same family in this set."""
    fam, inv, cnt = np.unique(families, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def group_folds(groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Fold index per row; all rows of a family share a fold."""
    from sklearn.model_selection import GroupKFold
    groups = np.asarray(groups)
    fold = np.full(len(groups), -1)
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(gkf.split(np.zeros(len(groups)), groups=groups)):
        fold[te] = k
    assert (fold >= 0).all()
    for g in np.unique(groups):
        assert len(set(fold[groups == g])) == 1, "family split across folds"
    return fold


def load_outer_splits() -> pd.DataFrame:
    return pd.read_csv(os.path.join(MANIFESTS, "splits_outer.tsv"), sep="\t")
