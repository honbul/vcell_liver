"""Study R1 data access: config, fixed-ID gene axis cache, Hallmark membership, splits, labels.

Track A modules (src/pathway_sensitivity_a1) are imported read-only for the pieces that must stay
identical (family weights, group folds, rank transform, singscore transcription, metrics).
Nothing here writes outside this worktree.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TRACK = "softprior_multitask_r1"
CONFIG = os.path.join(REPO, "configs", TRACK, "analysis.yaml")
MANIFESTS = os.path.join(REPO, "manifests", TRACK)
REPORTS = os.path.join(REPO, "reports", TRACK)
PREDICTIONS = os.path.join(REPO, "predictions", TRACK)
A1_SRC = os.path.join(REPO, "src", "pathway_sensitivity_a1")
A1_MANIFESTS = os.path.join(REPO, "manifests", "pathway_sensitivity_a1")

if A1_SRC not in sys.path:
    sys.path.insert(0, A1_SRC)
import core as a1core  # noqa: E402  (Track A helpers, unchanged)


def load_config(path: str = CONFIG) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def rpath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def sha256_file(path: str) -> str:
    return a1core.sha256_file(path)


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str(a.shape).encode() + str(a.dtype).encode()).hexdigest()


def cache_key(cfg: dict) -> str:
    parts = dict(limore_rna=cfg["expected_sha256"]["limore_rna"], lccl_rna=cfg["expected_sha256"]["lccl_rna"],
                 hallmark=cfg["expected_sha256"]["hallmark_gmt"], gene_axis=cfg["gene_axis"],
                 hallmark_rule=[cfg["hallmark"]["release"], cfg["hallmark"]["min_genes"],
                                cfg["hallmark"]["min_fraction"]],
                 perm=cfg["hallmark"]["permutation_seeds"])
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def cache_dir(cfg: dict) -> str:
    return os.path.join(rpath(cfg["paths"]["assets"]), "cache", f"data_{cache_key(cfg)}")


def runs_dir(cfg: dict) -> str:
    return os.path.join(rpath(cfg["paths"]["assets"]), "runs")


def _read_ids(path: str) -> list:
    with open(path) as fh:
        return [l.rstrip("\n") for l in fh if l.strip()]


def permuted_sets(sets: list, n_genes: int, seed: int) -> list:
    """Track A permutation scheme on index lists: gene i takes the label of gene perm[i]."""
    perm = np.random.RandomState(seed).permutation(n_genes)
    return [np.sort(perm[idx]) for idx in sets]


@dataclass
class Data:
    cfg: dict
    genes: pd.DataFrame            # gene axis (ensembl_id, symbol)
    set_names: list                # K Hallmark names
    sets: list                     # K index arrays into the gene axis
    models: list                   # included LIMORE models, sorted (row order of all LIMORE matrices below)
    fam: np.ndarray                # family_id per row
    rank: np.ndarray               # models x genes relative rank
    expr: np.ndarray               # models x genes log2 FPKM
    hallmark: np.ndarray           # models x K singscore
    drugs: list                    # 90 drug ids, sorted
    Y: np.ndarray                  # models x drugs Activity Area
    splits: pd.DataFrame           # Track A outer split table
    extra: dict = field(default_factory=dict)

    def outer_fold(self, rep: int) -> np.ndarray:
        sp = self.splits[self.splits.repeat == rep].set_index("model_id")
        fold = sp.loc[self.models, "outer_fold"].values
        assert (sp.loc[self.models, "family_id"].values == self.fam).all()
        return fold

    def prior_sets(self, prior: str) -> list:
        """'real' or 'permNNN'."""
        if prior == "real":
            return self.sets
        assert prior.startswith("perm")
        seed = int(prior[4:])
        assert seed in self.cfg["hallmark"]["permutation_seeds"]
        return permuted_sets(self.sets, len(self.genes), seed)


def load_registry() -> pd.DataFrame:
    reg = pd.read_csv(os.path.join(A1_MANIFESTS, "model_registry.tsv"), sep="\t")
    assert reg.model_id.is_unique
    return reg


def load_sets(d: str, genes: pd.DataFrame) -> tuple:
    sym_idx = {s: i for i, s in enumerate(genes.symbol)}
    gmt = a1core.read_gmt(os.path.join(d, "hallmark_on_axis.gmt"))
    names = list(gmt)
    sets = [np.array(sorted(sym_idx[g] for g in gmt[k])) for k in names]
    return names, sets


def load_data(cfg: dict | None = None) -> Data:
    cfg = cfg or load_config()
    d = cache_dir(cfg)
    genes = pd.read_csv(os.path.join(d, "gene_axis.tsv"), sep="\t")
    names, sets = load_sets(d, genes)
    reg = load_registry()
    elig = reg[(reg.study == "LIMORE") & reg.included].sort_values("model_id")
    models = list(elig.model_id)
    all_models = _read_ids(os.path.join(d, "limore__models.txt"))
    row = {m: i for i, m in enumerate(all_models)}
    idx = np.array([row[m] for m in models])
    rank = np.load(os.path.join(d, "limore__rank.npy"))[idx]
    expr = np.load(os.path.join(d, "limore__expr_log2.npy"))[idx]
    hm = pd.read_csv(os.path.join(d, "limore__hallmark_real.tsv"), sep="\t", index_col=0)
    assert list(hm.columns) == names
    hallmark = hm.loc[models].values.astype(float)
    long = pd.read_csv(os.path.join(cfg["paths"]["a1_cache"], "limore__response_long.tsv"), sep="\t")
    assert not long.duplicated(["model_id", "drug_id"]).any()
    drugs = sorted(long.drug_id.unique())
    Y = a1core.response_matrix(long, models, drugs, "y")
    assert not np.isnan(Y).any(), "R1 code assumes the complete LIMORE label matrix recorded by Track A"
    splits = pd.read_csv(os.path.join(A1_MANIFESTS, "splits_outer.tsv"), sep="\t")
    for rep, t in splits.groupby("repeat"):
        assert sorted(t.model_id) == models
    return Data(cfg, genes, names, sets, models, elig.family_id.values, rank, expr, hallmark, drugs, Y, splits)


@dataclass
class External:
    models: list                   # included LCCL models, sorted
    fam: np.ndarray
    eval_type: np.ndarray          # cross_study_leave_family_out | new_lineage
    rank: np.ndarray
    expr: np.ndarray
    hallmark: np.ndarray
    drug_match: pd.DataFrame       # matched rows: limore_name, lccl_name
    obs: pd.DataFrame              # LCCL long responses (model_id, lccl_name, AUC, GI50)


def load_external(cfg: dict, data: Data) -> External:
    d = cache_dir(cfg)
    reg = load_registry()
    tgt = reg[(reg.study == "LCCL") & reg.included].sort_values("model_id")
    models = list(tgt.model_id)
    all_models = _read_ids(os.path.join(d, "lccl__models.txt"))
    row = {m: i for i, m in enumerate(all_models)}
    idx = np.array([row[m] for m in models])
    hm = pd.read_csv(os.path.join(d, "lccl__hallmark_real.tsv"), sep="\t", index_col=0)
    assert list(hm.columns) == data.set_names
    dm = pd.read_csv(os.path.join(A1_MANIFESTS, "drug_match_lccl.tsv"), sep="\t").fillna("")
    dm = dm[dm.match_status == "matched"].reset_index(drop=True)
    obs = pd.read_csv(os.path.join(cfg["paths"]["a1_cache"], "lccl__response_long.tsv"), sep="\t")
    et = np.where(tgt.family_id.isin(set(data.fam)), "cross_study_leave_family_out", "new_lineage")
    return External(models, tgt.family_id.values, et, np.load(os.path.join(d, "lccl__rank.npy"))[idx],
                    np.load(os.path.join(d, "lccl__expr_log2.npy"))[idx], hm.loc[models].values.astype(float),
                    dm, obs)


def inner_seed(cfg: dict, rep: int, fold: int) -> int:
    return cfg["split"]["inner_seed_base"] + 100 * int(rep) + int(fold)
