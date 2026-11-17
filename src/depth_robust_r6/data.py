"""The fixed data contract of R6 and its loader.

Units
    pool        all DMSO cells of one (plate, cell line): the control population of every condition
                whose primary plate is that plate
    condition   one (cell line, drug structure group); the scored row
    drug group  the cluster unit (one split role across all cell lines)

Roles of control cells inside a pool (cells ordered by blake2b(barcode), so the roles depend on
nothing but the barcode; barcode-disjoint by construction):
    eval_input   ranks [0, 64)            the N=64 cells fed to every model at evaluation
    fit_input    ranks [64, 64 + n_fit)   cells that training draws its N=64 control sets from
    reference    all remaining ranks      defines the control mean the truth subtracts

Truth (never depends on p, never thinned):
    Y_r6[line, drug] = mean over wells of ( treated full-well mean - reference-control mean )
                     = Y_v2[line, drug] + sum_plate w[line, drug, plate] * (pool_all_mean - reference_mean)
where Y_v2 used the whole pool (input cells included). Only the control term moves; the treated
term is the earlier study's full-well mean, untouched.

Settings
    dev     fit: TRAIN drugs in dev folds {0, 1};  evaluate: TRAIN drugs in dev fold 2
    final   fit: all TRAIN drugs;                  evaluate: the 70 historical TEST drugs
The 70 historical VALIDATION drugs are not used by any step of this study.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_N = 64
FIT_MAX = 256
REF_MIN = 128
N_FOLDS = 3
DEV_EVAL_FOLD = 2
PS = (1.0, 0.5, 0.25)
THIN_SEEDS = (0, 1, 2)


def bhash(s: str) -> float:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") / 2.0 ** 64


def sha256_file(path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def pool_roles(n: int):
    """(n_eval, n_fit, n_ref) for a pool of n cells, or None if the pool cannot serve all three roles."""
    if n < EVAL_N + EVAL_N + REF_MIN:
        return None
    n_fit = min(FIT_MAX, n - EVAL_N - REF_MIN)
    return EVAL_N, n_fit, n - EVAL_N - n_fit


@dataclass
class R6Data:
    cnt: np.ndarray            # [n_in, G] int16 panel counts of the input cells (eval then fit, per pool)
    crest: np.ndarray          # [n_in] int32 counts outside the panel
    pools: pd.DataFrame        # pool_id, plate_index, line, eval_start/end, fit_start/end, n_ref, n_total
    pool_id: np.ndarray        # [P, L] -> pool id or -1
    Y: np.ndarray              # [L, D, G] float32 R6 truth
    valid: np.ndarray          # [L, D] bool
    plate: np.ndarray          # [L, D] primary plate
    fp: np.ndarray             # [D, 2048]
    splits: pd.DataFrame
    fold: np.ndarray           # [D] dev fold of TRAIN drugs, -1 otherwise
    lines: pd.DataFrame
    x_mean: np.ndarray         # [G] mean log-normalised expression of fit_input cells (controls only)
    x_sd: np.ndarray
    tiebreak: np.ndarray       # [G] in [0, 0.5): fixed token order among equal counts
    eval_views: dict           # (p, seed) -> (cnt [n_eval_cells, G] int16, crest [n_eval_cells] int32)
    root: Path

    def conditions(self, drugs) -> tuple[np.ndarray, np.ndarray]:
        L, D = np.nonzero(self.valid)
        keep = np.isin(D, np.asarray(list(drugs)))
        return L[keep], D[keep]

    def split_spec(self, setting: str):
        tr = [int(g) for g in self.splits.group_index[self.splits.split == "train"]]
        if setting == "dev":
            return [g for g in tr if self.fold[g] != DEV_EVAL_FOLD], [g for g in tr if self.fold[g] == DEV_EVAL_FOLD]
        if setting == "final":
            return tr, [int(g) for g in self.splits.group_index[self.splits.split == "test"]]
        raise ValueError(setting)


def load(root: str | None = None, v2data: str | None = None, manifest_dir: str = "manifests/chemcpa_moe_v2",
         eval_views: bool = True) -> R6Data:
    root = Path(root or Path(os.environ["R6_ASSETS"]) / "cache" / "contract_v1")
    v2 = Path(v2data or os.environ["R6_V2DATA"])
    z = np.load(root / "input_counts.npz")
    pools = pd.read_csv(root / "pools.tsv", sep="\t")
    splits = pd.read_csv(Path(manifest_dir) / "splits.tsv", sep="\t").sort_values("group_index").reset_index(drop=True)
    lines = pd.read_csv(Path(manifest_dir) / "lines.tsv", sep="\t").sort_values("line_index").reset_index(drop=True)
    Y = np.load(root / "Y_r6.npy")
    valid = np.load(root / "valid_r6.npy")
    plate = np.load(v2 / "plate.npy")
    fp = np.load(v2 / "fp.npy").astype(np.float32)
    P = int(pools.plate_index.max()) + 1
    pid = np.full((P, Y.shape[0]), -1, int)
    pid[pools.plate_index, pools.line] = pools.pool_id
    fold = np.full(len(splits), -1, int)
    for r in splits.itertuples():
        if r.split == "train":
            fold[r.group_index] = int(bhash("r3-devfold|" + str(r.group_smiles)) * N_FOLDS)   # same folds as R3
    st = np.load(root / "gene_stats.npz")
    ev = {}
    if eval_views:
        e = np.load(root / "eval_views.npz")
        for p in PS[1:]:
            for s in THIN_SEEDS:
                ev[(p, s)] = (e[f"c_p{p}_s{s}"], e[f"r_p{p}_s{s}"])
    return R6Data(cnt=z["cnt"], crest=z["crest"], pools=pools, pool_id=pid, Y=Y, valid=valid, plate=plate, fp=fp,
                  splits=splits, fold=fold, lines=lines, x_mean=st["x_mean"], x_sd=st["x_sd"],
                  tiebreak=st["tiebreak"], eval_views=ev, root=root)


def dump_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
