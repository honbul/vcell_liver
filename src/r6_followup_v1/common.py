"""Shared plumbing of the R6 follow-up: read-only access to the original contract, source folds.

The original R6 contract (count cache, truth, split, evaluation views) is read from the original
worktree's assets and never written. Everything the follow-up produces goes to $R6F_ASSETS.

Source folds (plan 3.3): the three drug-structure-group folds inside the ORIGINAL TRAIN drugs
    fold(g) = floor( blake2b('r3-devfold|' + group SMILES) * 3 )
which is the rule the contract already carries (`d.fold`). It depends on the structure string only;
no response value enters it. Outer fold k: fit = TRAIN drugs with fold != k, held = fold == k.
Inner development split (hyper-parameter choices only): a fixed 20% of the outer-fit drugs,
    inner_dev(g) = blake2b('r6f-inner|' + group SMILES) < 0.2 .
The 70 historical VALIDATION drugs are used by no step; the 70 TEST drugs only by diagnosis tables
of existing predictions and by the single frozen re-evaluation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.depth_robust_r6 import data as D

TRACK, VERSION = "R6", "followup_v1"
FOLDS = (0, 1, 2)
SEEDS = (42, 43, 44)
INNER_FRAC = 0.2
PKEYS = {1.0: ["p1.0_s0"], 0.5: [f"p0.5_s{s}" for s in D.THIN_SEEDS], 0.25: [f"p0.25_s{s}" for s in D.THIN_SEEDS]}
VIEWS = [(1.0, 0)] + [(p, s) for p in D.PS[1:] for s in D.THIN_SEEDS]


def orig_assets() -> Path:
    return Path(os.environ["R6_ORIG_ASSETS"])


def assets() -> Path:
    return Path(os.environ["R6F_ASSETS"])


def load(eval_views: bool = True) -> D.R6Data:
    """The original, frozen R6 contract (read-only)."""
    return D.load(root=orig_assets() / "cache" / "contract_v1", v2data=os.environ["R6_V2DATA"], eval_views=eval_views)


def train_drugs(d: D.R6Data) -> list[int]:
    return [int(g) for g in d.splits.group_index[d.splits.split == "train"]]


def test_drugs(d: D.R6Data) -> list[int]:
    return [int(g) for g in d.splits.group_index[d.splits.split == "test"]]


def inner_dev_flag(d: D.R6Data) -> np.ndarray:
    smi = d.splits.group_smiles.astype(str).to_numpy()
    return np.array([D.bhash("r6f-inner|" + s) < INNER_FRAC for s in smi])


def source_split(d: D.R6Data, fold: int, inner: bool = False):
    """(fit drugs, evaluation drugs) of a source fold. inner=True: fit = outer-fit minus inner-dev,
    evaluation = inner-dev (the held fold is then not touched at all)."""
    tr = train_drugs(d)
    fit = [g for g in tr if d.fold[g] != fold]
    held = [g for g in tr if d.fold[g] == fold]
    if not inner:
        return fit, held
    flag = inner_dev_flag(d)
    return [g for g in fit if not flag[g]], [g for g in fit if flag[g]]


def split_spec(d: D.R6Data, setting: str, fold: int | None = None, inner: bool = False):
    if setting == "source":
        assert fold in FOLDS
        return source_split(d, fold, inner)
    if setting == "final":
        assert not inner
        return train_drugs(d), test_drugs(d)
    raise ValueError(setting)


def pool_of(d: D.R6Data, L: np.ndarray, Dg: np.ndarray) -> np.ndarray:
    """Pool id (primary plate x cell line) of every condition."""
    pid = d.pool_id[d.plate[L, Dg], L]
    assert (pid >= 0).all()
    return pid


def plate_weights() -> np.ndarray:
    return np.load(Path(os.environ["R6_V2DATA"]) / "plate_weights.npy")


def load_pred(path, L=None, Dg=None) -> dict:
    """A stored prediction file: {key: float32 [n_cond, G]}; checks the condition order if given."""
    z = np.load(path)
    if L is not None:
        assert (z["line"] == L).all() and (z["drug"] == Dg).all(), path
    return {k: z[k].astype(np.float32) for k in z.files if k not in ("line", "drug")}


def std_fields(**kw) -> dict:
    out = dict(track=TRACK, version=VERSION)
    out.update(kw)
    return out


def git_commit() -> str:
    return os.popen("git rev-parse HEAD").read().strip()


def dump_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
