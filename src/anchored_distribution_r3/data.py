"""Data access for study R3: real cells, wells, controls, splits and the fixed gene scale.

Vocabulary
    group / well   one (well sample, cell line) slice of real cells
    condition      one (cell line, drug structure group); may have several wells
    role           the prior study's drug-group split: train / val / test (plus ctrl)
    dev fold       a fixed 3-fold partition of the TRAIN drug groups by blake2b(group SMILES);
                   all development happens inside it

Nothing here reads a validation- or test-role treated cell unless the caller asks for that role.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_CAP = 64            # cells per well used as "observed" and generated; fixed for every method
N_DEV_LINES = 8
N_FOLDS = 3


def bhash(s: str) -> float:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") / 2.0 ** 64


@dataclass
class Data:
    X: np.ndarray                 # [n_cells, G] float32, memory-mapped
    lib: np.ndarray
    wells: pd.DataFrame           # treated + ctrl groups
    Y: np.ndarray                 # [L, D, G] full-well condition mean delta (prior study's truth)
    valid: np.ndarray             # [L, D]
    fp: np.ndarray                # [D, 2048] ECFP4 bits
    dmso_mean: np.ndarray         # [P, L, G] full-pool same-plate DMSO mean
    splits: pd.DataFrame
    ctrl_slices: dict             # (plate_index, line) -> list of (start, end) of its DMSO wells
    fold: np.ndarray              # [D] dev fold of each TRAIN drug group, -1 otherwise
    dev_lines: list

    def cells(self, row, n=None) -> np.ndarray:
        """First n cells (barcode-hash order) of a well row."""
        e = row.end if n is None else min(row.end, row.start + n)
        return np.asarray(self.X[row.start:e])

    def ctrl_cells(self, plate_index: int, line: int) -> np.ndarray:
        """All sampled DMSO cells of a (plate, line), interleaved across its wells in hash order."""
        parts = [np.asarray(self.X[s:e]) for s, e in self.ctrl_slices[(plate_index, line)]]
        m = max(len(p) for p in parts)
        idx = [(i, k) for i in range(m) for k in range(len(parts)) if i < len(parts[k])]
        return np.stack([parts[k][i] for i, k in idx])

    def n_ctrl(self, plate_index: int, line: int) -> int:
        return sum(e - s for s, e in self.ctrl_slices[(plate_index, line)])


def load(cache: str | None = None, v2data: str | None = None,
         manifest_dir: str = "manifests/chemcpa_moe_v2") -> Data:
    cache = Path(cache or os.environ["R3_ASSETS"]) / "cache" if cache is None else Path(cache)
    v2 = Path(v2data or os.environ["R3_V2DATA"])
    X = np.load(cache / "X.npy", mmap_mode="r")
    cz = np.load(cache / "cells.npz")
    wells = pd.read_csv(cache / "wells.tsv", sep="\t")
    Y = np.load(v2 / "Y.npy")
    valid = np.load(v2 / "valid.npy")
    fp = np.load(v2 / "fp.npy").astype(np.float32)
    dm = np.load(v2 / "dmso_plate_mean.npy")
    splits = pd.read_csv(Path(manifest_dir) / "splits.tsv", sep="\t").sort_values("group_index")
    assert Y.shape[:2] == valid.shape == (dm.shape[1], len(splits)) and Y.shape[2] == X.shape[1]
    ctrl = {}
    for r in wells[wells.kind == "ctrl"].itertuples():
        ctrl.setdefault((r.plate_index, r.line), []).append((r.start, r.end))
    fold = np.full(len(splits), -1, int)
    for r in splits.itertuples():
        if r.split == "train":
            fold[r.group_index] = int(bhash("r3-devfold|" + str(r.group_smiles)) * N_FOLDS)
    d = Data(X=X, lib=cz["library_size"], wells=wells, Y=Y, valid=valid, fp=fp, dmso_mean=dm,
             splits=splits, ctrl_slices=ctrl, fold=fold, dev_lines=[])
    d.dev_lines = choose_dev_lines(d)
    return d


def choose_dev_lines(d: Data) -> list:
    """Eight development cell lines by metadata coverage only (never by a response score).

    Rank: number of valid TRAIN conditions (desc), sampled TRAIN treated cells (desc), cell line id.
    """
    w = d.wells[(d.wells.kind == "treated") & d.wells.valid & (d.wells.role == "train")]
    g = w.groupby("line").agg(n_cond=("drug", "nunique"), n_cells=("n_cells", "sum"),
                              cid=("cell_line_id", "first")).reset_index()
    g = g.sort_values(["n_cond", "n_cells", "cid"], ascending=[False, False, True])
    return [int(x) for x in g.line.head(N_DEV_LINES)]


def eval_wells(d: Data, role: str, lines=None, drugs=None, min_n: int = 32) -> pd.DataFrame:
    """Valid treated wells of a role with the cell count every method must match.

    n_eval = min(observed cells, available same-plate control cells, N_CAP). Wells whose n_eval is
    below `min_n` are dropped and counted by the caller.
    """
    w = d.wells[(d.wells.kind == "treated") & d.wells.valid & (d.wells.role == role)].copy()
    if lines is not None:
        w = w[w.line.isin(list(lines))]
    if drugs is not None:
        w = w[w.drug.isin(list(drugs))]
    w["n_ctrl"] = [d.n_ctrl(p, l) if (p, l) in d.ctrl_slices else 0 for p, l in zip(w.plate_index, w.line)]
    w["n_eval"] = np.minimum(np.minimum(w.n_cells, w.n_ctrl), N_CAP)
    w.attrs["n_dropped_small"] = int((w.n_eval < min_n).sum())
    return w[w.n_eval >= min_n].reset_index(drop=True)


def fit_gene_scale(d: Data, train_wells: pd.DataFrame, floor_q: float = 0.10) -> dict:
    """Per-gene scales from TRAINING wells only.

    sd_within   pooled within-well standard deviation (the metric's standardisation), floored at
                the `floor_q` quantile over genes
    var_resid   pooled within-well variance (the generator's residual scale is derived from it)
    mean_abs    grand mean of the training cells (used to standardise absolute expression)
    sd_total    total standard deviation over the training cells
    """
    G = d.X.shape[1]
    ss = np.zeros(G); dof = 0
    s1 = np.zeros(G); s2 = np.zeros(G); n = 0
    for r in train_wells.itertuples():
        x = np.asarray(d.X[r.start:r.end], dtype=np.float64)
        if len(x) < 2:
            continue
        ss += ((x - x.mean(0)) ** 2).sum(0); dof += len(x) - 1
        s1 += x.sum(0); s2 += (x ** 2).sum(0); n += len(x)
    var_resid = ss / dof
    sd = np.sqrt(var_resid)
    sd_within = np.maximum(sd, np.quantile(sd, floor_q))
    mean_abs = s1 / n
    sd_total = np.sqrt(np.maximum(s2 / n - mean_abs ** 2, 1e-12))
    return dict(sd_within=sd_within.astype(np.float32), var_resid=var_resid.astype(np.float32),
                mean_abs=mean_abs.astype(np.float32), sd_total=sd_total.astype(np.float32),
                n_cells=int(n), floor_q=floor_q)


def residual_scale(var_resid: np.ndarray, floor_q: float = 0.05) -> np.ndarray:
    """Plan default: per-gene residual variance shrunk 50:50 towards the median variance, then a
    floor at a pre-specified quantile, returned as a positive standard-deviation scale."""
    v = 0.5 * var_resid + 0.5 * np.median(var_resid)
    s = np.sqrt(v)
    return np.maximum(s, np.quantile(s, floor_q)).astype(np.float32)


def context_stats(d: Data, eps: float = 1e-3) -> dict:
    """Control-population context per (plate, line): full-pool mean and sampled-cell log variance."""
    out = {}
    for (p, l) in d.ctrl_slices:
        c = d.ctrl_cells(p, l).astype(np.float64)
        out[(p, l)] = (d.dmso_mean[p, l].astype(np.float32),
                       np.log(c.var(0, ddof=1) + eps).astype(np.float32) if len(c) > 1
                       else np.full(c.shape[1], np.log(eps), np.float32))
    return out


def sha256_file(path: str | Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dump_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
