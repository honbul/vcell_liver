"""Read immutable v1 packets and build fit-only fold state."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import rankdata

from vc_build.io import sha256
from nextgen_h1_anchor.inputs import PAN, fold_order
from nextgen_h1_anchor.models import assay_prior, chemistry, empirical, mean_scale, rna_forecast, rna_oof
from nextgen_h1_anchor.tune import scaled_assay

BASE = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong")
ROOT = BASE / "nextgen_v2/h1_v2"
V1 = BASE / "nextgen_h123/h1_anchor"
STUDY = V1 / "study-v1"
TASKS = [f"{study}-{fold}" for study in (*PAN, "LIMORE", "LICOB") for fold in range(3)] + ["LIMORE_to_LICOB-0"]
BUDGETS = (0, 1, 3, 5, 8, 12)
PRIMARY_BUDGETS = BUDGETS[:-1]


def source(task, query=False, root=STUDY):
    study, fold = task.rsplit("-", 1)
    stamps = json.loads((root / "receipt.json").read_text())["files"]

    def verified(rel):
        path = root / rel
        if sha256(path) != stamps[rel]:
            raise ValueError(f"Original input hash mismatch: {rel}")
        return path

    with np.load(verified(f"{study}/{fold}-develop.npz"), allow_pickle=False) as store:
        dev = dict(store)
    meta = pd.read_csv(verified(f"{study}/chemistry.tsv"), sep="\t", index_col=0).fillna("")
    if list(meta.index) != list(dev["panel"]):
        raise ValueError("Drug axes disagree")
    if not query:
        return study, dev, meta
    with np.load(verified(f"{study}/{fold}-query.npz"), allow_pickle=False) as store:
        held = dict(store)
    if set(dev["groups"]) & set(held["groups"]) or not np.array_equal(dev["panel"], held["panel"]):
        raise ValueError("Invalid fixed train/test or drug support")
    return study, dev, meta, held


def partitions(groups, salt, n=3):
    keys = fold_order(groups, salt)
    masks = [np.isin(groups, keys[i::n]) for i in range(n)]
    for mask in masks:
        if not mask.any() or mask.all():
            raise ValueError(f"Insufficient donor groups for {salt}")
        yield ~mask, mask


def rna_projection(x, later, dimensions=16):
    if not x.shape[1]:
        return np.zeros((len(x), dimensions)), np.zeros((len(later), dimensions))
    # Dual PCA provides bounded compute and fits every axis on the training lines only.
    mu = x.mean(axis=0)
    centered = x - mu
    gram = centered @ centered.T
    rank = min(dimensions, len(gram) - 1)
    val, vec = eigh(gram, subset_by_index=(len(gram) - rank, len(gram) - 1))
    root = np.sqrt(np.maximum(val, 1e-5))
    x_pc = (vec / root) @ np.diag(root ** 2)
    val_pc = ((later - mu) @ centered.T) @ (vec / root)
    spread = np.maximum(x_pc.std(axis=0), 1e-5)
    return tuple(np.pad(pc / spread, ((0, 0), (0, dimensions - rank))) for pc in (x_pc, val_pc))


def drug_features(meta):
    chem, sim = chemistry(meta)
    from vc_build.external import _chemistry
    fp = np.asarray(_chemistry(meta.reset_index(), fp_size=2048).loc[meta.index])[:, :2048]
    intersections = fp @ fp.T
    totals = fp.sum(axis=1)
    tan = intersections / np.maximum(1, totals[:, None] + totals - intersections)
    targets = (sim - .7 * tan) / .3
    val, vec = np.linalg.eigh(chem)
    emb = vec[:, -24:] * np.sqrt(np.maximum(val[-24:], 0))
    return chem, sim, tan, targets, emb


@dataclass
class Context:
    study: str
    train: np.ndarray
    valid: np.ndarray | None
    groups: np.ndarray
    valid_groups: np.ndarray
    train_tissue: np.ndarray
    tissue: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    offset: float
    span: float
    bfit: np.ndarray
    bvalid: np.ndarray
    pc_fit: np.ndarray
    pc_valid: np.ndarray
    rna_fit_kernel: np.ndarray
    rna_valid_kernel: np.ndarray
    chem: np.ndarray
    similarity: np.ndarray
    tanimoto: np.ndarray
    targets: np.ndarray
    embedding: np.ndarray
    prior_pan: np.ndarray
    cov: np.ndarray
    cov_rna: np.ndarray
    levels: bool


def build(study, dev, tr, xvalid, valid_groups, tissue, drug, valid=None, *, pca=True, use_rna=True):
    chem, sim, tan, targets, emb = drug
    transfer = study == "LIMORE_to_LICOB"
    y, off, span = scaled_assay(dev["y"][tr], transfer)
    mu, sd = mean_scale(y)
    if not np.all(np.isfinite(mu)):
        raise ValueError(f"Empty drug fit column in {study}")
    ids = dev["groups"][tr]
    zero_pc = (np.zeros((len(y), 16)), np.zeros((len(xvalid), 16)))
    if not transfer and use_rna:
        training_rna, query_rna = dev["x"][tr].astype(float), xvalid.astype(float)
        bval, bfit, cross, gram = rna_oof(y, training_rna, query_rna, chem, ids)
        px, pv = rna_projection(training_rna, query_rna) if pca else zero_pc
    else:
        bfit = np.broadcast_to(mu, y.shape).copy()
        bval = np.broadcast_to(mu, (len(xvalid), len(mu))).copy()
        px, pv = zero_pc
        gram, cross = np.eye(len(y)), np.zeros((len(xvalid), len(y)))
    priors = []
    for peer in dev["prior_names"]:
        donor_ids = dev[f"ids_{peer}"]
        mask = np.isin(donor_ids, ids) if study in PAN else ~np.isin(donor_ids, valid_groups)
        priors.append(dev[f"prior_{peer}"][mask])
    prior = assay_prior(priors, len(mu))
    return Context(study, y, None if valid is None else (valid - off) / span, ids, valid_groups, dev["tissue"][tr],
                   tissue, mu, sd, off, span, bfit, bval, px, pv, gram, cross, chem, sim, tan, targets, emb,
                   np.eye(len(mu)) if prior is None else prior,
                   empirical((y - mu) / sd), empirical((y - bfit) / sd), transfer)


def training_context(study, dev, tr, va, drug, *, pca=True, use_rna=True):
    return build(study, dev, tr, dev["x"][va], dev["groups"][va], dev["tissue"][va], drug,
                 dev["y"][va], pca=pca, use_rna=use_rna)


def within_drug(y, p):
    from vc_build.v0plus_score import correlations
    yy = rankdata(y, axis=0, nan_policy="omit")
    pp = rankdata(np.where(np.isfinite(y), p, np.nan), axis=0, nan_policy="omit")
    result = correlations(yy[None], pp[None])[0]
    return float(np.nanmean(result)) if np.isfinite(result).any() else 0.
