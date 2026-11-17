"""The two fixed settings every method is fitted and scored in.

    dev     8 metadata-chosen cell lines; fit on TRAIN drugs in dev folds {0, 1}; score on fold 2
    final   all eligible cell lines; fit on all TRAIN drugs; score on the held-out TEST drugs
            (run once, after freeze_manifest.json exists)

The prior study's VALIDATION drugs are used by neither setting.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D
from .metrics import Scorer, fit_pca_kmeans

DEV_EVAL_FOLD = 2


@dataclass
class Setting:
    name: str
    d: D.Data
    fit_wells: pd.DataFrame
    eval_wells: pd.DataFrame
    scale: dict
    s_resid: np.ndarray
    ctx: dict
    scorer: Scorer
    lines: list
    fit_drugs: list
    eval_drugs: list


def split_spec(d: D.Data, name: str):
    train_drugs = [int(g) for g in d.splits.group_index[d.splits.split == "train"]]
    if name == "dev":
        lines = list(d.dev_lines)
        fit = [g for g in train_drugs if d.fold[g] != DEV_EVAL_FOLD]
        ev = [g for g in train_drugs if d.fold[g] == DEV_EVAL_FOLD]
        role = "train"
    elif name == "final":
        lines = sorted(int(l) for l in np.flatnonzero(d.valid.any(1)))
        fit = train_drugs
        ev = [int(g) for g in d.splits.group_index[d.splits.split == "test"]]
        role = "test"
    else:
        raise ValueError(name)
    return lines, fit, ev, role


def build(name: str, device: str = "cpu", d: D.Data | None = None) -> Setting:
    d = d or D.load()
    lines, fit, ev, role = split_spec(d, name)
    assert not set(fit) & set(ev)
    fw = D.eval_wells(d, "train", lines, fit, min_n=8)
    ew = D.eval_wells(d, role, lines, ev, min_n=32)
    scale = D.fit_gene_scale(d, fw)
    pm, pc, km = fit_pca_kmeans(d, fw, scale["sd_within"])
    scorer = Scorer(scale["sd_within"], pm, pc, {k: torch.as_tensor(v, device=device) for k, v in km.items()}, device)
    return Setting(name=name, d=d, fit_wells=fw, eval_wells=ew, scale=scale,
                   s_resid=D.residual_scale(scale["var_resid"]), ctx=D.context_stats(d),
                   scorer=scorer, lines=lines, fit_drugs=fit, eval_drugs=ev)


def run_dir(*parts) -> Path:
    p = Path(os.environ["R3_ASSETS"]) / "runs" / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
