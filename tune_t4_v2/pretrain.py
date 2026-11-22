"""Learn a multi-cell temporal operator using the protected nonhepatic LINCS corpus."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn

from src.r11_chemical_probe.common import sha256, write_json
from src.r19_plan.t4_scaffold_transfer import scaffold

from .linear import conditions
from .neural import torch_matrix

SETTINGS = ((128, .10, False), (192, .18, True))


class PanNet(nn.Module):
    def __init__(self, genes, contexts, cells, width, drop, film):
        super().__init__()
        self.skip = nn.Sequential(nn.Linear(genes, width // 2, bias=False),
                                  nn.Linear(width // 2, genes, bias=False))
        self.cell = nn.Embedding(cells+1, 16, padding_idx=0)
        self.gene = nn.Sequential(nn.Linear(genes, width), nn.LayerNorm(width), nn.GELU())
        self.ctx = nn.Sequential(nn.Linear(contexts + 16, width), nn.LayerNorm(width), nn.GELU())
        self.interact = nn.Linear(width, width*2) if film else None
        self.body = nn.Sequential(nn.Dropout(drop), nn.Linear(width, width), nn.GELU(), nn.Linear(width, genes))
        self.drop = drop

    def forward(self, x, c, cell):
        if self.training:
            cell = torch.where(torch.rand(len(cell), device=x.device) < .42, 0, cell)
        genes = self.gene(x)
        cond = self.ctx(torch.cat((c, self.cell(cell)), dim=1))
        if self.interact is None:
            mixed = genes + cond
        else:
            scale, shift = self.interact(cond).chunk(2, dim=1)
            mixed = genes * (1. + .5*scale.tanh()) + shift
        return self.skip(x) + self.body(mixed)


@dataclass
class PanFit:
    setting: tuple
    x_mean: np.ndarray
    scale: np.ndarray
    y_mean: np.ndarray
    c_mean: np.ndarray
    c_scale: np.ndarray
    cellular_ids: dict
    weights: dict
    genes: np.ndarray

    def predict(self, rows, x, *, input_mean=None, output_anchor=None, device="cpu"):
        model = PanNet(len(self.genes), len(self.c_mean), len(self.cellular_ids), *self.setting).to(device)
        model.load_state_dict(self.weights)
        model.eval()
        x_mean = self.x_mean if input_mean is None else input_mean
        y_mean = self.y_mean if output_anchor is None else output_anchor
        c = (conditions(rows, x, True) - self.c_mean) / self.c_scale
        xp = np.clip((x-x_mean)/self.scale, -12, 12)
        index = rows.cell_id.map(self.cellular_ids).fillna(0).to_numpy(int) if "cell_id" in rows else np.zeros(len(x), int)
        result = []
        with torch.inference_mode():
            for ix in np.array_split(np.arange(len(x)), 1+len(x)//512):
                if len(ix):
                    z = model(torch_matrix(xp[ix], device), torch_matrix(c[ix], device),
                              torch.as_tensor(index[ix], device=device)).cpu().numpy()
                    result.append(z)
        return y_mean + np.concatenate(result, axis=0)


def load_pan(path):
    """Open the legacy G1 module-mode pretrain checkpoint without editing it."""
    import __main__

    previous = getattr(__main__, "PanFit", None)
    __main__.PanFit = PanFit
    try:
        return joblib.load(path)
    finally:
        if previous is None:
            del __main__.PanFit
        else:
            __main__.PanFit = previous


def datasets(root):
    meta = json.loads((root / "receipt.json").read_text())
    rows, xs, ys = [], [], []
    genes = None
    for study in meta["studies"]:
        if any(study[tag] for tag in ("protected_connectivity_overlap", "protected_scaffold_overlap")):
            raise ValueError("Protected structures entered pan pretraining")
        name = study["study"]
        path = root / f"{name}.npz"
        if sha256(path) != study["pairs_sha256"]:
            raise ValueError(f"Pan matrix changed after freeze: {name}")
        with np.load(path, allow_pickle=False) as parts:
            x, y, columns = [parts[key] for key in ("x", "y", "genes")]
        if genes is not None and not np.array_equal(genes, columns):
            raise ValueError("Pan studies disagree on landmark order")
        genes = columns
        metadata = pd.read_parquet(root / f"{name}.parquet")
        if len(metadata) != len(x) or x.shape != y.shape:
            raise ValueError("Pan exact-pair inventory differs from the matrix")
        rows.append(metadata.rename(columns={"dose": "dose_uM"}))
        xs.append(x)
        ys.append(y)
    return pd.concat(rows, ignore_index=True), np.concatenate(xs), np.concatenate(ys), genes


def train(rows, x, y, genes, setting, epochs, seed, device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Missing CUDA allocation for temporal pretraining")
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    random = np.random.default_rng(seed)
    per_parent = rows.groupby(["cell_id", "parent"]).parent.transform("size").to_numpy()
    per_cell = rows.groupby("cell_id").cell_id.transform("size").to_numpy()
    weight = 1. / (np.sqrt(per_cell) * np.sqrt(per_parent))
    weight /= weight.mean()
    xm, ym = (np.average(array, axis=0, weights=weight) for array in (x, y))
    dispersion = np.maximum(np.sqrt(np.average(np.square(x-xm), axis=0, weights=weight)), .3)
    query = conditions(rows, x, True)
    cm = query.mean(axis=0)
    scale = np.maximum(query.std(axis=0), .4)
    c = torch_matrix((query-cm)/scale, device)
    id_map = {name: i for i, name in enumerate(sorted(rows.cell_id.unique()), 1)}
    ids = torch.as_tensor(rows.cell_id.map(id_map).to_numpy(int), device=device)
    x = torch_matrix(np.clip((x-xm)/dispersion, -12, 12), device)
    y = torch_matrix(y-ym, device)
    w = torch_matrix(weight, device)
    model = PanNet(len(genes), c.shape[1], len(id_map), *setting).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0018, weight_decay=.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.00018)
    for epoch in range(epochs):
        model.train()
        total = []
        for ix in np.array_split(random.permutation(len(rows)), 1+len(rows)//256):
            inputs = x[ix] + .04*torch.randn_like(x[ix])
            p = model(inputs, c[ix], ids[ix])
            errors = torch.nn.functional.smooth_l1_loss(p, y[ix], reduction="none", beta=.75).mean(dim=1)
            value = (errors*w[ix]).mean()
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimizer.step()
            total.append(value.detach())
        scheduler.step()
        if epoch in (0, epochs-1):
            print("pan pretrain", setting, "epoch", epoch+1, "loss", float(torch.stack(total).mean()), flush=True)
    return PanFit(setting, xm, dispersion, ym, cm, scale, id_map,
                  {name: value.detach().cpu() for name, value in model.state_dict().items()}, genes)


def run(args):
    args.out.mkdir(exist_ok=False, parents=True)
    rows, x, y, genes = datasets(args.pairs)
    train_index, held = next(GroupShuffleSplit(n_splits=1, test_size=.18, random_state=78230).split(
        x, groups=rows.parent.map(scaffold)))
    result = []
    for setting in SETTINGS:
        cv = train(rows.iloc[train_index], x[train_index], y[train_index], genes, setting, 18, 174024, args.device)
        prediction = cv.predict(rows.iloc[held], x[held], device=args.device)
        metrics = {"setting": setting, "rmse": float(np.sqrt(np.mean(np.square(prediction-y[held]))))}
        result.append(metrics)
    champion = min(result, key=lambda item: item["rmse"])["setting"]
    fitted = train(rows, x, y, genes, champion, 22, 174024, args.device)
    filename = args.out / "pan_model.joblib"
    joblib.dump(fitted, filename, compress=3)
    write_json(args.out / "receipt.json", {"device": args.device, "pairs": len(rows), "cells": int(rows.cell_id.nunique()),
                                          "parents": int(rows.parent.nunique()), "validation_pairs": len(held),
                                          "cv": result, "selected": champion, "model_sha256": sha256(filename),
                                          "source_receipt_sha256": sha256(args.pairs / "receipt.json")})


if __name__ == "__main__":
    from .pretrain import PanFit

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pairs", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    run(cli.parse_args())
