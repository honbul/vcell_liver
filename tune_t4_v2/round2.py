"""Quantify nonlinear residual improvement using purged inner T4 wells."""

import argparse
from collections import defaultdict
from pathlib import Path
from zlib import crc32

import joblib
import numpy as np
import pandas as pd
import torch

from src.r11_chemical_probe.common import read_json, write_json

from .linear import loss, point_fits
from .neural import CONFIGS, cross_residual, fit

SCALES = (.25, .5, .75, 1.)
ARM_MODES = {"denoising": "denoising", "film": "film", "attention": "attention",
             "film_shuffled": "film", "film_no_structure": "film"}


def paired(rows, pred_cols, seed):
    by_parent = rows.groupby("parent")[["baseline", *pred_cols]].mean()
    value = by_parent.to_numpy()
    rng = np.random.default_rng(seed)
    sampled = value[rng.integers(len(value), size=(5000, len(value)))].mean(axis=1)
    baseline = np.sqrt(sampled[:, 0])
    return {name: {
        "rmse": float(np.sqrt(value[:, i+1].mean())),
        "rmse_gain_percent": float(100*(1-np.sqrt(value[:, i+1].mean()/value[:, 0].mean()))),
        "ci99_percent": (100*np.quantile(1-np.sqrt(sampled[:, i+1])/baseline, [.005, .995])).tolist(),
    } for i, name in enumerate(pred_cols)}


def train_inner(splits, config, setting_key, device):
    risk = defaultdict(list)
    evidence = []
    for i, (train, val, meta) in enumerate(splits):
        seed = crc32(f"{setting_key}|{i}".encode())
        _, base, _ = next(point_fits([config], train, val.x, val.rows, need_model=True))
        residual = cross_residual(train, config)
        square = np.mean(np.square(base - val.y), axis=1)
        rows = val.rows[["parent", "rna_plate", "dose_uM"]].copy().assign(inner_fold=i, baseline=square)
        rows["early_rms"] = np.linalg.norm(val.x, axis=1) / np.sqrt(val.x.shape[1])
        risk["linear", 0.].append(loss(val, base))
        for arm, cfg in CONFIGS.items():
            model = fit(train, residual, ARM_MODES[arm], cfg, seed, device)
            naive = model.predict(val.rows, val.x, seed=seed+1, device=device)
            rows[arm] = np.square(base + .5 * naive - val.y).mean(axis=1)
            for alpha in SCALES:
                risk[arm, alpha].append(loss(val, base + alpha * naive))
        evidence.append(rows)
    average = {f"{arm}/{alpha}": float(np.sqrt(np.mean(v))) for (arm, alpha), v in risk.items()}
    best = min((k for k in average if "shuffle" not in k), key=average.get)
    return best, average, pd.concat(evidence, ignore_index=True)


def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA resource missing for nonlinear temporal training")
    args.out.mkdir(parents=True, exist_ok=False)
    origin = read_json(args.round1 / "receipt.json")
    report = []
    for row in origin["slices"]:
        key, cfg = row["slice"], tuple(row["selected"])
        splits = joblib.load(args.round1 / f"{key}.source_only.joblib")
        champion, metrics, evidence = train_inner(splits, cfg, key, args.device)
        report.append({"slice": key, "linear": list(cfg), "champion": champion, "metrics": metrics})
        evidence.assign(slice=key).to_parquet(args.out / f"{key}.parquet", index=False)
        print(key, "linear", round(metrics["linear/0.0"], 5), champion, round(metrics[champion], 5), flush=True)
    combined = pd.concat([pd.read_parquet(args.out / f"{row['slice']}.parquet") for row in report])
    comparisons = paired(combined, list(CONFIGS), seed=38268)
    write_json(args.out / "receipt.json", {"cell": origin["cell"], "group": origin["group"],
                                          "slices": report, "paired_fixed_half_scale": comparisons})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(parser.parse_args())
