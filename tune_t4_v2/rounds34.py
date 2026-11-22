"""Evaluate Pan-LINCS adaptation and select temporal blends in the inner T4 folds."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zlib import crc32

import joblib
import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json

from .data import parent_weights
from .linear import loss, point_fits
from .neural import CONFIGS, cross_residual, fit as fit_neural
from .round2 import ARM_MODES
from .pretrain import load_pan
from .transfer import BUDGETS, adapt, centered_prior, fewshot

MIXES = (.2, .4, .65, .85)
TRANSFER_RANKS = (0, 4, 8)


def local_scalar(train, query):
    w = parent_weights(train.rows)
    mx, my = (np.average(z, axis=0, weights=w) for z in (train.x, train.y))
    x = train.x-mx
    y = train.y-my
    gain = float((x*y*w[:, None]).sum() / max((x*x*w[:, None]).sum(), 1e-5))
    return my + gain*(query-mx)


def prior_predictions(pan, train, validation, *, cell, seed, device):
    values = {"pan_zero_centered": centered_prior(pan, train.x, validation.rows, validation.x, device=device)}
    development = centered_prior(pan, train.x, train.rows, train.x, device=device)
    central = values["pan_zero_centered"]
    for rank in TRANSFER_RANKS:
        values[f"pan_all_rank{rank}"] = adapt(central, validation.rows, validation.x, train, development, 45., rank)
    if cell == "PHH":
        for budget in BUDGETS:
            small = fewshot(train, budget, seed+budget)
            if small is None:
                continue
            for rank in TRANSFER_RANKS:
                pred = centered_prior(pan, train.x, small.rows, small.x, device=device)
                values[f"pan_{budget}_rank{rank}"] = adapt(central, validation.rows, validation.x, small, pred, 45., rank)
            values[f"scalar_{budget}"] = local_scalar(small, validation.x)
            cfg = (False, "pcr", 100., min(8, budget-1), "")
            _, linear, _ = next(point_fits([cfg], small, validation.x, validation.rows))
            values[f"linear_{budget}"] = linear
    return values


def assemble(train, val, pan, config, neural, *, seed, device, cell):
    _, prediction, _ = next(point_fits([config], train, val.x, val.rows, need_model=True))
    arms = {"linear": prediction, "scalar": local_scalar(train, val.x)}
    name, multiplier = neural.split("/")
    if name != "linear":
        residual = cross_residual(train, config)
        fitted = fit_neural(train, residual, ARM_MODES[name], CONFIGS[name], seed, device)
        arms["linear_neural"] = prediction + float(multiplier)*fitted.predict(val.rows, val.x, seed=seed+1, device=device)
    arms.update(prior_predictions(pan, train, val, cell=cell, seed=seed, device=device))
    return arms


def exp4(arms):
    final = {}
    for base in ("linear", "linear_neural"):
        if base not in arms:
            continue
        for arm in ("pan_zero_centered", "pan_all_rank0", "pan_all_rank4", "pan_all_rank8"):
            for a in MIXES:
                final[f"{base}+{arm}/{a}"] = (1.-a)*arms[base] + a*arms[arm]
    return final


def run(args):
    args.out.mkdir(exist_ok=False, parents=True)
    prior = args.pretrain / "pan_model.joblib"
    pmeta = read_json(args.pretrain / "receipt.json")
    if sha256(prior) != pmeta["model_sha256"]:
        raise ValueError("Pan pretrained model changed after lock")
    pan = load_pan(prior)
    r1 = read_json(args.round1 / "receipt.json")
    r2 = read_json(args.round2 / "receipt.json")
    cell = r1["cell"]
    reports = []
    for row in r2["slices"]:
        key = row["slice"]
        settings = tuple(row["linear"])
        splits = joblib.load(args.round1 / f"{key}.source_only.joblib")
        errors = defaultdict(list)
        trace = []
        for index, (fit, val, _) in enumerate(splits):
            seed = crc32(f"{key}|{index}".encode())
            branches = assemble(fit, val, pan, settings, row["champion"], seed=seed, device=args.device, cell=cell)
            branches |= exp4(branches)
            mse = val.rows[["parent", "rna_plate", "dose_uM"]].copy().assign(inner=index)
            for name, pred in branches.items():
                errors[name].append(loss(val, pred))
                mse[name] = np.square(pred-val.y).mean(axis=1)
            trace.append(mse)
        measured = {name: float(np.sqrt(np.mean(s))) for name, s in errors.items() if len(s) == len(splits)}
        stage3 = [n for n in measured if n.startswith("pan_all")]
        stage4 = [n for n in measured if "+" in n or n in ("linear", "linear_neural") or n in stage3]
        primary = min(stage4, key=measured.get)
        ranked = sorted(stage4, key=measured.get)
        control = min(stage3, key=measured.get)
        selected_budgets = {}
        for budget in BUDGETS:
            eligible = [n for n in measured if n.startswith(f"pan_{budget}_rank")]
            if eligible:
                selected_budgets[str(budget)] = min(eligible, key=measured.get)
        pd.concat(trace, ignore_index=True).to_parquet(args.out / f"{key}.parquet", index=False)
        reports.append({"slice": key, "linear_config": row["linear"],
                        "round2": row["champion"], "round3_pan_complete": control, "round4_champion": primary,
                        "fewshot_pan": selected_budgets, "inner_rmse": measured,
                        "round4_top_three": ranked[:3]})
        print(key, "linear", round(measured["linear"], 5), "Pan", control, round(measured[control], 5),
              "Round4", primary, round(measured[primary], 5), flush=True)
    write_json(args.out / "receipt.json", {"cell": cell, "group": r1["group"], "slices": reports,
                                          "pretrain_sha256": sha256(prior),
                                          "round1_sha256": sha256(args.round1 / "receipt.json"),
                                          "round2_sha256": sha256(args.round2 / "receipt.json"),
                                          "score_labels_accessed": False})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("round1", "round2", "pretrain", "out"):
        cli.add_argument("--" + key, type=Path, required=True)
    cli.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    run(cli.parse_args())
