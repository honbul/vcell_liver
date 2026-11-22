"""Fit linear T4 setting using scaffold- and plate-held inner source predictions."""

import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from src.r11_chemical_probe.common import write_json
from src.r19_plan.t4_scaffold_transfer import scaffold

from .data import SourceWells, all_pairs, contexts, parent_folds, purged_inner
from .linear import base_grid, point_fits, loss

V1 = (False, "pcr", 1., 16, "")


def optimize(splits):
    risks = defaultdict(list)
    grid = list(base_grid())
    for train, validation, _ in splits:
        for cfg, prediction, _ in point_fits(grid, train, validation.x, validation.rows):
            risks[cfg].append(loss(validation, prediction))
    mean = {p: np.mean(s) for p, s in risks.items()}
    shortlist = [min((cfg for cfg in mean if cfg[1] == m), key=mean.get)
                 for m in ("pcr", "reduced", "nuclear")]
    gated = [(*p[:-1], gate) for p in shortlist for gate in ("rms_dose", "chem")]
    for train, validation, _ in splits:
        for cfg, prediction, _ in point_fits(gated, train, validation.x, validation.rows):
            risks[cfg].append(loss(validation, prediction))
    rank = sorted((float(np.sqrt(np.mean(values))), setting) for setting, values in risks.items())
    return rank


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    ctx = contexts(args.reports, args.hepg2, args.phh)[args.cell]
    fold = int(args.group[1])
    held = {scaffold(p) for p in parent_folds(ctx)[fold]}
    source = SourceWells(ctx, args.group, held)
    reports = []
    for key in source.plans:
        before, after = source.data(key)
        splits = list(purged_inner(before, after, key))
        ranks = optimize(splits)
        champ = ranks[0][1]
        fit = all_pairs(before, after)
        _, _, model = next(point_fits([champ], fit, fit.x[:0], fit.rows.iloc[:0], need_model=True))
        joblib.dump(model, args.out / f"{key}.joblib", compress=2)
        joblib.dump(splits, args.out / f"{key}.source_only.joblib", compress=3)
        v1_rmse = next(value for value, setting in ranks if setting == V1)
        summary = {"slice": key, "source_fit_conditions": len(fit.rows), "inner_support": [s[2] for s in splits],
                   "rank16_rmse": v1_rmse, "selected": list(champ), "selected_inner_rmse": ranks[0][0],
                   "grid": [{"rmse": v, "config": list(c)} for v, c in ranks]}
        reports.append(summary)
        print(key, "rank16", round(v1_rmse, 5), "candidate", round(ranks[0][0], 5), champ, flush=True)
    write_json(args.out / "receipt.json", {"cell": args.cell, "group": args.group, "slices": reports})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reports", "hepg2", "phh", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--cell", choices=("HEPG2", "PHH"), required=True)
    parser.add_argument("--group", required=True)
    run(parser.parse_args())
