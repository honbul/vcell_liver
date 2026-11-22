"""Develop lean two-axis predictors in source-only lineage CV."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vc_build.external import KEY
from vc_build.io import publish_json, sha256
from vc_build.v0plus_cv import attach_native, summarize, validation_roster

from .lean import CONFIGS, fm_paths, fm_read, forecast, liver_mask, train
from .source import grouped, load


def warm_fit(fit, rna, banks, split_seed, device, candidates=None):
    predictions = []
    model_utility = []
    selected = candidates or CONFIGS
    for i, (tr, va) in enumerate(grouped(fit, 2)):
        tune = list(tr.cell_id.unique())
        q = validation_roster(va)
        for name, config in selected.items():
            b = banks if name=="two_tower_cmap" else {k:v for k,v in banks.items() if k!="cmap"}
            learner = train(tr, rna.loc[tune], b, config, device=device, seed=split_seed * 37 + i)
            pred = forecast(learner, q, rna, banks, device=device)
            predictions.append(q.assign(method=name, forecast=pred, inner=i))
    return pd.concat(predictions, ignore_index=True)


def run(args):
    args.out.mkdir(exist_ok=False, parents=True)
    rows, rna, _ = load(args.bundle)
    train_rows, val = list(grouped(rows, 3))[args.fold]
    banks, sha = fm_paths(args.fm)
    banks["cmap"], sha["cmap"] = fm_read(args.cmap)
    inner = warm_fit(train_rows, rna, banks, 781 + args.fold, args.device)
    parts = []
    for i, (_, held) in enumerate(grouped(train_rows, 2)):
        forecast_data = inner.loc[inner.inner.eq(i)].rename(columns={"forecast": "predicted_value"})
        parts.append(attach_native(forecast_data, held, i))
    summary = summarize(pd.concat(parts, ignore_index=True))
    utility = summary.assign(utility=.3*summary.spearman + .2*summary.pearson +
                             .2*summary.macro_ndcg_at_5 + .1*summary.macro_top5_overlap + .2*summary.macro_spearman)
    ranked = utility.groupby("method").utility.mean().sort_values(ascending=False)
    champion = str(ranked.index[0])
    q = validation_roster(val)
    models, predictions = {}, []
    for name, cfg in CONFIGS.items():
        b = banks if name=="two_tower_cmap" else {k:v for k,v in banks.items() if k!="cmap"}
        models[name] = train(train_rows, rna.loc[train_rows.cell_id.unique()], b, cfg,
                             device=args.device, seed=32912+args.fold)
        predicted = forecast(models[name], q, rna, banks, device=args.device)
        predictions.append(q[KEY].assign(method=name, predicted_value=predicted))
    outer = attach_native(pd.concat(predictions, ignore_index=True), val, args.fold)
    subset_ids = val.cell_id.loc[liver_mask(val.tissue_origin)].unique()
    liver = outer.loc[outer.cell_id.isin(subset_ids)]
    inner.to_parquet(args.out / "inner.parquet", index=False)
    outer.to_parquet(args.out / "dev_outer.parquet", index=False)
    joblib.dump(models, args.out / "neural.joblib", compress=3)
    report = {"fold": args.fold, "device": args.device, "verified_fm_sha256": sha,
              "fm_train_coverage": {k: int(len(set(v.index) & set(train_rows.cell_id if k in
                                                                   ("geneformer", "scgpt") else train_rows.compound_id)))
                                    for k, v in banks.items()},
              "selected": champion, "inner": summary.to_dict("records"),
              "outer": summarize(outer).to_dict("records"), "liver_held_cells": len(subset_ids),
              "liver_held": summarize(liver).replace({np.nan: None}).to_dict("records") if len(liver) else []}
    publish_json(args.out / "receipt.json", report)
    print(ranked.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "fm", "out", "cmap"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(3), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run(parser.parse_args())
