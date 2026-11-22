"""Nested five-fold comparison of chemical and foundation rankers."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import KINDS, RegisteredData, check_disjoint, inner_groups
from .features import FM_NAMES, Features, subset, tanimoto
from .rankers import GRIDS, REPRESENTATIONS, train


def ranking(y, score):
    return {"auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score))}


def tune_models(table, blocks, methods, *, gpu):
    target = table.label.to_numpy()
    folds = list(inner_groups(table))
    records, selected, predictions = [], {}, {}
    for method in methods:
        fits = []
        for representation in REPRESENTATIONS[method]:
            for cfg in GRIDS[method]:
                raw = np.empty(len(table))
                local_scores = []
                for tr, va in folds:
                    model = train(method, representation, cfg, subset(blocks, tr), target[tr], gpu)
                    raw[va] = model.raw(subset(blocks, va))
                    result = ranking(target[va], raw[va])
                    local_scores.append(.60 * result["auroc"] + .40 * result["auprc"])
                stats = {"method": method, "representation": representation, "config": cfg,
                         "utility": float(np.mean(local_scores)), **ranking(target, raw)}
                records.append(stats)
                fits.append((stats, raw))
        best, predictions[method] = max(fits, key=lambda x: x[0]["utility"])
        selected[method] = {"representation": best["representation"], "config": best["config"]}
    return records, selected, predictions


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    data = RegisteredData(args.data)
    feature_bank = Features(args.fm, args.chemberta)
    table = data.fit_table(args.kind, args.fold)
    query = data.outer_query(args.kind, args.fold, development=True)
    check_disjoint(table, query)
    truth = data.outcomes(args.kind, args.fold, development=True).loc[query.index, "label"]
    y = table.label.to_numpy()
    blocks, q = [feature_bank.arrays(rows) for rows in (table, query)]
    methods = args.methods.split(",")
    records, configs, oof = tune_models(table, blocks, methods, gpu=args.gpu)
    forecasts, models = {}, {}
    for method, cfg in configs.items():
        models[method] = train(method, cfg["representation"], cfg["config"], blocks, y, args.gpu)
        forecasts[method] = models[method].raw(q)
    inner = pd.DataFrame(oof, index=table.index).assign(label=y, group=table.group)
    outer = pd.DataFrame(forecasts, index=query.index).assign(
        label=truth,
        nearest_tanimoto=tanimoto(q["base"], blocks["base"], mode="bits").max(axis=1))
    inner.to_parquet(args.out / "inner.parquet")
    outer.to_parquet(args.out / "dev_outer.parquet")
    joblib.dump(models, args.out / "models.joblib", compress=2)
    metrics = {method: ranking(truth, forecasts[method]) for method in forecasts}
    report = {
        "kind": args.kind, "fold": args.fold, "gpu": args.gpu,
        "fit_n": len(table), "development_outer_n": len(query), "increment_access": "identity mask only",
        "fm_hashes": feature_bank.hashes, "fm_coverage_fit": {
            key: int(np.isfinite(blocks[key]).all(axis=1).sum()) for key in FM_NAMES
        }, "selected": configs, "grid": records, "outer_metrics": metrics,
    }
    (args.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "fm", "chemberta", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--methods", default=",".join(GRIDS))
    parser.add_argument("--gpu", action="store_true")
    run(parser.parse_args())
