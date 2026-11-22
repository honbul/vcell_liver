"""Nested directed graph models with nonoverlapping in-vitro mechanism supervision."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from liver_dili.splits import digest

from .data import KINDS, RegisteredData, check_disjoint, inner_groups
from .graph import train_graph
from .round1 import ranking

CONFIGS = {
    "dmpnn": ((80, 3, .17, 0., 36), (128, 4, .22, 0., 48)),
    "mechanism_dmpnn": ((80, 3, .17, .25, 36), (128, 4, .22, .50, 48)),
}


def safe_auxiliary(path, exclusions):
    receipt = json.loads((path / "receipt.json").read_text())
    if receipt["mechanistic_sha256"] != digest(path / "mechanistic.parquet"):
        raise ValueError("The auxiliary assay data differ from their receipt")
    panel = pd.read_parquet(path / "mechanistic.parquet")
    known = pd.read_csv(exclusions, sep="\t", usecols=["connectivity", "scaffold"])
    allowed = panel.loc[~panel.connectivity.isin(known.connectivity) & ~panel.scaffold.isin(known.scaffold)]
    if len(allowed) < 500:
        raise ValueError("Insufficient mechanism compounds after global FDA identity/scaffold purge")
    return allowed.reset_index(drop=True)


def workflow(fit, query, auxiliary, *, device, configurations=CONFIGS):
    cv = list(inner_groups(fit))
    y = fit.label.to_numpy()
    train = fit.smiles.to_numpy()
    check = query.smiles.to_numpy()
    candidates = {}
    statistics = []
    for name, grid in configurations.items():
        variants = []
        for setting in grid:
            scores = []
            models = []
            raw = np.empty(len(y))
            aux_predictions = [None] * len(cv)
            for i, (tr, va) in enumerate(cv):
                model = train_graph(train[tr], y[tr], auxiliary, setting,
                                    seed=20261122+i, device=device)
                scores.append(model.raw(check, device=device))
                pred = model.raw(train[va], device=device)
                raw[va] = pred[:, 0]
                aux_predictions[i] = (va, pred[:, 1:])
                models.append(model)
            forecasts = np.mean(scores, axis=0)
            values = [ranking(y[va], raw[va]) for _, va in cv]
            utility = float(np.mean([.6 * m["auroc"] + .4 * m["auprc"] for m in values]))
            item = {"method": name, "setting": list(setting), "utility": utility, **ranking(y, raw)}
            statistics.append(item)
            variants.append((utility, raw, forecasts, models, aux_predictions))
        candidates[name] = max(variants, key=lambda x: x[0])
    inner = {}
    outer = {}
    models = {}
    for name, (_, raw, forecast, members, aux_predictions) in candidates.items():
        inner[name] = raw
        outer[name] = forecast[:, 0]
        models[name] = members
        for ix, mechanism in enumerate(members[0].names):
            for va, values in aux_predictions:
                inner.setdefault(mechanism, np.empty(len(y)))[va] = values[:, ix]
            outer[mechanism] = forecast[:, ix+1]
    return inner, outer, models, statistics


def run(args):
    args.out.mkdir(exist_ok=False, parents=True)
    data = RegisteredData(args.data)
    fit, query = data.fit_table(args.kind, args.fold), data.outer_query(args.kind, args.fold, development=True)
    check_disjoint(fit, query)
    aux = safe_auxiliary(args.aux, args.all_fda_structures)
    y = data.outcomes(args.kind, args.fold, development=True).loc[query.index, "label"]
    inner, forecasts, models, statistics = workflow(fit, query, aux, device=args.device)
    pd.DataFrame(inner, index=fit.index).assign(label=fit.label, group=fit.group).to_parquet(args.out / "inner.parquet")
    pd.DataFrame(forecasts, index=query.index).assign(label=y).to_parquet(args.out / "dev_outer.parquet")
    joblib.dump(models, args.out / "models.joblib", compress=2)
    targets = [key for key in aux if "__" in key]
    result = {
        "kind": args.kind, "fold": args.fold, "device": args.device,
        "outer_support": len(query), "auxiliary_n": len(aux), "auxiliary_tasks": {
            name: {"n": int(aux[name].count()), "positive": int(aux[name].sum())} for name in targets
        },
        "auxiliary_input_sha256": digest(args.aux / "mechanistic.parquet"),
        "fda_exclusion_registry_sha256": digest(args.all_fda_structures),
        "outer_metrics": {name: ranking(y, forecasts[name]) for name in CONFIGS},
        "inner_results": statistics,
    }
    (args.out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "aux", "all-fda-structures", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--fold", choices=range(5), type=int, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    run(parser.parse_args())
