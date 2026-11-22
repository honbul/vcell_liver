"""Freeze the historical candidate, then issue registrable outcome-free predictions."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold

from vc_fm_bench.artifacts import digest, file_record, new_directory, write_json
from .data import CHEMBERTA, CLAMP_VECTOR_SHA, SCOPES, V2, FeatureBank, Registered, read
from .models import (FEATURES, SEED, calibration_draws, crossfit, decision_threshold,
                     dev_selection, fit_calibrator, fit_head, group_splits, raw_score)
from .molecular import CHECKPOINT, nearest


def fit_local(rows, values, model_policy, *, deploy_name=None):
    y = rows.label.to_numpy()
    folds = list(group_splits(rows, 5, SEED))
    models = {}
    for name, policy in model_policy.items():
        method, setting = policy["method"], policy["setting"]
        x = values[FEATURES[name]]
        p = crossfit(name, setting, x, y, folds)
        calibrator = fit_calibrator(method, p, y)
        calibrator.threshold = decision_threshold(y, calibrator.probability(p))
        draws = calibration_draws(method, p, rows) if name == deploy_name else None
        models[name] = dict(head=fit_head(name, setting, x, y), calibrator=calibrator, draws=draws)
    return models


def run_history(args):
    roster = Registered()
    hist = roster.historical()
    values = FeatureBank().get(hist)
    policy = dev_selection(hist, values)
    chosen = policy["deployment"]
    models = fit_local(hist, values, policy["methods"], deploy_name=chosen)
    x = values["bits"]
    group = hist.scaffold.to_numpy()
    support = [nearest(x[i:i+1], x[group != group[i]])[0] for i in range(len(x))]
    with new_directory(args.out) as stage:
        filename = stage / "checkpoint.joblib"
        contents = dict(model=chosen, **models[chosen], references=x,
                        ood_cutoff=float(np.quantile(support, .05)))
        joblib.dump(contents, filename, compress=3)
        joblib.dump(models, stage / "historical_heads.joblib", compress=3)
        write_json(stage / "freeze.json", dict(
            **policy, history_rows=len(hist), history_connectivity=int(hist.connectivity.nunique()),
            history_sha256=roster.manifest["release-0/train.parquet"],
            split_sha256=roster.split_hash, external_labels_loaded=False,
            clamp_checkpoint=CHECKPOINT, frozen_vectors_sha256=CLAMP_VECTOR_SHA,
            checkpoint=file_record(filename), historical_heads=file_record(stage / "historical_heads.joblib"),
            program={str(path): digest(path) for path in sorted(Path("confirm_dili").glob("*.py"))
                     if path.name in {"fit.py", "models.py", "molecular.py", "data.py"}}))
        print("Historical release model frozen", chosen, policy["methods"][chosen]["method"], flush=True)


def v1_baseline(fit, query):
    # The original structure + ChemBERTa implementation, refitted on this exact roster.
    from liver_dili.benchmark import Dataset, choose
    expected = "c233542dc52de4d7ce43ce7615c3b8d26517b0e78ee3ce2bf45b79f5d1802bf4"
    if digest(CHEMBERTA) != expected:
        raise ValueError("Unexpected frozen FDA ChemBERTa features")
    dataset = Dataset(CHEMBERTA, None)
    folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=20261122).split(fit, fit.label, fit.group))
    for a, b in folds:
        if set(fit.iloc[a].connectivity) & set(fit.iloc[b].connectivity):
            raise ValueError("DILI v1 comparator inner connectivity overlap")
    result, policy = choose(dataset, fit, query, "structure", "lightgbm", folds)
    return result, policy["setting"]


def old_comparators(scope, fit, query):
    old_freeze = read(V2 / "freeze.json")
    forecast = V2 / "forecasts" / scope
    model = V2 / "models" / scope
    receipt = read(forecast / "receipt.json")
    card = read(model / "release.json")
    checkpoint = model / "checkpoint.joblib"
    if (digest(forecast / "receipt.json") != old_freeze["predictions"][scope]
            or digest(forecast / "forecast.parquet") != receipt["predictions_sha256"]
            or digest(checkpoint) != card["checkpoint_sha256"]
            or receipt["checkpoint_sha256"] != card["checkpoint_sha256"]
            or receipt["policy_sha256"] != old_freeze["policy_sha256"]
            or card["fit_rows"] != len(fit)
            or card["increment_labels_seen"]):
        raise ValueError(f"Unexpected archived DILI v2 artifact: {scope}")
    table = pd.read_parquet(forecast / "forecast.parquet")
    if not table.index.equals(query.index):
        raise ValueError(f"Frozen DILI v2 prediction roster changed: {scope}")
    import torch
    torch.set_num_threads(2)
    packed = joblib.load(checkpoint)
    raw = np.mean([member.raw(query.smiles)[:, 0] for member in packed.graphs["dmpnn"]], axis=0)
    return table, expit(raw), digest(checkpoint)


def forecast(args):
    bank, reg = FeatureBank(), Registered()
    scope = args.scope
    query, fit = reg.query(scope), reg.fit(scope)
    policy = read(args.history / "freeze.json")
    if policy["split_sha256"] != reg.split_hash:
        raise ValueError("FDA split changed after historical lock")
    if any(digest(Path(f)) != checksum for f, checksum in policy["program"].items()):
        raise ValueError("Forecast code changed after historical lock")
    if scope == "release-0":
        path = args.history / "historical_heads.joblib"
        if file_record(path) != policy["historical_heads"]:
            raise ValueError("Historical heads changed after freeze")
        fitted = joblib.load(path)
    else:
        fitted = fit_local(fit, bank.get(fit), policy["methods"])
    query_values = bank.get(query)
    result = {}
    for name, obj in fitted.items():
        raw = raw_score(name, obj["head"], query_values[FEATURES[name]])
        prob = obj["calibrator"].probability(raw)
        result[name] = prob, prob >= obj["calibrator"].threshold
    baseline, v1_setting = v1_baseline(fit, query)
    v2, graph, v2_hash = old_comparators(scope, fit, query)
    result["dili_v1"], result["dili_v2"] = [(p.probability, p.decision) for p in (baseline, v2)]
    result["dmpnn"] = graph, graph >= .5
    output = pd.DataFrame({f"{name}/{field}": arr for name, pair in result.items()
                           for field, arr in zip(("p", "decision"), pair)}, index=query.index)
    if output.isna().any().any() or not output.index.is_unique:
        raise ValueError(f"A comparator does not cover its registered denominator: {scope}")
    with new_directory(args.out) as stage:
        output.to_parquet(stage / "forecast.parquet")
        write_json(stage / "receipt.json", dict(
            scope=scope, fit_rows=len(fit), query_rows=len(query), connectivity_overlap=0,
            history_freeze=digest(args.history / "freeze.json"), v1_structure_head=v1_setting,
            v2_checkpoint=v2_hash, forecast=file_record(stage / "forecast.parquet"),
            label_file_opened=False))
    print(scope, len(fit), len(query), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("history", "forecast"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--scope", choices=SCOPES)
    args = parser.parse_args()
    (run_history if args.stage == "history" else forecast)(args)
