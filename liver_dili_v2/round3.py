"""Investigate DILI errors, reweight structure groups, and evaluate calibration."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import GroupKFold

from .calibration import calibrate, diagnostics, risk_coverage
from .data import KINDS, RegisteredData, check_disjoint, inner_groups
from .features import Features, subset, tanimoto
from .rankers import train
from .round1 import ranking
from .stack import mechanistic_features

REWEIGHT = ("extra", "linear", "lgbm", "tsvm")


def read_level(paths, filename):
    frames = [pd.read_parquet(p / filename) for p in paths]
    result = frames[0].copy()
    for a in frames[1:]:
        if set(a.index) != set(result.index):
            raise ValueError("Level 1 fold identifiers changed")
        labels = a.reindex(result.index)
        for name in set(result.columns) & set(labels):
            if not result[name].equals(labels[name]):
                raise ValueError(f"Fold-aligned {name} disagrees")
        result = result.join(a.drop(columns=list(set(a) & set(result))), validate="one_to_one")
    if not np.isfinite(result.drop(columns="group", errors="ignore").to_numpy()).all():
        raise ValueError("Missing/nonfinite OOF scores")
    return result


def error_features(blocks):
    props = blocks["base"][:, 2048:]
    fm = np.stack([np.isfinite(v).all(axis=1) for k,v in blocks.items() if k not in ("base", "counts", "v1")], axis=1)
    return {
        "high_mw": props[:, 0] >= 500,
        "high_lipophilicity": props[:, 1] >= 4.,
        "charged_parent": np.abs(props[:, 16]) > .01,
        "metal": props[:, -1] > 0,
        "two_plus_structural_alerts": props[:, -12:].sum(axis=1) >= 2,
        "missing_any_foundation": ~fm.all(axis=1),
    }


def hard_weights(masks, y, raw):
    p = expit(raw)
    confidence_error = np.abs(p-y)
    result = np.ones(len(y))
    avg = confidence_error.mean()
    for mask in masks.values():
        if mask.sum() >= 10:
            bump = max(0., confidence_error[mask].mean()-avg)
            result[mask] += min(.5, 2.*bump)
    return result/result.mean()


def add_mechanisms(inner, outer):
    endpoints = [name for name in inner if "__" in name]
    if not endpoints:
        return
    result = np.empty(len(inner))
    for t, q in GroupKFold(5).split(inner, groups=inner.group):
        result[q] = mechanistic_features(inner.iloc[t], inner.iloc[q], endpoints)
    inner["signed_mechanism"] = result
    outer["signed_mechanism"] = mechanistic_features(inner, outer, endpoints)


def run(args):
    args.out.mkdir(exist_ok=False, parents=True)
    suffix = f"{args.kind}-{args.fold}"
    paths = [p / suffix for p in (args.round1_cpu, args.round1_cat, args.round2, args.round1_kernels)]
    inner = read_level(paths, "inner.parquet")
    outer = read_level(paths, "dev_outer.parquet")
    config = json.loads((paths[0] / "receipt.json").read_text())
    data = RegisteredData(args.data)
    train_table, query = data.fit_table(args.kind, args.fold), data.outer_query(args.kind, args.fold, development=True)
    check_disjoint(train_table, query)
    inner = inner.reindex(train_table.index)
    outer = outer.reindex(query.index)
    if not np.array_equal(inner.label, train_table.label):
        raise ValueError("OOF rows differ from the training labels")
    actual = inner.label.to_numpy()
    bank = Features(args.fm, args.chemberta)
    x, q = [bank.arrays(t) for t in (train_table, query)]
    difficult, query_masks = error_features(x), error_features(q)
    weights = hard_weights(difficult, actual, inner.v1_lgbm.to_numpy())
    models = {}
    weight_sets = []
    baseline = config["selected"]["v1_lgbm"]
    for f, v in inner_groups(train_table):
        part = train_table.iloc[f]
        error_scores = np.empty(len(part))
        for p, t in GroupKFold(3).split(part, groups=part.group):
            va = train("v1_lgbm", baseline["representation"], baseline["config"],
                       subset(x, f[p]), actual[f[p]])
            error_scores[t] = va.raw(subset(x, f[t]))
        local = hard_weights({k: mask[f] for k,mask in difficult.items()}, actual[f], error_scores)
        weight_sets.append((f, v, local))
    for name in REWEIGHT:
        choice = config["selected"][name]
        method = name+"_reweighted"
        raw = np.empty(len(train_table))
        for f, v, w in weight_sets:
            estimate = train(name, choice["representation"], choice["config"], subset(x, f), actual[f],
                             sample_weight=w)
            raw[v] = estimate.raw(subset(x, v))
        estimator = train(name, choice["representation"], choice["config"], x, actual, sample_weight=weights)
        outer[method] = estimator.raw(q)
        inner[method] = raw
        models[method] = estimator
    add_mechanisms(inner, outer)
    cols = [name for name in inner if name not in ("label", "group") and "__" not in name]
    y = outer.label.to_numpy()
    results = {}
    for col in cols:
        calibrated, trials, _ = calibrate(inner[col].to_numpy(), actual, inner.group)
        prob, prior = calibrated.probabilities(outer[col].to_numpy(), estimate_prior=True)
        decision = prob >= calibrated.threshold
        results[col] = {"calibration": calibrated.method, "input_only_prior": prior, "trials": trials,
                        "outer": diagnostics(y, prob, decision),
                        "risk_coverage": risk_coverage(y, prob, decision, outer.nearest_tanimoto.to_numpy())}
    p = expit(inner.v1_lgbm.to_numpy())
    p_out = expit(outer.v1_lgbm.to_numpy())
    diagnostics_rows = []
    for label, masks, labels, pred in (("inner_oof", difficult, actual, p), ("outer", query_masks, y, p_out)):
        for group, mask in masks.items():
            diagnostics_rows.append({
                "role": label, "category": group, "n": int(mask.sum()),
                "false_positive": int((mask & (pred >= .5) & (labels==0)).sum()),
                "false_negative": int((mask & (pred < .5) & (labels==1)).sum()),
                "mean_absolute_error": float(np.mean(np.abs(pred[mask]-labels[mask]))) if mask.any() else None})
    misses = pd.DataFrame({"label": y, "probability": p_out}, index=outer.index)
    misses["error"] = np.abs(misses.label-misses.probability)
    for group, mask in query_masks.items():
        misses[group] = mask
    misses.sort_values("error", ascending=False).head(30).to_csv(args.out / "error_roster.tsv", sep="\t")
    inner.to_parquet(args.out / "inner.parquet")
    outer.to_parquet(args.out / "dev_outer.parquet")
    joblib.dump(models, args.out / "reweighted.joblib", compress=2)
    receipt = {"kind": args.kind, "fold": args.fold, "round3_results": results, "errors": diagnostics_rows}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({m: r["outer"] for m,r in results.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "fm", "chemberta", "round1-cpu", "round1-cat", "round2", "round1-kernels", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--fold", choices=range(5), type=int, required=True)
    run(parser.parse_args())
