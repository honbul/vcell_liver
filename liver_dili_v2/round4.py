"""Develop the DILI ensemble and freeze model-choice policy on source-only OOF."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from liver_dili.splits import digest

from .calibration import calibrate, diagnostics, risk_coverage
from .data import KINDS
from .stack import construct, tune, GRID

CONTROLS = ("structure_only", "kernel_tree", "foundation_tree")


def develop(paths, target):
    fit = pd.read_parquet(paths / "inner.parquet")
    check = pd.read_parquet(paths / "dev_outer.parquet")
    truth, labels = fit.label.to_numpy(), check.label.to_numpy()
    main, raw_oof, choice, grid = tune(fit, truth, fit.group)
    models = {"v2_inner_selected": main}
    candidates = {"v2_inner_selected": raw_oof}
    source = {}
    for name in (*GRID, "weighted_heterogeneous", "mechanism_oof"):
        match = next((part for part in grid if part["recipe"] == name), None)
        if match is None:
            continue
        preds = list(GRID.get(name, GRID["heterogeneous"] if name=="weighted_heterogeneous" else GRID["kernel_tree"]))
        if name == "weighted_heterogeneous":
            preds += [k for k in fit if k.endswith("_reweighted")]
        elif name == "mechanism_oof":
            preds += [k for k in fit if "__" in k or k == "signed_mechanism"]
        selected = [col for col in preds if col in fit]
        settings = match["penalty"]
        raw = np.empty(len(fit))
        from sklearn.model_selection import GroupKFold

        for t, v in GroupKFold(5).split(fit, groups=fit.group):
            cv = construct(fit.iloc[t], selected, truth[t], name, settings)
            raw[v] = cv.raw(fit.iloc[v])
        models[name] = construct(fit, selected, truth, name, settings)
        candidates[name] = raw
    report, hold = {}, {}
    for recipe, model in models.items():
        fitter, comparisons, _ = calibrate(candidates[recipe], truth, fit.group.to_numpy())
        uncalibrated = model.raw(check)
        scores, prior = fitter.probabilities(uncalibrated, estimate_prior=True)
        keep = scores >= fitter.threshold
        hold[recipe] = scores
        source[recipe] = (model, fitter)
        report[recipe] = {"method": fitter.method, "threshold": fitter.threshold, "prevalence": prior,
                          "weights": dict(zip(model.transform.features, model.coeff[1:].tolist())),
                          "metrics": diagnostics(labels, scores, keep),
                          "risk_coverage": risk_coverage(labels, scores, keep, check.nearest_tanimoto.to_numpy()),
                          "calibrator_grid": comparisons}
    joblib.dump(source, target / "meta.joblib", compress=3)
    pd.DataFrame(hold, index=check.index).assign(label=labels,
                                                nearest_tanimoto=check.nearest_tanimoto).to_parquet(
                                                    target / "dev_outer.parquet")
    meta = {"base_artifact_sha256": digest(paths / "receipt.json"), "inner_optimum": choice,
            "development_grid": grid, "evaluations": report}
    (target/"receipt.json").write_text(json.dumps(meta, indent=2, allow_nan=False)+"\n")
    return report


def lock(runs, output):
    parts = []
    chain = {}
    for kind in KINDS:
        for fold in range(5):
            tag = f"{kind}-{fold}"
            filename = runs / tag / "receipt.json"
            record = json.loads(filename.read_text())
            chain[tag] = digest(filename)
            parts.extend({"split": kind, "fold": fold, "method": method, **part["metrics"]}
                         for method, part in record["evaluations"].items())
    table = pd.DataFrame(parts).groupby(["method", "split"], as_index=False).mean(numeric_only=True)
    table = table.loc[table.method.ne("v2_inner_selected")].copy()
    table["objective"] = .62*table.auroc + .35*table.auprc - .03*table.brier
    # Avoid selecting a method that gains almost entirely on just one split definition.
    support = table.groupby("method").objective.agg(["mean", "min"])
    support["utility"] = .7*support["mean"] + .3*support["min"]
    selected = support.utility.idxmax()
    report = {"selected": selected, "fold_sha256": chain,
              "selection_uses_increment": False, "source_dev_results": table.to_dict("records")}
    with output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("develop", "lock"))
    for key in ("round3", "round4", "out"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--fold", choices=range(5), type=int)
    args = parser.parse_args()
    if args.action=="develop":
        args.out.mkdir(exist_ok=False, parents=True)
        result = develop(args.round3 / f"{args.kind}-{args.fold}", args.out)
        print(json.dumps({n:r["metrics"] for n,r in result.items()}), flush=True)
    else:
        lock(args.round4, args.out)
