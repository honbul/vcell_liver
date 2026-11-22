"""Per-drug boosts and OOF-calibrated residual amplitudes."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vc_build.external import KEY
from vc_build.io import publish_json, sha256
from vc_build.v0plus_cv import attach_native, summarize, validation_roster

from .source import grouped, load, load_drug_models
from .trees import apply_gain, development_targets, fit_models, gains, tune

CAPS = (.35, .6, .85, 1., 1.25)


def run(args):
    args.out.mkdir(exist_ok=False, parents=True)
    rows, rna, _ = load(args.bundle)
    train, val = list(grouped(rows, 3))[args.fold]
    targets = json.loads(args.targets.read_text())
    expression = rna.loc[rna.index.intersection(train.cell_id.unique())]
    r1 = args.round1 / f"fold{args.fold}"
    adapt = load_drug_models(r1 / "models.joblib")
    r1_inner = pd.read_parquet(r1 / "inner.parquet")
    selections, tree_oof = tune(train, expression, targets)
    tree = fit_models(train, expression, targets, selections)
    fitted = pd.concat([r1_inner, tree_oof], ignore_index=True)
    inner_targets = development_targets(train)
    gamma = gains(fitted, inner_targets)
    labels = []
    for i, (_, held) in enumerate(grouped(train, 2)):
        labels.append(held.assign(inner=i))
    settings = []
    for cap in CAPS:
        q = fitted.merge(inner_targets[["cell_id", "compound_id", "inner", "anchor"]],
                         on=["cell_id", "compound_id", "inner"], validate="many_to_one")
        q["gamma"] = [gamma[m].get(d, .6) for m, d in zip(q.method, q.compound_id)]
        q["forecast"] = q.anchor + cap * q.gamma * (q.forecast - q.anchor)
        parts = []
        for i, (_, held) in enumerate(grouped(train, 2)):
            predictions = q.loc[q.inner.eq(i)].merge(
                validation_roster(held)[KEY], on=["cell_id", "compound_id"], validate="many_to_one")
            parts.append(attach_native(predictions.rename(columns={"forecast": "predicted_value"}),
                                       held, i))
        scores = summarize(pd.concat(parts, ignore_index=True))
        scores = scores.assign(value=.30*scores.spearman + .20*scores.pearson +
                               .20*scores.macro_ndcg_at_5 + .1*scores.macro_top5_overlap +
                               .20*scores.macro_spearman)
        settings.append((cap, scores.groupby("method").value.mean().to_dict()))
    caps = {method: max(CAPS, key=lambda c: next(s for p,s in settings if p==c).get(method, -1.))
            for method in gamma}
    q = validation_roster(val)
    check = rna.loc[rna.index.intersection(val.cell_id.unique())]
    raw = adapt.predict(q, check) | tree.predict(q, check)
    output = []
    for method, value in raw.items():
        output.append(q[KEY].assign(method=method, predicted_value=value))
        corrected = apply_gain(value, q, adapt.anchor, gamma[method], caps[method])
        output.append(q[KEY].assign(method=method + "_shrunk", predicted_value=corrected))
    scored = attach_native(pd.concat(output, ignore_index=True), val, args.fold)
    scored.to_parquet(args.out / "dev_outer.parquet", index=False)
    fitted.to_parquet(args.out / "inner.parquet", index=False)
    joblib.dump(tree, args.out / "trees.joblib", compress=3)
    result = summarize(scored)
    publish_json(args.out / "receipt.json",
                 {"fold": args.fold, "round1_sha256": sha256(r1 / "receipt.json"),
                  "selected_trees": selections, "fit_only_gain": gamma, "fit_only_cap": caps,
                  "cap_grid": [{"cap": cap, "utility": s} for cap,s in settings],
                  "outer": result.to_dict("records")})
    print(result.groupby("method").mean(numeric_only=True).to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "targets", "out", "round1"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(3))
    run(parser.parse_args())
