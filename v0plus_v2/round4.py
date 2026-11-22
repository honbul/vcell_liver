"""Compare sparse, global and drug-specific residual stacking in grouped source development."""

import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vc_build.external import KEY, SOURCE, native_sensitivity
from vc_build.io import publish_json, sha256
from vc_build.v0plus_cv import attach_native, panel_targets, summarize, validation_roster

from .source import grouped, load
from .stack import CAPS, RECIPES, build, pivot

BASELINE = np.array([.366, .404, .893, .627, .734])
AXES = ("spearman", "pearson", "macro_ndcg_at_5", "macro_top5_overlap", "macro_spearman")
BLEND_METRIC = np.array([.27, .21, .24, .10, .18])


def ensure_wide(base, neural):
    left = pivot(base)
    right = pivot(neural)
    if len(right.index.difference(left.index)) > 300:
        raise ValueError("Linear and neural OOF groups differ")
    return left.join(right, how="outer").sort_index()


def internal_measurement(fit, splits):
    output = []
    for idx, (train, valid) in enumerate(grouped(fit, splits)):
        _, anchor, surface, transforms = panel_targets(train)
        shares = surface.groupby([*SOURCE, "compound_id"]).source_mean.first().rename("source_mean")
        u = valid.merge(pd.DataFrame(transforms), on=SOURCE, validate="many_to_one")
        y = native_sensitivity(u) * u.slope + u.intercept
        u = u.assign(aligned=y).join(shares, on=[*SOURCE, "compound_id"], validate="many_to_one")
        u["residual"] = u.aligned-u.source_mean
        u["anchor"] = u.compound_id.map(anchor)
        output.append(u.groupby(["cell_id", "compound_id", "lineage_group"], as_index=False)[["residual", "anchor"]]
                      .mean().assign(inner=idx))
    return pd.concat(output, ignore_index=True)


def angle(records):
    score = np.array([records[k] for k in AXES])
    delta = score - BASELINE
    return float(np.dot(delta, BLEND_METRIC) + .36*min(0., delta.min()))


def fit_recipes(fit, arms, evaluations):
    registries = internal_measurement(fit, len(evaluations)).set_index(["inner", "cell_id", "compound_id"])
    aligned = registries.join(arms, how="left")
    lead = aligned.loc[:, list(arms.columns)]
    truth = aligned.reset_index()[["inner", "cell_id", "compound_id", "lineage_group", "residual", "anchor"]]
    anchor = truth.anchor.to_numpy()
    methods, ranking, models = [], [], {}
    for name in RECIPES:
        estimator = build(lead, truth, name)
        preds = estimator.predict(lead, anchor, truth.compound_id)
        for a in CAPS:
            values = anchor + a*(preds-anchor)
            tag = f"{name}/{a}"
            methods.append(truth[KEY[1:]+["inner"]].assign(method=tag, predicted_value=values))
        models[name] = estimator
    pred = pd.concat(methods, ignore_index=True)
    pred["row_id"] = pred.cell_id + "|" + pred.compound_id
    parts = []
    for idx, valid in enumerate(evaluations):
        held = pred.loc[pred.inner.eq(idx)]
        parts.append(attach_native(held, valid, idx))
    description = summarize(pd.concat(parts, ignore_index=True))
    scores = description.groupby("method")[list(AXES)].mean()
    for method, vals in scores.iterrows():
        ranking.append({"candidate": method, "utility": angle(vals), **vals.to_dict()})
    ordered = sorted(ranking, key=lambda s: -s["utility"])
    found = {}
    for record in ordered:
        key, amp = record["candidate"].split("/")
        if key not in found:
            found[key] = record
            models[key].amplitude = float(amp)
    return models, list(found.values()), ordered


def as_native_outer(outer, neural, q):
    ref = pd.concat([outer, neural], ignore_index=True)
    unique = ref[["cell_id", "compound_id", "method", "predicted_value"]].drop_duplicates()
    return q.merge(unique.pivot(index=["cell_id", "compound_id"], columns="method", values="predicted_value",
                                ).reset_index(), on=["cell_id", "compound_id"], how="left", validate="one_to_one")


def run(args):
    args.out.mkdir(exist_ok=False, parents=True)
    development, _, _ = load(args.bundle)
    train, held = list(grouped(development, 3))[args.fold]
    names = [args.round2 / f"fold{args.fold}", args.round3 / f"fold{args.fold}"]
    inner_r2, inner_r3 = [pd.read_parquet(p / "inner.parquet") for p in names]
    arms = ensure_wide(inner_r2, inner_r3)
    inner_observations = [v for _,v in grouped(train, 2)]
    estimators, order, grid = fit_recipes(train, arms, inner_observations)
    q = validation_roster(held)
    xx, yy = [pd.read_parquet(p / "dev_outer.parquet") for p in names]
    predictions = as_native_outer(xx, yy, q)
    _, anchor, _, _ = panel_targets(train)
    means = q.compound_id.map(anchor).to_numpy()
    combined = []
    for name, method in estimators.items():
        values = method.predict(predictions, means, q.compound_id)
        combined.append(q[KEY].assign(method=name, predicted_value=values))
    best = order[0]["candidate"].split("/")[0]
    combined.append(combined[list(estimators).index(best)].assign(method="fold_selected"))
    result = attach_native(pd.concat(combined, ignore_index=True), held, args.fold)
    result.to_parquet(args.out / "dev_outer.parquet", index=False)
    joblib.dump(estimators, args.out / "meta.joblib", compress=2)
    frame = summarize(result)
    publish_json(args.out / "receipt.json", {"fold": args.fold, "grid": grid, "inner_top_by_recipe": order,
                                            "outer": frame.to_dict("records"),
                                            "meta_predictors": {k: v.columns for k,v in estimators.items()},
                                            "inputs": {str(f): sha256(f/"receipt.json") for f in names}})
    print(frame.groupby("method")[list(AXES)].mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "round2", "round3", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--fold", choices=range(3), type=int, required=True)
    run(parser.parse_args())
