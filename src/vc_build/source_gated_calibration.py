"""Select reliable neural corrections in held screens and transport their score scales."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import across_cell_metrics
from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .io import freeze_table, publish_json, sha256, verify_forecast
from .neural_offset_factorial import ARCHIVE_RECEIPT, BASE, FULL, PREDICTION_KEYS
from .response_component_benchmark import BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC
from .response_components import component_scores, interaction_intervals
from .rna_state_projection import score_external
from .structured_cv import ASSAYS, observations
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

PRIMARY = "source_gated_affine"
TRANSDUCTIVE = "source_gated_quantile"
UNIFORM = "uniform_affine"
SHUFFLED = "shuffled_affine"
ARMS = (PRIMARY, TRANSDUCTIVE, "gated_raw", UNIFORM, "uniform_quantile", SHUFFLED)
REFERENCE_ARMS = (FULL, "native_neural_blend")
GRID = (0., .25, .5, 1.)
METRICS = ("spearman", "pearson")


def old_forecast(args, register, mode, fold):
    name = mode if mode in EXTERNAL_TRACKS else f"{mode}-{fold}"
    rows, stamp = verify_forecast(args.archive / f"{name}.parquet")
    prior = register["forecasts"][name]
    info = stamp["provenance"]
    if (stamp["sha256"], info["mode"], info["fold"], info["core_sha256"], info["inputs_sha256"]) != (
        prior["sha256"], mode, fold, RECEIPTS["core"], prior["inputs_sha256"],
    ):
        raise ValueError(f"Unexpected frozen study support for {name}")
    return rows, stamp


def base_and_neural(rows):
    view = rows[rows.method.isin((BASE, FULL))]
    wide = view.pivot(index=PREDICTION_KEYS, columns="method", values="predicted_value").reset_index()
    if len(wide) * 2 != len(view) or not np.isfinite(wide[[BASE, FULL]]).all(axis=None):
        raise ValueError("Missing matched structural or neural forecasts")
    return wide.assign(delta=wide[FULL] - wide[BASE])


def select_by_drug(development, drugs):
    scores = []
    for (track, fold, compound), part in development.groupby(["track", "fold", "compound_id"], sort=True):
        lineages = sorted(part.lineage_group.unique())
        if len(lineages) < 5:
            continue
        for weight in GRID:
            rated = part.assign(predicted_value=part[BASE] + weight * part.delta)
            metrics = across_cell_metrics(rated)
            if all(metrics[k] is not None for k in METRICS):
                scores.append({"track": track, "fold": fold, "compound_id": compound, "weight": weight,
                               "lineages": lineages, **{k: metrics[k] for k in METRICS}})
    report = {}
    data = pd.DataFrame(scores)
    for compound, table in data.groupby("compound_id"):
        choices = {}
        support = {}
        for track, part in table.groupby("track"):
            base = part[part.weight.eq(0)]
            groups = set().union(*base.lineages)
            if len(base) < 2 or len(groups) < 20 or not part.groupby("weight").fold.nunique().eq(len(base)).all():
                continue
            paired = part.pivot(index="fold", columns="weight", values=list(METRICS))
            if len(paired.columns) != len(GRID) * len(METRICS) or paired.isna().any(axis=None):
                continue
            choices[track] = {str(w): [float((paired[m, w] - paired[m, 0.]).mean()) for m in METRICS] for w in GRID}
            support[track] = {"lineages": len(groups), "folds": len(base)}
        valid = [(w, float(delta.mean())) for w in GRID[1:]
                 if len(choices) >= 2 and ((delta := np.asarray([x[str(w)] for x in choices.values()])) > 0).all()]
        weight = max(valid, key=lambda item: (item[1], -item[0]), default=(0., 0.))[0]
        report[compound] = {"weight": weight, "screen_support": support, "gains": choices}
    return {name: report.get(name, {"weight": 0., "screen_support": {}, "gains": {}}) for name in drugs}


def source_moments(pooled, drugs):
    # Pooled forecasts repeat for identities measured in multiple assays.
    cells = pooled.groupby(["compound_id", "lineage_group"])[[BASE, "delta"]].mean()
    results = {}
    for name in drugs:
        values = cells.loc[name]
        base = values[BASE].to_numpy()
        results[name] = {}
        for w in GRID:
            mixed = base + w * values.delta.to_numpy()
            spread, original = mixed.std(), base.std()
            tilt = float(np.clip(original / spread, .5, 2)) if min(spread, original) > 1e-8 else 1.
            results[name][str(w)] = [float(mixed.mean()), float(base.mean()), tilt]
    return results


def fit(args):
    require_hash(args.archive_receipt, ARCHIVE_RECEIPT)
    register = json.loads(args.archive_receipt.read_text())
    old, clean, _, _ = registered_inputs(args)
    development, pooled, traces = [], [], {}
    for assay in ASSAYS:
        for fold in range(3):
            rows, stamp = old_forecast(args, register, assay, fold)
            traces[f"{assay}-{fold}"] = stamp["sha256"]
            matrix = base_and_neural(rows)
            if assay == "pooled":
                values = matrix.groupby(["fold", "row_id"])[[BASE, "delta"]].nunique()
                if not values.eq(1).all(axis=None):
                    raise ValueError("Pooled response forecast repeated by assay with different values")
                pooled.append(matrix.drop_duplicates(["fold", "row_id"]))
            else:
                _, _, check = same_split(old, clean, fold, assay)
                observed = observations(check)
                joined = matrix.merge(observed, on=[x for x in PREDICTION_KEYS if x != "fold"],
                                      how="left", validate="one_to_one")
                if joined.endpoint_value.isna().any() or len(joined) != len(observed):
                    raise ValueError("Development outcomes and manifest do not match")
                development.append(joined)
    drugs = sorted(clean.compound_id.unique())
    selection = select_by_drug(pd.concat(development), drugs)
    moments = source_moments(pd.concat(pooled), drugs)
    weights = [selection[name]["weight"] for name in drugs]
    shuffle = np.random.default_rng(97211).permutation(weights)
    params = {"selection": selection, "moments": moments, "shuffled": dict(zip(drugs, shuffle.tolist()))}
    publish_json(args.out, params)
    publish_json(args.stamp, {"sha256": sha256(args.out), "forecast_sha256": traces,
                             "archive_receipt_sha256": ARCHIVE_RECEIPT, "core_sha256": RECEIPTS["core"]})


def place_at_ranks(reference, values):
    order = np.argsort(values, kind="stable")
    arranged = np.sort(reference)
    starts = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    means = np.add.reduceat(arranged, starts) / counts
    mapped = np.empty_like(reference)
    mapped[order] = np.repeat(means, counts)
    return mapped


def calibrated(rows, params):
    chunks = []
    for name, block in base_and_neural(rows).groupby("compound_id", sort=True):
        weight = params["selection"][name]["weight"]
        base, delta = block[BASE].to_numpy(), block.delta.to_numpy()
        by_arm = {}
        variants = ((PRIMARY, weight), (UNIFORM, .5), (SHUFFLED, params["shuffled"][name]))
        for arm, current in variants:
            origin, target, scale = params["moments"][name][str(current)]
            mixed = base + current * delta
            by_arm[arm] = base if current == 0 else target + scale * (mixed - origin)
        by_arm["gated_raw"] = base + weight * delta
        by_arm[TRANSDUCTIVE] = base if weight == 0 else place_at_ranks(base, by_arm["gated_raw"])
        by_arm["uniform_quantile"] = place_at_ranks(base, base + .5 * delta)
        for method, scores in by_arm.items():
            chunks.append(block[PREDICTION_KEYS].assign(method=method, predicted_value=scores))
    return pd.concat(chunks, ignore_index=True)


def verified_fit(args):
    stamp = json.loads(args.stamp.read_text())
    require_hash(args.calibrator, stamp["sha256"])
    if stamp["archive_receipt_sha256"] != ARCHIVE_RECEIPT or stamp["core_sha256"] != RECEIPTS["core"]:
        raise ValueError("Calibrator was not fit on the registered source forecasts")
    return json.loads(args.calibrator.read_text()), stamp


def predict(args):
    require_hash(args.archive_receipt, ARCHIVE_RECEIPT)
    register = json.loads(args.archive_receipt.read_text())
    params, receipt = verified_fit(args)
    rows, old = old_forecast(args, register, args.study, 0)
    derived = calibrated(rows, params)
    expected = {*BASELINES, *PUBLIC, *REFERENCE_ARMS, *ARMS}
    joint = pd.concat([rows, derived], ignore_index=True)
    wide = joint.pivot(index=PREDICTION_KEYS, columns="method", values="predicted_value")
    if set(wide) != expected or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("Calibration cannot change evaluation support")
    freeze_table(joint[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.study, "fold": 0, "inputs_sha256": old["provenance"]["inputs_sha256"],
        "core_sha256": RECEIPTS["core"], "predecessor_sha256": old["sha256"], "calibrator_sha256": receipt["sha256"],
    })


def assess(frame, *, external):
    paired = [(PRIMARY, arm) for arm in (BASE, UNIFORM, SHUFFLED)] + [(TRANSDUCTIVE, BASE)]
    metrics, _, _ = axis_metrics(frame, paired)
    across = {a+"/"+b: crossed_patient_drug_ci(frame, a, b) for a, b in paired}
    inter = interaction_intervals(frame, PRIMARY, BASE)
    ranks = {x["metric"]: x["ci95"] for x in metrics["paired_contrasts"]
             if (x["candidate"], x["baseline"]) == (PRIMARY, BASE)}
    gate = (all(across[PRIMARY+"/"+BASE][m][0] > 0 for m in METRICS) and inter[0] > 0
            and all(ranks[m][0] > -.005 for m in ("spearman", "ndcg_at_5", "top5_overlap")))
    return {"metrics": metrics, "across_cell_ci95": across, "gates": {BASE: gate}, "interaction_ci95": inter,
            "components": component_scores(frame, cell_axis="lineage_group"),
            "groups": frame.lineage_group.nunique(), "observations": frame.row_id.nunique()}


def score(args):
    _, stamp = verified_fit(args)
    for study in EXTERNAL_TRACKS:
        _, forecast = verify_forecast(args.forecasts / f"{study}.parquet")
        if forecast["provenance"]["calibrator_sha256"] != stamp["sha256"]:
            raise ValueError("Forecasts use a different source-only calibrator")
    score_external(args, assessor=assess, arms=(*REFERENCE_ARMS, *ARMS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    options = {
        "fit": ("original", "core", "archive", "archive-receipt", "stamp"),
        "predict": ("archive", "archive-receipt", "calibrator", "stamp"),
        "score": ("original", "core", "licob", "forecasts", "limore-labels", "licob-labels", "calibrator", "stamp"),
    }
    for name, fields in options.items():
        stage = stages.add_parser(name)
        for field in (*fields, "out"):
            stage.add_argument(f"--{field}", type=Path, required=True)
        if name == "predict":
            stage.add_argument("--study", choices=EXTERNAL_TRACKS, required=True)
    args = parser.parse_args()
    {"fit": fit, "predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
