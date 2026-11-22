"""Score the five public reference models from the locked prospective pilot intake."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256
from src.r19_plan.hcc_off_position_pilot import SLOTS
from src.r19_plan.hcc_off_position_pilot_score import measurements, pinned_npz, prepared, quantile, write_stage
from src.r19_plan.hcc_pilot_layout import LINES, ROLES
from src.r19_plan.hcc_pilot_score import DRAWS, bootstrap, vehicle_contrasts
from src.r19_plan.i_score import checked_array
from src.r19_plan.jhh5_off_position import CALIBRATED
from src.r19_plan.jhh7_off_position_native import MODELS, REFERENCES


ARMS = (CALIBRATED, *REFERENCES)


def native_predictions(args, original_sha256, genes, targets):
    path = args.native / "freeze.json"
    if sha256(path) != sha256(args.native_pin):
        raise ValueError("The supplementary public forecast freeze differs from its registered hash")
    frozen = read_json(args.native_pin)
    if (frozen["original_freeze_sha256"] != original_sha256 or frozen["target_measured"]
        or frozen["protocol_sha256"] != sha256(args.native_protocol) or tuple(frozen["slots"]) != SLOTS
        or not np.array_equal(frozen["genes"], genes) or tuple(frozen["models"]) != MODELS):
        raise ValueError("Supplementary forecasts are detached from the original cohort or axes")
    idx = pd.Index(SLOTS).get_indexer(targets.parent_slot)
    if (idx < 0).any():
        raise ValueError("An eligible target has no prospective forecast")
    predictions = {model: [] for model in MODELS}
    for role in ROLES:
        root = args.native / "prepared" / role / "receipt.json"
        if sha256(root) != frozen["prepared"][role] or not read_json(root)["anchor_verified"]:
            raise ValueError("The original donor anchor and fit preparation must be verified")
        for model in MODELS:
            path = args.native / "forecasts" / role / model / "receipt.json"
            if sha256(path) != frozen["roles"][role][model]:
                raise ValueError("A supplementary model differs from the forecast freeze")
            receipt = read_json(path)
            if (receipt["input_sha256"] != frozen["prepared"][role] or receipt["role"] != role or receipt["model"] != model
                or receipt["original_freeze_sha256"] != original_sha256):
                raise ValueError("Unexpected role, model, or fit lineage in the native comparator")
            arr = pinned_npz(path.parent / "prediction.npy", receipt["output_sha256"])
            predictions[model].append(checked_array(arr, (len(SLOTS), len(genes)), model)[idx])
    return {model: np.stack(value) for model, value in predictions.items()}


def compare_metrics(estimates, samples, context):
    rows, wins = [], []
    model_idx = np.arange(1, len(ARMS))
    for metric, center, sample, choose in zip(("RMSE", "pooled_R2"), estimates, samples, (np.argmin, np.argmax), strict=True):
        finite = np.isfinite(sample).all(axis=1)
        actual = sample[finite]
        if np.isfinite(center).all():
            best = 1+int(choose(center[1:]))
            idx = 1+choose(actual[:, 1:], axis=1)
        else:
            best, idx = None, np.array([], dtype=int)
        comparisons = [(ARMS[ix], ARMS[ix], center[ix], actual[:, ix]) for ix in model_idx]
        comparisons.append(("best_VC", ARMS[best] if best is not None else "", center[best] if best is not None else np.nan,
                            actual[np.arange(len(actual)), idx] if best is not None else np.array([])))
        for name, point_winner, val, model_draws in comparisons:
            difference = actual[:, 0] - model_draws if len(model_draws) else np.array([])
            lower, upper = quantile(difference)
            relative = rel_lo = rel_hi = np.nan
            if metric == "RMSE" and val > 0:
                relative = 1-center[0]/val
                ratio = np.divide(actual[:, 0], model_draws, out=np.full(len(actual), np.nan), where=model_draws>0)
                rel_lo, rel_hi = quantile(1-ratio)
            rows.append(dict(context, metric=metric, arm=CALIBRATED, reference=name, best_at_point=point_winner,
                arm_value=center[0], reference_value=val, difference=center[0]-val, low99=lower, high99=upper,
                relative_gain=relative, relative_low99=rel_lo, relative_high99=rel_hi, valid_draws=len(actual)))
        wins.extend(dict(context, metric=metric, model=ARMS[k], wins=int(np.count_nonzero(idx==k)), valid_draws=len(actual))
                    for k in model_idx)
    return rows, wins


def evaluate(targets, paired, truth, predictions, draws=DRAWS):
    energy = np.square(truth).mean(axis=(0, 2))
    errors = np.stack([np.square(truth-predictions[name]).mean(axis=2) for name in ARMS])
    rows, frequency, support = [], [], []
    for cell in LINES:
        line = targets.cell.eq(cell).to_numpy()
        for scope, active in (("all_eligible", line), ("different_culture_repeat", line & paired)):
            samples = targets.loc[active]
            if samples.empty:
                continue
            sizes = dict(cell=cell, scope=scope, parents=samples.canonical_parent.nunique(), wells=len(samples),
                         cultures=samples.culture_id.nunique(), batches=samples.batch_id.nunique(), roles=len(ROLES))
            support.append(sizes)
            for mode, crossed in (("parent", False), ("parent_and_culture", True)):
                if crossed and sizes["cultures"] < 2:
                    continue
                estimates, distributions = bootstrap(samples, errors[:, :, active], energy[active], draws, crossed)
                summary, wins = compare_metrics(estimates, distributions, dict(sizes, resampling=mode))
                rows.extend(summary)
                frequency.extend(wins)
    return {"paired": pd.DataFrame(rows), "winner_frequency": pd.DataFrame(frequency), "support": pd.DataFrame(support)}


def run(args):
    physical, accepted, controls, targets, paired, counts, genes, predictions, proof = prepared(args)
    intake = args.intake / "receipt.json"
    lock = read_json(intake)
    if sha256(intake) != args.intake_sha256 or {k:v for k,v in lock.items() if k!="files"} != proof:
        raise ValueError("Supplementary scoring requires the original locked assay metadata")
    if any(sha256(args.intake / name) != h for name,h in lock["files"].items()):
        raise ValueError("Original intake metadata have changed")
    predictions.update(native_predictions(args, proof["forecast_sha256"], genes, targets))
    expression = measurements(args.measurements, proof["measurements_sha256"], accepted, genes)
    truth = vehicle_contrasts(expression, accepted, controls, targets)
    result = evaluate(targets, paired, truth, predictions)
    write_stage(args.output, dict(proof, expression_read=True, actual_measurement_used=not args.synthetic,
        independent_confirmation=False, intake_sha256=args.intake_sha256, native_freeze_sha256=sha256(args.native_pin),
        supplementary_protocol_sha256=sha256(args.native_protocol), scorer_sha256=sha256(__file__),
        existing_vcs=list(REFERENCES), draws=DRAWS), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("audit", "forecast", "forecast-pin", "native", "native-pin", "native-protocol", "wells", "cultures", "qc-record",
                  "assay-receipt", "protocol", "intake", "measurements", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--intake-sha256", required=True)
    parser.add_argument("--synthetic", action="store_true")
    run(parser.parse_args())
