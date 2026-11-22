"""Freeze registered R18 forecasts and apply the unchanged joint confirmation rule."""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r15_lincs.forecast import PROPOSED
from src.r15_lincs.score import METRICS
from src.r16_lincs.predict import CELL_GAIN, HEADS
from src.r16_lincs.score import LEGACY_SCALED, REFERENCES, summarize

PRIMARY = "blend_FiLM"
MEASURED_ARMS = (PROPOSED, *HEADS, "B0", "B_pos", *LEGACY_SCALED)
ANCHORS = ("anchor", "blend_anchor")
PREDICTION_FILES = {"rows.tsv", "landmarks.tsv", "forecast.npz", "selection.tsv"}
PREDICT_PLAN_FILES = {"inner_and_outer_labels.tsv", "query_features.tsv", "fit_controls.tsv",
                      "select_controls.tsv", "B_donor_inputs.tsv", "B_donor_controls.tsv"}
SCOREABLE_CASES = (
    "C1", "C2", "C3_6h_HEPG2_only", "C3_6h_HUH7_added",
    "C3_24h_HEPG2_only", "C3_24h_HUH7_added",
)
SLICE = re.compile(r"([^/]+)/([AB])/f([0-2])-([ABC])")


def reference_arms(scenario):
    refs = dict(REFERENCES, B0=("B0",), B_pos=("B_pos",), R15=(PROPOSED,))
    if scenario == "B":
        refs[CELL_GAIN] = (CELL_GAIN,)
    elif scenario != "A":
        raise ValueError(f"Unexpected R18 scenario: {scenario}")
    return refs


def frozen_arms(scenario):
    arms = {*MEASURED_ARMS, *ANCHORS}
    if scenario == "B":
        arms.add(CELL_GAIN)
    return arms


def role_names(fit):
    roles = "ABC"
    i = roles.index(fit)
    return fit, roles[(i + 1) % 3], roles[(i + 2) % 3]


def check_slice_id(value):
    match = SLICE.fullmatch(value)
    if match is None or match[1] not in SCOREABLE_CASES:
        raise ValueError(f"Invalid registered slice: {value}")
    return match.groups()


def all_slices():
    return {f"{case}/{scenario}/f{fold}-{role}" for case in SCOREABLE_CASES for scenario in "AB"
            for fold in range(3) for role in "ABC"}


def check_artifacts(path, names, hashes):
    if set(hashes) != names:
        raise ValueError(f"Unexpected prediction bundle file manifest: {path}")
    for name, digest in hashes.items():
        if sha256(path / name) != digest:
            raise ValueError(f"File changed relative to its receipt: {path / name}")


def inspect_forecast(path, slice_id, predict_plan_files, plan_root, protocol_hash, landmark_hash, gene_count):
    """Verify a forecast without opening any treatment truth or score-only control."""
    _, scenario, _, fit = check_slice_id(slice_id)
    stamp_path = path / "receipt.json"
    stamp = read_json(stamp_path)
    if (stamp["slice"] != slice_id or stamp["protocol_sha256"] != protocol_hash
            or stamp["predict_plan_sha256"] != predict_plan_files):
        raise ValueError(f"Forecast was not bound to the registered prediction plan: {slice_id}")
    if {item.name for item in path.iterdir()} != PREDICTION_FILES | {"receipt.json"}:
        raise ValueError(f"Unexpected file in predictor bundle: {slice_id}")
    check_artifacts(path, PREDICTION_FILES, stamp["files"])
    check_artifacts(plan_root / slice_id, PREDICT_PLAN_FILES, predict_plan_files)
    row_hash = predict_plan_files["query_features.tsv"]
    if stamp["files"]["rows.tsv"] != row_hash:
        raise ValueError(f"Forecast physical well axis changed: {slice_id}")
    if stamp["files"]["landmarks.tsv"] != landmark_hash:
        raise ValueError(f"Forecast gene axis changed: {slice_id}")

    rows = table(path / "rows.tsv")
    if not len(rows) or rows.duplicated(["study", "inst_id"]).any():
        raise ValueError(f"Missing or duplicated forecast physical wells: {slice_id}")
    selection = table(path / "selection.tsv")
    expected = role_names(fit)
    if not len(selection) or not all(selection[role].eq(value).all()
                                     for role, value in zip(("fit", "select", "score"), expected, strict=True)):
        raise ValueError(f"Forecast selection receipt has changed control roles: {slice_id}")
    expected_shape = (len(rows), gene_count)
    with np.load(path / "forecast.npz", allow_pickle=False) as arrays:
        if len(arrays.files) != len(set(arrays.files)) or set(arrays.files) != frozen_arms(scenario):
            raise ValueError(f"Forecast arm registry changed: {slice_id}")
        for name in arrays.files:
            arm = arrays[name]
            if arm.shape != expected_shape or arm.dtype.kind != "f" or not np.isfinite(arm).all():
                raise ValueError(f"Invalid forecast {slice_id}: {name}")
            if name == "B0" and arm.any():
                raise ValueError(f"B0 is not the registered zero-change baseline: {slice_id}")
    return sha256(stamp_path)


def forecast_manifest(forecasts, access, access_ref, protocol, d0_ref):
    if sha256(access / "receipt.json") != sha256(access_ref):
        raise ValueError("The extraction allow-lists are not the registered R18 input version")
    locked = read_json(access_ref)["files"]
    if set(locked) != all_slices():
        raise ValueError("The registered R18 slice list is incomplete or different")
    d0 = read_json(d0_ref)
    landmark_hash, = set(d0["gene_axis_sha256"].values())
    protocol_hash = sha256(protocol)
    prediction_receipts = {
        name: inspect_forecast(forecasts / name, name, receipt["predict"], access / "predict-plan",
                               protocol_hash, landmark_hash, d0["landmarks"])
        for name, receipt in sorted(locked.items())
    }
    return dict(protocol_sha256=protocol_hash, access_sha256=sha256(access_ref),
                freeze_code_sha256=sha256(__file__), score_code_sha256=sha256(Path(__file__).with_name("r18_score.py")),
                d0_sha256=sha256(d0_ref),
                landmark_sha256=landmark_hash, prediction_receipts=prediction_receipts)


def check_freeze(path, current, expected_hash):
    """Compare a new receipt audit to a separately handed-off freeze identifier."""
    if sha256(path) != expected_hash or read_json(path) != current:
        raise ValueError("The R18 prediction freeze changed")


def summarize_case(records, case, scenario):
    # R18 uses the same 10,000 parent draws, best-arm-per-draw rule, and 99% intervals.
    cohort = case.split(":", 1)[0].split("_", 1)[0]
    minimum = {"C1": 1, "C2": 100, "C3": np.inf}[cohort]
    expected = frozen_arms(scenario) - set(ANCHORS)
    if not expected == set(records):
        raise ValueError("R18 statistics require identical registered arms")
    points, pairs = summarize(records, case, scenario, tuple(HEADS) + (PROPOSED,),
                              reference_arms(scenario), minimum)
    return points, [row for row in pairs if row["arm"] != row["reference"]]


def confirmation_label(rows):
    """Apply the C1 conjunction and its A/B hierarchy to the primary candidate."""
    selected = [row for row in rows if row["scope"] == "C1" and row["arm"] == PRIMARY]
    expected = {(scenario, reference, metric) for scenario in "AB"
                for reference in reference_arms(scenario) for metric in METRICS}
    supplied = {(row["scenario"], row["reference"], row["metric"]) for row in selected}
    if expected != supplied or len(selected) != len(expected):
        raise ValueError("Incomplete or duplicated C1 joint comparisons")
    good = dict.fromkeys("AB", True)
    for row in selected:
        low, high = row["low99"], row["high99"]
        valid = np.isfinite((low, high)).all() and low <= high
        passes = valid and (high < 0 if row["metric"] == "rmse" else low > 0)
        good[row["scenario"]] &= bool(row["eligible"] and passes)
    if all(good.values()):
        return "확인됨"
    if good["B"]:
        return "전이 조건에서만 확인됨"
    return "확인되지 않음"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("freeze", "check"))
    for name in ("forecasts", "access", "access-ref", "protocol", "d0-ref", "freeze-file"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--expected-sha256")
    args = cli.parse_args()
    if args.action == "check" and not args.expected_sha256:
        cli.error("check requires --expected-sha256 from the completed freeze job")
    current = forecast_manifest(args.forecasts, args.access, args.access_ref, args.protocol, args.d0_ref)
    if args.action == "freeze":
        write_json(args.freeze_file, current)
    else:
        check_freeze(args.freeze_file, current, args.expected_sha256)
    print(sha256(args.freeze_file))


if __name__ == "__main__":
    main()
