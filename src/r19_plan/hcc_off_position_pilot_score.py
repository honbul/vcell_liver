"""Intake and score a realized 10 µM pilot against the existing immutable forecasts."""

import argparse
import hashlib
import re
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_off_position_pilot import SLOTS, verify_layout
from src.r19_plan.hcc_pilot_layout import LINES, ROLES
from src.r19_plan.hcc_pilot_score import (
    COORD, DRAWS, REPEAT, availability, bind_realized, bootstrap, repeat_reference, table, vehicle_contrasts,
)
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import checked_array
from src.r19_plan.jhh5_off_position import ARMS, PAIRS


SCALE = "CMap2020_Level3_log2"
MODELS = ("B0", *ARMS)
MIN_CONTROLS = 3


def pinned_npz(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Array file differs from the approved hash: {path.name}")
    return np.load(BytesIO(data), allow_pickle=False)


def registered_layout(audit, expected):
    receipt = audit / "receipt.json"
    if sha256(receipt) != expected:
        raise ValueError("Pilot coordinates and identities differ from the forecast freeze")
    stamp = read_json(receipt)
    if not stamp["metadata_only"] or stamp["measured"] or stamp["selected"] != 96:
        raise ValueError("Only the complete and previously unmeasured layout is admissible")
    slots = read_pinned(audit, "slots.tsv", stamp["files"]["slots.tsv"])
    coordinates = read_pinned(audit, "forbidden_positions.tsv", stamp["files"]["forbidden_positions.tsv"])
    path = audit / "plate_layout.tsv"
    if sha256(path) != stamp["files"][path.name]:
        raise ValueError("The registered plate map changed")
    layout = pd.read_csv(path, sep="\t").fillna({"parent_slot": "", "vehicle_role": ""})
    positions = {slot: set(coordinates.loc[coordinates.parent_slot.eq(slot), "forbidden_well"]) for slot in slots.parent_slot}
    if tuple(slots.parent_slot) != SLOTS or not slots.parent.is_unique or not slots.scaffold_unit.is_unique:
        raise ValueError("The planned molecular roster differs from the fixed slot axis")
    verify_layout(layout, positions)
    return layout, slots


def assay_intake(args, layout, slots):
    receipt = read_json(args.assay_receipt)
    mode = "synthetic" if args.synthetic else "measured"
    if (receipt.get("mode") != mode or receipt.get("target_scale") != SCALE
        or not receipt.get("study_id", "").strip() or not receipt.get("platform", "").strip()
        or not receipt.get("verified_by", "").strip()
        or not re.fullmatch(r"[0-9a-f]{64}", receipt.get("measurements_sha256", ""))):
        raise ValueError("Assay mode, identity, scale, verification, or expression digest is incomplete")
    for field, path in (("wells", args.wells), ("cultures", args.cultures), ("qc_record", args.qc_record)):
        if receipt.get(field + "_sha256") != sha256(path):
            raise ValueError(f"The realized {field} differs from its assay receipt")
    wells = table(args.wells)
    physical = bind_realized(layout, wells, dose=10)
    real_fields = COORD + ["actual_cell", "material_lot_id"]
    physical = physical.merge(wells[real_fields], on=COORD, validate="one_to_one")
    treated = physical.kind.eq("treated")
    registered = slots.set_index("parent_slot").parent
    if (not physical.actual_cell.eq(physical.cell).all() or physical.loc[treated, "material_lot_id"].str.strip().eq("").any()
            or not physical.loc[treated, "parent_slot"].map(registered).eq(physical.loc[treated, "canonical_parent"]).all()):
        raise ValueError("Realized cell, chemical identity or lot differs from the registered assay")

    cultures = table(args.cultures)
    keys = ["cell", "plate_id", "culture_id"]
    actual = physical[keys].drop_duplicates()
    if (cultures.duplicated(keys).any() or cultures.culture_id.duplicated().any()
            or cultures.independent_start_id.duplicated().any()
            or cultures[[*keys, "batch_id", "independent_start_id", "independence_record"]].apply(lambda c: c.str.strip().eq("")).any().any()
            or set(cultures[keys].itertuples(index=False, name=None)) != set(actual.itertuples(index=False, name=None))):
        raise ValueError("An actual culture start is duplicated, missing, or mismatched")
    return physical.merge(cultures, on=keys, validate="many_to_one"), receipt


def frozen_arrays(args, frozen):
    by_arm = {arm: [] for arm in MODELS}
    genes = None
    for role in ROLES:
        folder = args.forecast / "forecasts" / role
        receipt_path = folder / "receipt.json"
        if sha256(receipt_path) != frozen["roles"][role]:
            raise ValueError("A forecast role receipt differs from the original freeze")
        role_receipt = read_json(receipt_path)
        if role_receipt["role"] != role or role_receipt["prospective_labels_read"] or role_receipt["access_sha256"] != frozen["access_sha256"]:
            raise ValueError("A forecast role contains unexpected target access or lineage")
        with pinned_npz(folder / "predictions.npz", role_receipt["array_sha256"]) as a:
            axis = a["genes"].astype(str)
            if (set(a.files) != {"genes", "slots", *MODELS} or tuple(a["slots"]) != SLOTS
                or axis.shape != (978,) or len(set(axis)) != 978 or "" in axis):
                raise ValueError("The frozen 96 × 978 forecast axis differs")
            if genes is None:
                genes = axis
            elif not np.array_equal(genes, axis):
                raise ValueError("The gene axis differs between roles")
            for arm in MODELS:
                by_arm[arm].append(checked_array(a[arm], (96, 978), arm))
    arrays = {arm: np.stack(values) for arm, values in by_arm.items()}
    if arrays["B0"].any():
        raise ValueError("The original B0 reference must be zero")
    return genes, arrays


def prepared(args):
    path = args.forecast / "freeze.json"
    if sha256(path) != sha256(args.forecast_pin):
        raise ValueError("Pilot forecast freeze differs from its committed reference")
    frozen = read_json(args.forecast_pin)
    if frozen["target_measured"] or frozen["slots"] != 96 or frozen["genes"] != 978 or tuple(frozen["planned_lines"]) != LINES:
        raise ValueError("Expected the original outcome-free pilot forecast")
    layout, slots = registered_layout(args.audit, frozen["pilot_sha256"])
    physical, assay = assay_intake(args, layout, slots)
    accepted, controls, targets, paired, control_counts = availability(physical, MIN_CONTROLS)
    genes, predictions = frozen_arrays(args, frozen)
    index = pd.Index(SLOTS).get_indexer(targets.parent_slot)
    if (index < 0).any():
        raise ValueError("A scored specimen lacks a prospective forecast slot")
    mapped = {arm: values[:, index] for arm, values in predictions.items()}
    proof = dict(forecast_sha256=sha256(args.forecast_pin), layout_sha256=frozen["pilot_sha256"],
        assay_sha256=sha256(args.assay_receipt), protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
        wells_sha256=sha256(args.wells), cultures_sha256=sha256(args.cultures), qc_record_sha256=sha256(args.qc_record),
        measurements_sha256=assay["measurements_sha256"], mode=assay["mode"], scored_wells=len(targets),
        eligible_wells=len(accepted), min_vehicles_each_role=MIN_CONTROLS, arms=list(MODELS), expression_read=False)
    return physical, accepted, controls, targets, paired, control_counts, genes, mapped, proof


def measurements(path, digest, accepted, genes):
    with pinned_npz(path, digest) as data:
        if set(data.files) != {"values", "specimen_id", "genes"}:
            raise ValueError("Unexpected measurement array payload")
        ids, axis, values = pd.Index(data["specimen_id"].astype(str)), pd.Index(data["genes"].astype(str)), data["values"]
    if (not ids.is_unique or not axis.is_unique or set(ids) != set(accepted.specimen_id) or set(axis) != set(genes)
            or values.shape != (len(ids), len(axis)) or values.dtype.kind != "f" or not np.isfinite(values).all()):
        raise ValueError("Measured IDs, genes, or matrix values do not fit the locked assay axis")
    return values[np.ix_(ids.get_indexer(accepted.specimen_id), axis.get_indexer(genes))]


def quantile(values):
    vals = values[np.isfinite(values)]
    return np.quantile(vals, (.005, .995)) if len(vals) else (np.nan, np.nan)


def evaluate(targets, y, paired, predictions, draws=DRAWS):
    models = (*MODELS, REPEAT)
    vectors = {**predictions, REPEAT: repeat_reference(targets, y)}
    err = np.stack([np.square(y-vectors[name]).mean(axis=2) for name in models])
    points, pairs, support = [], [], []
    for cell in LINES:
        line = targets.cell.eq(cell).to_numpy()
        for label, use in (("all_eligible", line), ("different_culture_repeat", line & paired)):
            rows = targets.loc[use]
            if rows.empty:
                continue
            info = dict(cell=cell, scope=label, parents=rows.canonical_parent.nunique(), wells=len(rows),
                        cultures=rows.culture_id.nunique(), batches=rows.batch_id.nunique(), roles=len(ROLES))
            support.append(info)
            names = MODELS if label == "all_eligible" else models
            comparisons = PAIRS if label == "all_eligible" else (*PAIRS, (REPEAT, "B0"), ("calibrated_shrink", REPEAT))
            for resample, cross in (("parent", False), ("parent_and_culture", True)):
                if cross and info["cultures"] < 2:
                    continue
                estimates, draws_by_metric = bootstrap(rows, err[:len(names), :, use], err[0][:, use].mean(axis=0), draws, cross)
                for name, estimate, sample in zip(("RMSE", "pooled_R2"), estimates, draws_by_metric, strict=True):
                    for i, model in enumerate(names):
                        lo, hi = quantile(sample[:, i])
                        points.append(dict(info, resampling=resample, metric=name, arm=model, estimate=estimate[i], low99=lo, high99=hi))
                    for model, ref in comparisons:
                        i, j = names.index(model), names.index(ref)
                        lo, hi = quantile(sample[:, i]-sample[:, j])
                        relative = np.nan
                        relative_lo = relative_hi = np.nan
                        if name == "RMSE" and estimate[j] > 0:
                            relative = 1-estimate[i]/estimate[j]
                            fractions = np.divide(sample[:, i], sample[:, j], out=np.full(draws, np.nan), where=sample[:, j]>0)
                            relative_lo, relative_hi = quantile(1-fractions)
                        pairs.append(dict(info, resampling=resample, metric=name, arm=model, reference=ref,
                            arm_value=estimate[i], reference_value=estimate[j], difference=estimate[i]-estimate[j], low99=lo, high99=hi,
                            relative_gain=relative, relative_low99=relative_lo, relative_high99=relative_hi))
    return {"points": pd.DataFrame(points), "paired": pd.DataFrame(pairs), "support": pd.DataFrame(support)}


def plate_support(physical, count, targets):
    out = physical.groupby(["cell", "plate_id"], as_index=False).agg(registered=("specimen_id", "size"),
                qc_pass=("qc_status", lambda x: x.eq("PASS").sum())).set_index("plate_id").join(count)
    out["scored_treatments"] = targets.groupby("plate_id").size().reindex(out.index, fill_value=0)
    return out.reset_index()


def write_stage(path, proof, tables):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".pilot-score-") as scratch:
        stage = Path(scratch)
        for name, rows in tables.items():
            rows.to_csv(stage / (name + ".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(proof, files={f.name: sha256(f) for f in stage.iterdir()}))
        stage.rename(path)


def lock(args):
    physical, accepted, controls, targets, paired, counts, genes, predictions, proof = prepared(args)
    write_stage(args.output, proof, {"plate_support": plate_support(physical, counts, targets),
        "eligible": targets[["specimen_id", "cell", "plate_id", "culture_id", "well", "parent_slot"]]})


def score(args):
    physical, accepted, controls, targets, paired, counts, genes, predictions, proof = prepared(args)
    path = args.intake / "receipt.json"
    frozen = read_json(path)
    if sha256(path) != args.intake_sha256 or {k: v for k,v in frozen.items() if k != "files"} != proof:
        raise ValueError("The scoring job requires the exact locked assay and prediction metadata")
    if any(sha256(args.intake / name) != h for name,h in frozen["files"].items()):
        raise ValueError("An intake metadata table differs from its lock")
    y = vehicle_contrasts(measurements(args.measurements, proof["measurements_sha256"], accepted, genes), accepted, controls, targets)
    result = evaluate(targets, y, paired, predictions)
    result["plate_support"] = plate_support(physical, counts, targets)
    write_stage(args.output, dict(proof, expression_read=True, lab_measurements_used=not args.synthetic,
                independent_confirmation=False, bootstrap_draws=DRAWS, intake_sha256=args.intake_sha256), result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("lock", "score"))
    for field in ("audit", "forecast", "forecast-pin", "wells", "cultures", "qc-record", "assay-receipt", "protocol", "output"):
        cli.add_argument("--" + field, type=Path, required=True)
    for field in ("measurements", "intake"):
        cli.add_argument("--" + field, type=Path)
    cli.add_argument("--intake-sha256")
    cli.add_argument("--synthetic", action="store_true")
    args = cli.parse_args()
    if args.phase == "score" and not all((args.measurements, args.intake, args.intake_sha256)):
        cli.error("score requires the registered measurements and intake receipt with its SHA-256")
    (lock if args.phase == "lock" else score)(args)
