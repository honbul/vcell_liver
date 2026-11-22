"""Freeze and separately score adaptive-gain controls for the prospective HCC pilot."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_lincs_24h_transfer import context, effect, group_means
from src.r19_plan.hcc_donor_panel_size import choices
from src.r19_plan.hcc_off_position_pilot import SLOTS
from src.r19_plan.hcc_off_position_pilot_freeze import old_panels
from src.r19_plan.hcc_off_position_pilot_native_score import native_predictions
from src.r19_plan.hcc_off_position_pilot_score import frozen_arrays, measurements, pinned_npz, prepared, quantile, write_stage
from src.r19_plan.hcc_pilot_layout import LINES, ROLES
from src.r19_plan.hcc_pilot_score import DRAWS, bootstrap, vehicle_contrasts
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import checked_array
from src.r19_plan.jhh5_donor_attribution import shrink_consensus
from src.r19_plan.jhh5_off_position import CALIBRATED, COUNT, MEAN, SCALAR, SHRINK
from src.r19_plan.jhh7_off_position_native import REFERENCES
from src.r19_plan.jhh_donor_agreement_gain import ENERGY, MEAN_ENERGY, model_gain
from src.r19_plan.t4_workflow import scalar


ADAPTIVE = (MEAN_ENERGY, ENERGY)
PAIRS = (
    (ENERGY, MEAN_ENERGY), (MEAN_ENERGY, SCALAR), (MEAN_ENERGY, CALIBRATED),
    (ENERGY, CALIBRATED), (MEAN_ENERGY, "B0"), (ENERGY, "B0"),
    (MEAN_ENERGY, "best_VC"), (ENERGY, "best_VC"),
)
MODELS = (*ADAPTIVE, SCALAR, CALIBRATED, "B0", *REFERENCES)


def registered_freezes(args):
    documents = {}
    for name, root, pin in (("original", args.forecast, args.forecast_pin),
                            ("native", args.native, args.native_pin)):
        if sha256(root/"freeze.json") != sha256(pin):
            raise ValueError(f"Registered {name} prediction freeze changed")
        documents[name] = read_json(pin)
    base, native = documents["original"], documents["native"]
    if (base["target_measured"] or native["target_measured"] or tuple(base["planned_lines"]) != LINES
            or native["original_freeze_sha256"] != sha256(args.forecast_pin)
            or native["original_access_sha256"] != base["access_sha256"]
            or native["protocol_sha256"] != sha256(args.native_protocol)
            or base["slots"] != len(SLOTS) or tuple(native["slots"]) != SLOTS
            or base["genes"] != len(native["genes"]) or set(native["prepared"]) != set(ROLES)):
        raise ValueError("The original prediction and fit bundles belong to different cohorts or axes")
    stamp = dict(original_freeze_sha256=sha256(args.forecast_pin), native_freeze_sha256=sha256(args.native_pin),
        amplitude_protocol_sha256=sha256(args.amplitude_protocol), code_sha256=sha256(__file__),
        gain_code_sha256=sha256(Path(__file__).with_name("jhh_donor_agreement_gain.py")))
    return base, native, stamp


def log_rms(mean):
    return np.log(np.maximum(np.sqrt(np.square(mean).mean(axis=1)), 1e-8))[:, None]


def query_panel(source, controls, ref, genes, cols, role, slots):
    values = effect(source, controls, ref, genes, cols, role)
    plates, vectors = group_means(values, source, ["parent_slot", "parent", "cell_id", "rna_plate"])
    cells, vectors = group_means(vectors, plates, ["parent_slot", "parent", "cell_id"])
    result = []
    for (slot, parent), idx in cells.groupby(["parent_slot", "parent"], sort=True).indices.items():
        names = cells.cell_id.iloc[idx].tolist()
        if len(names) != COUNT:
            raise ValueError("Expected the original eight source cell lines for every slot")
        result.append((slot, parent, vectors[idx[choices(parent, names, 0)]]))
    if [(slot,parent) for slot,parent,_ in result] != list(slots[["parent_slot","parent"]].itertuples(index=False, name=None)):
        raise ValueError("The source panel and assigned prospective parents differ")
    return np.stack([panel for _,_,panel in result])


def predict(args, base, native, stamp):
    role = args.role
    idx = ROLES.index(role)
    genes, arrays = frozen_arrays(args, base)
    if not np.array_equal(genes, native["genes"]):
        raise ValueError("The prepared expressions differ from the prospective gene axis")
    root = args.forecast/"access"
    if sha256(root/"receipt.json") != base["access_sha256"]:
        raise ValueError("The original fit contract changed")
    access = read_json(root/"receipt.json")
    if access["target_measured"] or access["expression_read"] or not access["metadata_only"]:
        raise ValueError("Only the original prospective access plan is supported")
    fit = read_pinned(root/role, "fit.tsv", access["role_files"][role]["fit.tsv"])
    slots = read_pinned(root, "slots.tsv", access["files"]["slots.tsv"])
    prep = args.native/"prepared"/role
    if sha256(prep/"receipt.json") != native["prepared"][role]:
        raise ValueError("A prepared predictor receipt changed")
    record = read_json(prep/"receipt.json")
    if (record["query_expression_read"] or not record["anchor_verified"] or record["fit_role"] != role
            or record["original_freeze_sha256"] != stamp["original_freeze_sha256"]
            or record["original_access_sha256"] != base["access_sha256"]):
        raise ValueError("The predictor bundle has unapproved target data or role lineage")
    train = read_pinned(prep, "training.tsv", record["files"]["training.tsv"])
    query = read_pinned(prep, "query.tsv", record["files"]["query.tsv"])
    if (tuple(slots.parent_slot) != SLOTS or not query.equals(slots[["parent_slot", "parent"]])
            or tuple(train.parent) != tuple(sorted(set(fit.parent))) or len(train) != 277
            or set(fit.parent) & set(slots.parent) or set(fit.scaffold) & set(slots.scaffold_unit)
            or not fit.cell_id.eq("JHH5").all() or not fit.actual_dose_uM.eq("10").all()):
        raise ValueError("The prospective assessment crosses an original training identity, scaffold or context")
    with pinned_npz(prep/"input.npz", record["files"]["input.npz"]) as payload:
        xfit, y = (checked_array(payload[name], (len(train), len(genes)), name) for name in ("x_outer", "y_outer"))
        prepared_xnew = checked_array(payload["x_query"], (len(SLOTS), len(genes)), "x_query")
    if (sha256(args.previous/"freeze.json") != sha256(args.previous_pin)
            or sha256(args.previous_pin) != base["previous_freeze_sha256"]):
        raise ValueError("The donor arrays must come from the original JHH-5 development freeze")
    development = read_json(args.previous_pin)
    old = SimpleNamespace(access=args.previous/"access", access_pin=args.previous_access_pin, protocol=args.previous_protocol)
    ref, axis, cols, lineage = context(old)
    if not np.array_equal(axis.astype(str), genes) or any(development[k] != v for k,v in lineage.items()):
        raise ValueError("Source matrix axes differ from the original JHH-5 donor freeze")
    parents, vectors = old_panels(args, ref, axis, development, role)
    positions = parents.get_indexer(train.parent)
    if not parents.is_unique or (positions < 0).any():
        raise ValueError("Expected the original eight donor vectors per training parent")
    train_vectors = vectors[positions]
    np.testing.assert_array_equal(train_vectors.mean(axis=1), xfit)
    source, controls = (read_pinned(root/role, name, access["role_files"][role][name])
                        for name in ("source.tsv", "source_controls.tsv"))
    # Replay in the original precision before casting, including both fixed-gain references.
    target_vectors = query_panel(source, controls, ref, axis, cols, role, slots)
    xnew = target_vectors.mean(axis=1)
    np.testing.assert_array_equal(xnew, prepared_xnew)
    np.testing.assert_array_equal(xnew.astype("float32"), arrays[MEAN][idx])
    sfit = shrink_consensus(train_vectors, [COUNT]*len(train), xfit)
    snew = shrink_consensus(target_vectors, [COUNT]*len(target_vectors), xnew)
    np.testing.assert_array_equal(snew.astype("float32"), arrays[SHRINK][idx])
    output, fit_summary = {}, {}
    for arm, ref, x, z in ((MEAN_ENERGY, SCALAR, xfit, xnew), (ENERGY, CALIBRATED, sfit, snew)):
        constant = scalar(train, x, y)
        np.testing.assert_array_equal((constant*z).astype("float32"), arrays[ref][idx])
        gain, beta = model_gain(log_rms(xfit), log_rms(xnew), x, y, agreement=False)
        output[arm] = (gain[:, None]*z).astype("float32")
        fit_summary[arm] = dict(fixed_gain=constant, intercept=beta[0], norm_slope=beta[1],
                                gain_min=gain.min(), gain_median=np.median(gain), gain_max=gain.max())
    dest = args.amplitude/"predictions"/role
    dest.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".amplitude-") as scratch:
        stage = Path(scratch)
        payload = stage/"predictions.npz"
        np.savez_compressed(payload, genes=genes, slots=np.asarray(SLOTS), **output)
        write_json(stage/"receipt.json", dict(stamp, role=role, fit_parents=len(train), fits=fit_summary,
            fit_bundle_sha256=native["prepared"][role], fixed_predictions_replayed=True,
            target_expression_read=False, payload_sha256=sha256(payload)))
        stage.rename(dest)


def verified_predictions(args, native, stamp, genes):
    receipts, values = {}, {arm:[] for arm in ADAPTIVE}
    if not np.array_equal(genes, native["genes"]):
        raise ValueError("The supplemental prediction gene axis changed")
    for role in ROLES:
        root = args.amplitude/"predictions"/role
        path = root/"receipt.json"
        record = read_json(path)
        if (any(record[k] != v for k,v in stamp.items()) or record["role"] != role
                or record["fit_bundle_sha256"] != native["prepared"][role] or record["fit_parents"] != 277
                or record["target_expression_read"] or not record["fixed_predictions_replayed"]):
            raise ValueError("An amplitude prediction differs from the registered label-free contract")
        with pinned_npz(root/"predictions.npz", record["payload_sha256"]) as pred:
            if set(pred.files) != {"genes", "slots", *ADAPTIVE} or tuple(pred["slots"]) != SLOTS or not np.array_equal(pred["genes"], genes):
                raise ValueError("Supplemental forecast keys, genes or blinded slots changed")
            for arm in ADAPTIVE:
                values[arm].append(checked_array(pred[arm], (len(SLOTS), len(genes)), arm))
        receipts[role] = sha256(path)
    manifest = dict(stamp, roles=receipts, genes=list(genes), slots=list(SLOTS), arms=list(ADAPTIVE), target_measured=False)
    return manifest, {arm:np.stack(rows) for arm,rows in values.items()}


def evaluate(targets, paired, truth, predictions, draws=DRAWS):
    energy = np.square(truth).mean(axis=(0, 2))
    errors = np.stack([np.square(truth-predictions[name]).mean(axis=2) for name in MODELS])
    refs = [MODELS.index(name) for name in REFERENCES]
    rows, support = [], []
    for cell in LINES:
        line = targets.cell.eq(cell).to_numpy()
        for scope, keep in (("all_eligible", line), ("different_culture_repeat", line & paired)):
            subset = targets.loc[keep]
            if subset.empty:
                continue
            count = dict(cell=cell, scope=scope, parents=subset.canonical_parent.nunique(), wells=len(subset),
                         cultures=subset.culture_id.nunique(), batches=subset.batch_id.nunique(), roles=len(ROLES))
            support.append(count)
            for mode, crossed in (("parent", False), ("parent_and_culture", True)):
                if crossed and count["cultures"] < 2:
                    continue
                points, samples = bootstrap(subset, errors[:, :, keep], energy[keep], draws, crossed)
                for metric, point, sample, best in zip(("RMSE", "pooled_R2"), points, samples, (np.min, np.max), strict=True):
                    for arm, ref in PAIRS:
                        i = MODELS.index(arm)
                        if ref == "best_VC":
                            value, replicates = best(point[refs]), best(sample[:,refs], axis=1)
                        else:
                            j = MODELS.index(ref)
                            value, replicates = point[j], sample[:,j]
                        delta = sample[:,i]-replicates
                        lo, hi = quantile(delta)
                        rel, rel_lo, rel_hi = np.nan, np.nan, np.nan
                        if metric == "RMSE" and value > 0:
                            rel = 1-point[i]/value
                            ratio = np.divide(sample[:,i], replicates, out=np.full(draws, np.nan), where=replicates>0)
                            rel_lo, rel_hi = quantile(1-ratio)
                        rows.append(dict(count, resampling=mode, metric=metric, arm=arm, reference=ref,
                            arm_value=point[i], reference_value=value, difference=point[i]-value, low99=lo, high99=hi,
                            relative_gain=rel, relative_low99=rel_lo, relative_high99=rel_hi,
                            valid_draws=int(np.isfinite(delta).sum())))
    return {"paired":pd.DataFrame(rows), "support":pd.DataFrame(support)}


def score(args, native, stamp):
    _, accepted, controls, targets, paired, _, genes, predictions, proof = prepared(args)
    path = args.intake/"receipt.json"
    lock = read_json(path)
    if (sha256(path) != args.intake_sha256 or {k:v for k,v in lock.items() if k!="files"} != proof
            or any(sha256(args.intake/name) != digest for name,digest in lock["files"].items())):
        raise ValueError("Supplemental scoring requires the exact original locked assay")
    manifest, vectors = verified_predictions(args, native, stamp, genes)
    path = args.amplitude/"freeze.json"
    if sha256(path) != args.amplitude_sha256 or read_json(path) != manifest:
        raise ValueError("Supplemental scoring requires the exact amplitude prediction freeze")
    ix = pd.Index(SLOTS).get_indexer(targets.parent_slot)
    if (ix < 0).any():
        raise ValueError("A target slipped outside the registered design slots")
    predictions.update({arm:arr[:,ix] for arm,arr in vectors.items()})
    predictions.update(native_predictions(args, stamp["original_freeze_sha256"], genes, targets))
    truth = vehicle_contrasts(measurements(args.measurements, proof["measurements_sha256"], accepted, genes),
                              accepted, controls, targets)
    write_stage(args.output, dict(proof, expression_read=True, lab_measurements_used=not args.synthetic,
        independent_confirmation=False, intake_sha256=args.intake_sha256, amplitude_freeze_sha256=args.amplitude_sha256,
        native_freeze_sha256=stamp["native_freeze_sha256"], amplitude_protocol_sha256=stamp["amplitude_protocol_sha256"],
        amplitude_scorer_sha256=stamp["code_sha256"], bootstrap_draws=DRAWS), evaluate(targets, paired, truth, predictions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    runs = parser.add_subparsers(dest="phase", required=True)
    for phase in ("predict", "freeze", "score"):
        sub = runs.add_parser(phase)
        for name in ("amplitude", "amplitude-protocol", "forecast", "forecast-pin", "native", "native-pin", "native-protocol"):
            sub.add_argument("--"+name, type=Path, required=True)
        if phase == "predict":
            sub.add_argument("--role", choices=ROLES, required=True)
            for name in ("previous", "previous-pin", "previous-access-pin", "previous-protocol"):
                sub.add_argument("--"+name, type=Path, required=True)
        if phase == "score":
            for name in ("audit", "wells", "cultures", "qc-record", "assay-receipt", "protocol", "intake", "measurements", "output"):
                sub.add_argument("--"+name, type=Path, required=True)
            sub.add_argument("--intake-sha256", required=True)
            sub.add_argument("--amplitude-sha256", required=True)
            sub.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    base, native, stamp = registered_freezes(args)
    if args.phase == "predict":
        predict(args, base, native, stamp)
    elif args.phase == "freeze":
        genes, _ = frozen_arrays(args, base)
        manifest, _ = verified_predictions(args, native, stamp, genes)
        write_json(args.amplitude/"freeze.json", manifest)
    else:
        score(args, native, stamp)
