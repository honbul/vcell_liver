"""Read only the pinned role-specific I-track rows from passed development caches."""

import argparse
from dataclasses import dataclass
from hashlib import sha256 as sha256_bytes
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r15_lincs.data import table
from src.r19_plan.contracts import coord_keys, id_keys
from src.r19_plan.i_preflight import GEO, PHYSICAL_RNA, ROLES, TARGET_KEYS

PREDICT = ("inner_and_outer_labels", "query_features", "fit_controls", "select_controls")
SCORE = ("score_labels", "score_controls")
ARRAYS = ("inner_and_outer_labels", "fit_controls", "select_controls")
SLICES = {f"f{fold}-{role}" for fold in range(3) for role in ROLES}
FORBIDDEN_QUERY_COLUMNS = {"cache_index", "cache_plate_index", "cache_role_index", "inner_phase"}


@dataclass(frozen=True)
class PredictionInputs:
    tables: dict[str, pd.DataFrame]
    arrays: dict[str, np.ndarray]
    landmarks: pd.DataFrame
    input_receipts: dict[str, str]


def pinned_bytes(path, digest):
    raw = path.read_bytes()
    if sha256_bytes(raw).hexdigest() != digest:
        raise ValueError(f"Changed registered I-track metadata: {path}")
    return raw


def slice_roles(slice_id):
    match = re.fullmatch(r"f([0-2])-([ABC])", slice_id)
    if match is None:
        raise ValueError(f"Unknown I-track slice: {slice_id}")
    fold, fit = match.groups()
    at = ROLES.index(fit)
    return int(fold), fit, ROLES[(at + 1) % 3], ROLES[(at + 2) % 3]


def validate_plan(stamp):
    if (not stamp["metadata_only"] or stamp["expression_read"] or stamp["fitted"]
            or stamp["R18_protected_input_overlap"] != 0
            or stamp["slices"] != len(SLICES) or set(stamp["files"]) != SLICES):
        raise ValueError("I-track stage lacks its complete, metadata-only, R18-excluded lock")


def plan_receipt(plan, pin, slice_id):
    slice_roles(slice_id)
    digest = sha256(pin)
    stamp = json.loads(pinned_bytes(plan / "receipt.json", digest))
    validate_plan(stamp)
    return stamp, digest


def parse_tsv(raw):
    return table(BytesIO(raw))


def read_tables(plan, stamp, slice_id, section, names):
    registered = stamp["files"][slice_id][section]
    if set(registered) != {name + ".tsv" for name in names}:
        raise ValueError(f"Unexpected I-track {section} plan inventory")
    path = plan / (section + "-plan") / slice_id
    raw = {name: pinned_bytes(path / name, digest) for name, digest in registered.items()}
    return {name: parse_tsv(raw[name + ".tsv"]) for name in names}, raw


def checked_query(rows):
    if rows.empty or set(rows.columns) & FORBIDDEN_QUERY_COLUMNS:
        raise ValueError("The I-track query must contain only registered metadata")
    return rows


def read_query_metadata(plan, pin, slice_id):
    """Read the registered query metadata without opening either expression cache."""
    stamp, _ = plan_receipt(plan, pin, slice_id)
    name = "query_features.tsv"
    raw = pinned_bytes(plan / "predict-plan" / slice_id / name, stamp["files"][slice_id]["predict"][name])
    return checked_query(parse_tsv(raw))


def check_wells(rows):
    if (rows.empty or not rows.study.eq(GEO).all()
            or len(id_keys(rows, "inst_id")) != len(rows)
            or len(coord_keys(rows, *PHYSICAL_RNA)) != len(rows)):
        raise ValueError("An I-track physical RNA well is repeated or out of scope")


def check_role(controls, target, role):
    if not controls.role.eq(role).all() or set(controls.rna_plate) != set(target.rna_plate):
        raise ValueError(f"The I-track control role or plate axis changed: {role}")


def check_prediction_scope(tables, slice_id):
    fold, fit, select, _ = slice_roles(slice_id)
    labels, query = tables["inner_and_outer_labels"], checked_query(tables["query_features"])
    control, selected_control = tables["fit_controls"], tables["select_controls"]
    seen = labels.loc[labels.inner_phase.eq("inner_fit")]
    selected = labels.loc[labels.inner_phase.eq("select")]
    if (set(labels.inner_phase) != {"inner_fit", "select"}
            or set(labels.parent) & set(query.parent) or set(seen.parent) & set(selected.parent)
            or not query.outer_fold.eq(str(fold)).all() or labels.outer_fold.eq(str(fold)).any()):
        raise ValueError("I-track fit, selection and held parent axes overlap or changed")
    active = pd.concat((labels, query), ignore_index=True)
    check_wells(pd.concat((active, control, selected_control), ignore_index=True))
    check_role(control, active, fit)
    check_role(selected_control, selected, select)


def numbers(series, maximum, name):
    if not series.str.fullmatch(r"[0-9]+").all():
        raise ValueError(f"An I-track cache index is invalid: {name}")
    result = series.to_numpy(dtype=np.int64)
    if (result >= maximum).any():
        raise ValueError(f"An I-track cache index is out of range: {name}")
    return result


def parse_landmarks(raw):
    genes = parse_tsv(raw)
    if (genes.empty or not genes.pr_gene_id.str.fullmatch(r"[0-9]+").all()
            or not genes.pr_gene_id.is_unique):
        raise ValueError("The I-track landmark ID axis is invalid")
    genes["pr_gene_id"] = genes.pr_gene_id.astype(np.int64)
    return genes


class ExpressionCache:
    """Check cache headers and metadata before issuing exact NumPy index selections."""

    def __init__(self, cache, vehicles, stamp):
        self.cache, self.vehicles = cache, vehicles
        target_pin = json.loads(pinned_bytes(cache / "receipt.json", stamp["cache_metadata_sha256"]))
        control_pin = json.loads(pinned_bytes(vehicles / "receipt.json", stamp["control_cache_metadata_sha256"]))
        self.target_count = target_pin["target_wells"]
        if (target_pin["study"] != GEO or control_pin["audit_sha256"] != stamp["role_metadata_sha256"]
                or target_pin["files"]["landmarks.tsv"] != stamp["landmarks_sha256"]):
            raise ValueError("The development expression caches have different metadata lineage")
        self.landmark_bytes = pinned_bytes(cache / "landmarks.tsv", stamp["landmarks_sha256"])
        self.landmarks = parse_landmarks(self.landmark_bytes)
        target = parse_tsv(pinned_bytes(cache / "target_rows.tsv", target_pin["files"]["target_rows.tsv"]))
        index = numbers(target["index"], self.target_count, "target metadata")
        if len(target) != self.target_count or len(np.unique(index)) != self.target_count:
            raise ValueError("The registered target cache row axis is incomplete or duplicated")
        self.target_rows = target.set_index(index, verify_integrity=True)
        self.plates = parse_tsv(pinned_bytes(vehicles / "vehicles.tsv", control_pin["files"]["vehicles.tsv"]))
        if self.plates.empty or not self.plates.rna_plate.is_unique or self.plates.rna_plate.eq("").any():
            raise ValueError("The registered aggregate DMSO plate axis is invalid")

    def target_indices(self, rows):
        indices = numbers(rows.cache_index, self.target_count, "treatments")
        registered = self.target_rows.loc[indices, TARGET_KEYS].reset_index(drop=True)
        if (len(np.unique(indices)) != len(indices)
                or not registered.equals(rows[TARGET_KEYS].reset_index(drop=True))):
            raise ValueError("A locked treatment index does not match its registered metadata")
        return indices

    def control_indices(self, rows, role):
        unique = rows[["rna_plate", "cache_plate_index", "cache_role_index"]].drop_duplicates()
        plates = numbers(unique.cache_plate_index, len(self.plates), "DMSO plates")
        roles = numbers(unique.cache_role_index, len(ROLES), "DMSO roles")
        if (not unique.rna_plate.is_unique or len(np.unique(plates)) != len(plates)
                or not np.array_equal(self.plates.rna_plate.to_numpy()[plates], unique.rna_plate.to_numpy())
                or not np.all(roles == ROLES.index(role))):
            raise ValueError("A locked DMSO index does not match its registered role and plate")
        expand = pd.Index(unique.rna_plate).get_indexer(rows.rna_plate)
        return plates, roles, expand

    @staticmethod
    def take(path, shape, indices):
        source = np.load(path, mmap_mode="r", allow_pickle=False)
        if source.shape != shape or source.dtype != np.float32:
            raise ValueError(f"I-track cache header changed: {path.name}")
        result = source[indices]
        if not np.isfinite(result).all():
            raise ValueError(f"I-track permitted cache rows have nonfinite expression: {path.name}")
        return result

    def expression(self, labels, controls):
        target_index = self.target_indices(labels)
        control_index = {name: self.control_indices(rows, role) for name, (rows, role) in controls.items()}
        genes = len(self.landmarks)
        arrays = {"labels": self.take(self.cache / "target_wells.npy", (self.target_count, genes), target_index)}
        for name, (plates, roles, expand) in control_index.items():
            means = self.take(self.vehicles / "vehicles.npy", (len(self.plates), len(ROLES), genes), (plates, roles))
            arrays[name] = means[expand]
        return arrays


def input_receipts(stamp, digest):
    return {"access": digest, "target_cache": stamp["cache_metadata_sha256"],
            "vehicle_cache": stamp["control_cache_metadata_sha256"], "landmarks": stamp["landmarks_sha256"]}


def prepare_prediction(cache, vehicles, plan, pin, slice_id):
    stamp, digest = plan_receipt(plan, pin, slice_id)
    tables, files = read_tables(plan, stamp, slice_id, "predict", PREDICT)
    check_prediction_scope(tables, slice_id)
    _, fit, select, _ = slice_roles(slice_id)
    source = ExpressionCache(cache, vehicles, stamp)
    arrays = source.expression(tables["inner_and_outer_labels"],
                               {f"{name}_controls": (tables[f"{name}_controls"], role)
                                for name, role in (("fit", fit), ("select", select))})
    arrays["inner_and_outer_labels"] = arrays.pop("labels")
    bundle = PredictionInputs(tables, arrays, source.landmarks, input_receipts(stamp, digest))
    return bundle, files | {"landmarks.tsv": source.landmark_bytes}


def read_prediction_inputs(cache, vehicles, plan, pin, slice_id):
    """Load only fit and inner-select expression in the registered prediction row order."""
    bundle, _ = prepare_prediction(cache, vehicles, plan, pin, slice_id)
    return bundle


def stage_prediction(cache, vehicles, plan, pin, slice_id, out):
    """Stage a predictor-only bundle with neither score cache paths nor score values."""
    if out.exists():
        raise FileExistsError(out)
    bundle, files = prepare_prediction(cache, vehicles, plan, pin, slice_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=".i-stage-") as temporary:
        stage = Path(temporary)
        for name, raw in files.items():
            (stage / name).write_bytes(raw)
        np.savez(stage / "expression.npz", **bundle.arrays)
        write_json(stage / "receipt.json", dict(
            slice=slice_id, inputs=bundle.input_receipts, stage_code_sha256=sha256(__file__),
            files={path.name: sha256(path) for path in sorted(stage.iterdir())},
        ))
        stage.rename(out)
    return sha256(out / "receipt.json")


def read_prediction_bundle(out, pin, slice_id):
    """Recheck all identifiers and artifacts before giving the bundle to a predictor."""
    slice_roles(slice_id)
    pin_bytes = pin.read_bytes()
    reference = json.loads(pin_bytes)
    validate_plan(reference)
    path = out / "receipt.json"
    receipt_bytes = path.read_bytes()
    bundle = json.loads(receipt_bytes)
    pinned = reference["files"][slice_id]["predict"] | {"landmarks.tsv": reference["landmarks_sha256"]}
    inventory = set(pinned) | {"expression.npz"}
    if (set(pinned) != {f"{name}.tsv" for name in PREDICT} | {"landmarks.tsv"}
            or bundle["slice"] != slice_id
            or bundle["inputs"] != input_receipts(reference, sha256_bytes(pin_bytes).hexdigest())
            or bundle["stage_code_sha256"] != sha256(__file__)
            or set(bundle["files"]) != inventory or {item.name for item in out.iterdir()} != inventory | {"receipt.json"}
            or any(bundle["files"][file] != digest for file, digest in pinned.items())):
        raise ValueError("The I-track predictor bundle differs from its pinned plan")
    tables = {name: parse_tsv(pinned_bytes(out / f"{name}.tsv", pinned[f"{name}.tsv"])) for name in PREDICT}
    check_prediction_scope(tables, slice_id)
    landmarks = parse_landmarks(pinned_bytes(out / "landmarks.tsv", pinned["landmarks.tsv"]))
    with np.load(BytesIO(pinned_bytes(out / "expression.npz", bundle["files"]["expression.npz"])),
                 allow_pickle=False) as package:
        if set(package.files) != set(ARRAYS) or len(package.files) != len(ARRAYS):
            raise ValueError("The I-track predictor bundle includes an unexpected expression section")
        arrays = {name: package[name] for name in ARRAYS}
    for name, value in arrays.items():
        if value.shape != (len(tables[name]), len(landmarks)) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"The I-track predictor bundle has changed expression axes: {name}")
    return PredictionInputs(tables, arrays, landmarks,
                            bundle["inputs"] | {"stage": sha256_bytes(receipt_bytes).hexdigest()})


def load_score_delta(cache, vehicles, plan, pin, slice_id, expected_query, expected_landmarks):
    """Read the held score delta after the caller verifies the completed prediction freeze."""
    stamp, _ = plan_receipt(plan, pin, slice_id)
    tables, _ = read_tables(plan, stamp, slice_id, "score", SCORE)
    fold, _, _, score = slice_roles(slice_id)
    labels, controls = tables["score_labels"], tables["score_controls"]
    query = checked_query(labels.drop(columns="cache_index"))
    if not query.equals(expected_query) or not query.outer_fold.eq(str(fold)).all():
        raise ValueError("The I-track score wells differ from the registered query axis")
    check_wells(pd.concat((query, controls), ignore_index=True))
    check_role(controls, query, score)
    source = ExpressionCache(cache, vehicles, stamp)
    if not source.landmarks.equals(expected_landmarks):
        raise ValueError("The I-track score landmark order differs from the frozen gene axis")
    values = source.expression(labels, {"controls": (controls, score)})
    first = np.flatnonzero(~controls.rna_plate.duplicated().to_numpy())
    rows = pd.Index(controls.rna_plate.iloc[first]).get_indexer(query.rna_plate)
    control = values["controls"][first[rows]]
    return values["labels"] - control


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "vehicles", "plan", "pin", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--slice", dest="slice_id", required=True)
    print(stage_prediction(**vars(parser.parse_args())))
