"""Stage D1 fit/select data from the registered metadata with no score arrays."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.d1_access import CONTROL_KEYS, SLICES
from src.r19_plan.i_stage import (
    ARRAYS, PREDICT, PredictionInputs, check_prediction_scope, parse_landmarks, parse_tsv, pinned_bytes,
    read_prediction_bundle,
)
from src.r19_plan.r18_input_bundle import extract_matrix, indices
from src.r19_plan.r18_predict import vehicle_means

PREDICT_FILES = {f"{name}.tsv" for name in PREDICT}
BUNDLE_FILES = PREDICT_FILES | {"expression.npz", "landmarks.tsv"}


def access_reference(pin, slice_id):
    ref = read_json(pin)
    if (slice_id not in SLICES or set(ref["slice_files"]) != set(SLICES)
            or not ref["metadata_only"] or ref["expression_read"]
            or set(ref["slice_files"][slice_id]["predict"]) != PREDICT_FILES):
        raise ValueError("D1 requires the pinned predictor-only, balanced treatment and control plan")
    return ref


def physical_subset(current, approved):
    keys = current[CONTROL_KEYS]
    if (keys.duplicated().any() or not keys.merge(approved[CONTROL_KEYS], on=CONTROL_KEYS, validate="one_to_one").equals(keys)):
        raise ValueError("D1 control departed from the original admitted fit/select roles")


def prepare(access, reference, approved_bundle, approved_reference, slice_id):
    """Read only the pinned D1 fit and select physical controls and admitted treatment arrays."""
    ref = access_reference(reference, slice_id)
    if sha256(access / "receipt.json") != sha256(reference) or sha256(approved_reference) != ref["source_access_sha256"]:
        raise ValueError("D1 control or treatment access lineage changed")
    approved = read_prediction_bundle(approved_bundle, approved_reference, slice_id)
    files = ref["slice_files"][slice_id]["predict"]
    plan = access / "predict-plan" / slice_id
    raw = {name: pinned_bytes(plan / name, digest) for name, digest in files.items()}
    tables = {name: parse_tsv(raw[f"{name}.tsv"]) for name in PREDICT}
    check_prediction_scope(tables, slice_id)
    for name in ("inner_and_outer_labels", "query_features"):
        if not tables[name].equals(approved.tables[name]):
            raise ValueError("D1 changed a registered non-control predictor axis")
    axis = parse_tsv(pinned_bytes(access / "gene_indices.tsv", ref["files"]["gene_indices.tsv"]))
    genes = approved.landmarks.pr_gene_id.to_numpy()
    if not np.array_equal(indices(axis.gene_id), genes):
        raise ValueError("D1 changed the registered L1000 landmark ordering")
    arrays = {"inner_and_outer_labels": approved.arrays["inner_and_outer_labels"]}
    for name in ("fit_controls", "select_controls"):
        controls = tables[name]
        physical_subset(controls, approved.tables[name])
        arrays[name] = extract_matrix(ref["matrix"], controls, indices(axis.level3_gene_index), genes)
    return raw, approved.landmarks, arrays


def stage(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    raw, landmarks, arrays = prepare(args.access, args.reference, args.approved_bundle, args.approved_reference,
                                     args.slice)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".d1-inputs-") as tmp:
        output = Path(tmp)
        for name, data in raw.items():
            (output / name).write_bytes(data)
        landmarks.to_csv(output / "landmarks.tsv", sep="\t", index=False)
        np.savez_compressed(output / "expression.npz", **arrays)
        write_json(output / "receipt.json", dict(slice=args.slice, access_sha256=sha256(args.reference),
            source_bundle_sha256=sha256(args.approved_bundle / "receipt.json"),
            code_sha256=sha256(__file__), arrays={name: list(value.shape) for name, value in arrays.items()},
            files={p.name: sha256(p) for p in output.iterdir()},
        ))
        output.rename(args.out)
    return sha256(args.out / "receipt.json")


def read_bundle(root, reference, slice_id):
    """Give the fitting process only admitted labels and fit/select control arrays."""
    ref = access_reference(reference, slice_id)
    receipt = read_json(root / "receipt.json")
    files = receipt["files"]
    if (receipt["slice"] != slice_id or receipt["access_sha256"] != sha256(reference)
            or receipt["code_sha256"] != sha256(__file__) or set(files) != BUNDLE_FILES
            or {p.name for p in root.iterdir()} != BUNDLE_FILES | {"receipt.json"}
            or files["landmarks.tsv"] != ref["landmarks_sha256"]
            or any(files[name] != digest for name, digest in ref["slice_files"][slice_id]["predict"].items())):
        raise ValueError("Unexpected D1 predictor bundle or protected-data inventory")
    tables = {name: parse_tsv(pinned_bytes(root / f"{name}.tsv", files[f"{name}.tsv"])) for name in PREDICT}
    check_prediction_scope(tables, slice_id)
    genes = parse_landmarks(pinned_bytes(root / "landmarks.tsv", files["landmarks.tsv"]))
    package = root / "expression.npz"
    if sha256(package) != files[package.name]:
        raise ValueError("D1 predictor-only arrays changed after staging")
    with np.load(package, allow_pickle=False) as reader:
        if set(reader.files) != set(ARRAYS) or set(reader.files) != set(receipt["arrays"]):
            raise ValueError("D1 predictor received unregistered RNA arrays")
        raw = {name: reader[name] for name in ARRAYS}
    for name, x in raw.items():
        if (x.dtype != np.float32 or list(x.shape) != receipt["arrays"][name]
                or x.shape != (len(tables[name]), len(genes)) or not np.isfinite(x).all()):
            raise ValueError(f"D1 received a nonfinite or misaligned RNA array: {name}")
    for name in ("fit_controls", "select_controls"):
        plate = tables[name].rna_plate
        means = vehicle_means(tables[name], raw[name], "rna_plate")
        raw[name] = np.stack([means[key] for key in plate])
    inputs = {"access": sha256(reference), "bundle": sha256(root / "receipt.json")}
    return PredictionInputs(tables, raw, genes, inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", choices=SLICES, required=True)
    for name in ("access", "reference", "approved-bundle", "approved-reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(stage(parser.parse_args()))
