"""Materialize a registered R18 predictor slice without opening the score plan."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r19_plan.r18_access_plan import SOURCE_KEY, TARGET_FIELDS, WELL, keys
from src.r19_plan.r18_matrix_index import GENE_IDS, SAMPLE_IDS, STUDIES

EXPRESSION = "0/DATA/0/matrix"
ROWS = (*WELL, "matrix", "level3_row_index")
EXPRESSION_TABLES = (
    "inner_and_outer_labels", "fit_controls", "select_controls", "B_donor_inputs", "B_donor_controls",
)
PLAN_TABLES = (*EXPRESSION_TABLES, "query_features")
SLICE = re.compile(r"(C1|C2|C3_(?:6|24)h_(?:HEPG2_only|HUH7_added))/([AB])/f[0-2]-([ABC])")


def checked_bytes(path, expected):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Changed registered R18 input: {path}")
    return payload


def check_predictor_scope(tables, case, scenario, fit):
    query = tables["query_features"]
    labels = tables["inner_and_outer_labels"]
    fit_controls = tables["fit_controls"]
    select_controls = tables["select_controls"]
    donors = tables["B_donor_inputs"]
    donor_controls = tables["B_donor_controls"]
    if list(query) != TARGET_FIELDS:
        raise ValueError("R18 query must have metadata only in the registered order")
    if not len(query) or set(labels.inner_phase) != {"inner_fit", "select"}:
        raise ValueError("R18 predictor is missing its held-out query or an inner partition")
    inner = labels.loc[labels.inner_phase.eq("inner_fit")]
    held = labels.loc[labels.inner_phase.eq("select")]
    if set(query.parent) & set(labels.parent) or set(inner.parent) & set(held.parent):
        raise ValueError("R18 target parent overlaps an inner or outer held partition")
    all_rows = pd.concat(tables.values(), ignore_index=True)
    if all_rows.duplicated(WELL).any() or all_rows[WELL].eq("").any().any():
        raise ValueError("R18 predictor repeats a well or uses a missing sample ID")
    # Cross-check the physical RNA coordinates, even where alternate IDs were available.
    known_coordinates = all_rows.dropna(subset=["rna_plate", "rna_well"])
    known_coordinates = known_coordinates.loc[
        known_coordinates.rna_plate.ne("") & known_coordinates.rna_well.ne("")
    ]
    if known_coordinates.duplicated(["study", "rna_plate", "rna_well"]).any():
        raise ValueError("R18 predictor repeats a physical RNA well")

    context = pd.concat([labels, query], ignore_index=True)
    plate_key = ["study", "rna_plate"]
    select = "ABC"[("ABC".index(fit) + 1) % 3]
    for controls, target, role in ((fit_controls, context, fit), (select_controls, held, select)):
        if (not controls.role.eq(role).all()
            or set(keys(controls, plate_key)) != set(keys(target, plate_key))):
            raise ValueError(f"R18 predictor has an unexpected DMSO role or plate for role {role}")

    if scenario == "A":
        if len(donors) or len(donor_controls):
            raise ValueError("R18 scenario A must not receive B source inputs")
        return
    cohort = case.split("_")[0]
    if (not donors.cohort.eq(cohort).all() or
        not donors.CONFIRM_R18_B_source_donor_allowed.eq("True").all() or
        set(keys(donors, SOURCE_KEY)) != set(keys(context, SOURCE_KEY))):
        raise ValueError("R18 scenario B donor inputs differ from the registered scope")
    source_plates = ["study", "plate"]
    if set(donor_controls.half) != {"1", "2"} or any(
        set(keys(donor_controls.loc[donor_controls.half.eq(half)], source_plates)) != set(keys(donors, source_plates))
        for half in ("1", "2")
    ):
        raise ValueError("R18 scenario B must have both source control halves for every donor plate")


def read_predictor_plan(access, access_reference, slice_id):
    match = SLICE.fullmatch(slice_id)
    if match is None:
        raise ValueError(f"Unknown R18 prediction slice: {slice_id}")
    receipt = read_json(access_reference)
    checked_bytes(access / "receipt.json", sha256(access_reference))
    if not receipt["metadata_only"] or receipt["expression_read"]:
        raise ValueError("R18 extraction plan lacks its metadata-only provenance")
    registered = receipt["files"][slice_id]["predict"]
    if set(registered) != {f"{name}.tsv" for name in PLAN_TABLES}:
        raise ValueError(f"R18 predictor table registry changed: {slice_id}")
    plan = access / "predict-plan" / slice_id
    files = {name: checked_bytes(plan / name, digest) for name, digest in registered.items()}
    tables = {name: table(BytesIO(files[f"{name}.tsv"])) for name in PLAN_TABLES}
    check_predictor_scope(tables, *match.groups())
    return receipt, files, tables


def indices(values):
    if not values.astype(str).str.fullmatch(r"[0-9]+").all():
        raise ValueError("Invalid R18 matrix row or gene index")
    return values.to_numpy(dtype=np.int64)


def extract_matrix(record, wanted, gene_indices, genes):
    """Read explicit row indices, avoiding intervening held-out expression slabs."""
    path = Path(record["path"])
    shape = (record["indexed_wells"], record["indexed_genes"])
    selected = indices(wanted.level3_row_index)
    order = np.argsort(selected)
    selected = selected[order]
    if (
        path.stat().st_size != record["bytes"]
        or np.any(selected >= shape[0]) or np.any(np.diff(selected) <= 0)
        or np.any(gene_indices >= shape[1]) or len(set(gene_indices)) != len(genes)
    ):
        raise ValueError(f"Registered R18 matrix extent or index changed: {path}")
    out = np.empty((len(wanted), len(genes)), dtype=np.float32)
    with h5py.File(path, "r") as archive:
        samples = archive[SAMPLE_IDS]
        source_genes = archive[GENE_IDS][:].astype(np.int64)
        gene_hash = hashlib.sha256("\n".join(map(str, source_genes)).encode()).hexdigest()
        if (samples.shape != (shape[0],) or source_genes.shape != (shape[1],)
            or gene_hash != record["gene_axis_sha256"]
            or not np.array_equal(source_genes[gene_indices], genes)):
            raise ValueError(f"Registered R18 sample or gene axis changed: {path}")
        ids = wanted.inst_id.to_numpy()[order]
        starts = range(0, len(selected), 128)
        for start in starts:
            chunk = selected[start:start + 128]
            if not np.array_equal(samples[chunk].astype(str), ids[start:start + len(chunk)]):
                raise ValueError(f"Registered R18 row ID changed: {path}")
        matrix = archive[EXPRESSION]
        if matrix.shape != shape or matrix.dtype.kind != "f":
            raise ValueError(f"Unexpected Level 3 expression extent or dtype: {path}")
        for start in starts:
            chunk = selected[start:start + 128]
            out[order[start:start + len(chunk)]] = matrix[chunk, :][:, gene_indices]
    if not np.isfinite(out).all():
        raise ValueError(f"Nonfinite registered R18 expression: {path}")
    return out


def materialize_expression(tables, index, index_reference, access_receipt, landmarks):
    """Return named arrays in supplied table order and the study-index receipt hashes."""
    gene_data = landmarks.read_bytes()
    gene_hash = hashlib.sha256(gene_data).hexdigest()
    genes = table(BytesIO(gene_data)).pr_gene_id.to_numpy(dtype=np.int64)
    if not len(genes) or len(set(genes)) != len(genes):
        raise ValueError("R18 landmark axis is empty or duplicated")

    arrays = {name: np.empty((len(rows), len(genes)), dtype=np.float32) for name, rows in tables.items()}
    rows = pd.concat([rows[list(ROWS)].assign(section=name, section_row=np.arange(len(rows)))
                      for name, rows in tables.items()], ignore_index=True)
    if (not len(rows) or rows[list(ROWS)].eq("").any().any() or rows.duplicated(WELL).any()
        or not set(rows.study) <= set(STUDIES)):
        raise ValueError("Expression rows must be unique and tied to a registered R18 study")

    index_sha256 = {}
    for study, study_rows in rows.groupby("study", sort=False):
        folder = index / study
        index_sha256[study] = sha256(index_reference / study / "receipt.json")
        stamp = json.loads(checked_bytes(folder / "receipt.json", index_sha256[study]))
        if (
            stamp["study"] != study or not stamp["metadata_only"] or stamp["expression_read"]
            or stamp["lock_sha256"] != access_receipt["lock_sha256"]
            or stamp["landmark_axis_sha256"] != gene_hash or stamp["landmarks"] != len(genes)
        ):
            raise ValueError(f"R18 Level 3 index lineage or landmark order changed for {study}")
        axis_name = "level3_landmarks.tsv"
        axis = table(BytesIO(checked_bytes(folder / axis_name, stamp["files"][axis_name])))
        for kind, wanted in study_rows.groupby("matrix", sort=False):
            record = stamp["matrix_records"][kind]
            ordered = axis.loc[axis.matrix.eq(kind)]
            if not np.array_equal(ordered.gene_id.to_numpy(dtype=np.int64), genes):
                raise ValueError(f"R18 indexed landmark order changed for {study}: {kind}")
            values = extract_matrix(record, wanted, indices(ordered.level3_gene_index), genes)
            for section, positions in wanted.groupby("section", sort=False).indices.items():
                arrays[section][wanted.iloc[positions].section_row] = values[positions]
    return arrays, index_sha256


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    receipt, files, tables = read_predictor_plan(args.access, args.access_reference, args.slice)
    arrays, index_hashes = materialize_expression(
        {name: tables[name] for name in EXPRESSION_TABLES},
        args.index, args.index_reference, receipt, args.landmarks,
    )
    args.out.mkdir(parents=True)
    for name, payload in files.items():
        (args.out / name).write_bytes(payload)
    (args.out / "landmarks.tsv").write_bytes(args.landmarks.read_bytes())
    np.savez(args.out / "expression.npz", **arrays)
    write_json(args.out / "receipt.json", dict(
        slice=args.slice, access_sha256=sha256(args.access_reference),
        materializer_sha256=sha256(__file__), index_sha256=index_hashes,
        predict_plan_sha256=receipt["files"][args.slice]["predict"],
        arrays={name: list(values.shape) for name, values in arrays.items()},
        files={path.name: sha256(path) for path in args.out.iterdir()},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("access", "access-reference", "index", "index-reference", "landmarks", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--slice", required=True)
    run(parser.parse_args())
