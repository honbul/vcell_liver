"""Separate admitted X fit/select expression and untreated basal features from score wells."""

import argparse
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.design import fold_id
from src.r19_plan.i_stage import parse_landmarks, parse_tsv, pinned_bytes, slice_roles
from src.r19_plan.r18_input_bundle import extract_matrix, indices
from src.r19_plan.x_access import PHYSICAL, PREDICT, SCORE

ARRAYS = ("labels", "fit_controls", "select_controls")


def slice_axes(key, cells):
    target, tag = key.rsplit("-f", 1)
    if target not in cells:
        raise ValueError("Unknown registered X context")
    fold, fit, select, score = slice_roles("f" + tag)
    validate = cells[(cells.index(target) + 1) % len(cells)]
    return target, validate, fold, fit, select, score


def reference(path):
    receipt = read_json(path)
    if (not receipt["metadata_only"] or receipt["expression_read"] or len(receipt["cells"]) != 5
            or set(receipt["slices"]) != {f"{c}-f{f}-{r}" for c in receipt["cells"] for f in range(3) for r in "ABC"}):
        raise ValueError("X input manifest is missing one or more registered slices")
    return receipt


def read_plan(root, pin, key, phase):
    if phase not in {"predict", "score"}:
        raise ValueError("Unknown X access role")
    ref = reference(pin)
    if sha256(root / "receipt.json") != sha256(pin):
        raise ValueError("X physical plan differs from its registered receipt")
    names = {f"{name}.tsv" for name in (PREDICT if phase == "predict" else SCORE)}
    inventory = ref["slices"][key][phase]
    if set(inventory) != names:
        raise ValueError("Unexpected X data inventory for its task role")
    directory = root / (phase + "-plan") / key
    raw = {name: pinned_bytes(directory / name, digest) for name, digest in inventory.items()}
    return ref, raw, {name[:-4]: parse_tsv(data) for name, data in raw.items()}


def control_axis(control, rows, role):
    keys = ["study", "rna_plate", "cell_id"]
    wanted = set(rows[keys].itertuples(index=False, name=None))
    available = set(control[keys].itertuples(index=False, name=None))
    if not wanted or not control.role.eq(role).all() or wanted != available:
        raise ValueError("The X control panel has mismatched physical plates or role")


def validate_predict(tables, key, cells):
    target, validate, fold, fit, select, _ = slice_axes(key, cells)
    labels, query = tables["labels"], tables["query"]
    held_folds = query.parent.map(lambda p: fold_id(p, "R15-outer", 3))
    label_folds = labels.parent.map(lambda p: fold_id(p, "R15-outer", 3))
    selection = labels.parent.map(lambda p: fold_id(p, f"R15-inner-{fold}", 5) == 0)
    inner = labels.inner_phase.eq("inner_fit")
    validation = labels.inner_phase.eq("select")
    if ("level3_row_index" in query or not query.cell_id.eq(target).all() or not held_folds.eq(fold).all()
            or set(labels.cell_id) != set(cells) - {target} or label_folds.eq(fold).any()
            or not labels.inner_phase.isin(["outer_only", "inner_fit", "select"]).all()
            or not inner.eq(~selection & labels.cell_id.ne(validate)).all()
            or not validation.eq(selection & labels.cell_id.eq(validate)).all()
            or set(labels.parent) & set(query.parent)):
        raise ValueError("X labels violate its held-context or compound partitions")
    active = pd.concat((labels, query), ignore_index=True)
    check_unique(pd.concat([active, tables["fit_controls"], tables["select_controls"]], ignore_index=True))
    control_axis(tables["fit_controls"], active, fit)
    control_axis(tables["select_controls"], labels.loc[validation], select)


def check_unique(rows):
    if rows.empty or rows.duplicated(["study", *PHYSICAL]).any() or rows.inst_id.duplicated().any():
        raise ValueError("An X physical sample was assigned to multiple roles")


def matrix_axes(root, ref):
    genes = parse_landmarks(pinned_bytes(root / "landmarks.tsv", ref["files"]["landmarks.tsv"]))
    axis = parse_tsv(pinned_bytes(root / "gene_indices.tsv", ref["files"]["gene_indices.tsv"]))
    if not np.array_equal(indices(axis.gene_id), genes.pr_gene_id.to_numpy()):
        raise ValueError("The X Level 3 index differs from its gene order")
    return genes, indices(axis.level3_gene_index)


def basal_features(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    ref = read_json(args.basal_reference)
    raw = ref["upstream_sha256"]
    if not ref["metadata_only"] or ref["expression_values_loaded"] or sha256(args.matrix) != raw[str(args.matrix)]:
        raise ValueError("The X2 independent DepMap source changed")
    gene_path = str(args.landmarks)
    genes = parse_landmarks(pinned_bytes(args.landmarks, raw[gene_path]))
    profile = parse_tsv(pinned_bytes(args.profiles, ref["files"]["context_profiles.tsv"]))
    cells = ref["selected_contexts"]
    profile = profile.loc[profile.eligible.eq("True")].set_index("cell", verify_integrity=True).loc[cells]
    wanted = {}
    one_gene = set(genes.pr_gene_id)
    for field in pd.read_csv(args.matrix, nrows=0).columns:
        if (match := re.search(r"\(([0-9]+)\)$", field)) and (gene := int(match[1])) in one_gene:
            if gene in wanted:
                raise ValueError("An X2 expression gene ID is ambiguous")
            wanted[gene] = field
    columns = [wanted[j] for j in genes.pr_gene_id]
    fields = ["SequencingID", "ModelID", "ModelConditionID", "IsDefaultEntryForModel"]
    hits = []
    for chunk in pd.read_csv(args.matrix, usecols=[*fields, *columns], chunksize=64, low_memory=False):
        hit = chunk.loc[chunk.SequencingID.isin(profile.sequence_id)]
        if len(hit):
            hits.append(hit)
    values = pd.concat(hits, ignore_index=True).set_index("SequencingID", verify_integrity=True).loc[profile.sequence_id]
    if (not np.array_equal(values.ModelID, profile.model_id)
            or not np.array_equal(values.ModelConditionID, profile.condition_id)
            or not values.IsDefaultEntryForModel.astype(str).str.lower().isin(["true", "1", "yes"]).all()):
        raise ValueError("X2 basal expression rows differ from their model and culture registry")
    matrix = values[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Missing expression in the matched X2 basal features")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".x-basal-") as tmp:
        stage = Path(tmp)
        np.savez_compressed(stage / "basal.npz", cells=np.asarray(cells), genes=genes.pr_gene_id.to_numpy(), values=matrix)
        write_json(stage / "receipt.json", dict(reference_sha256=sha256(args.basal_reference),
                   files={"basal.npz": sha256(stage / "basal.npz")}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


def stage_predict(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    ref, raw, tables = read_plan(args.access, args.pin, args.slice, "predict")
    validate_predict(tables, args.slice, ref["cells"])
    genes, columns = matrix_axes(args.access, ref)
    by_study = pd.concat([tables[name] for name in ARRAYS], ignore_index=True)
    outputs = extract_matrix(ref["matrix"], by_study, columns, genes.pr_gene_id.to_numpy())
    arrays = dict(zip(ARRAYS, np.split(outputs, np.cumsum([len(tables[n]) for n in ARRAYS])[:-1]), strict=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".x-input-") as tmp:
        stage = Path(tmp)
        for name, data in raw.items():
            (stage / name).write_bytes(data)
        genes.to_csv(stage / "landmarks.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "expression.npz", **arrays)
        write_json(stage / "receipt.json", dict(slice=args.slice, access_sha256=sha256(args.pin),
                   stage_sha256=sha256(__file__), files={p.name: sha256(p) for p in sorted(stage.iterdir())}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


def read_predict(bundle, pin, key):
    ref = reference(pin)
    stamp = read_json(bundle / "receipt.json")
    registered = ref["slices"][key]["predict"] | {"landmarks.tsv": ref["files"]["landmarks.tsv"]}
    if (stamp["slice"] != key or stamp["access_sha256"] != sha256(pin) or stamp["stage_sha256"] != sha256(__file__)
            or set(stamp["files"]) != set(registered) | {"expression.npz"}
            or any(stamp["files"][name] != digest for name, digest in registered.items())):
        raise ValueError("X predictor bundle differs from its physical access authority")
    tables = {name: parse_tsv(pinned_bytes(bundle / f"{name}.tsv", registered[f"{name}.tsv"])) for name in PREDICT}
    validate_predict(tables, key, ref["cells"])
    genes = parse_landmarks(pinned_bytes(bundle / "landmarks.tsv", registered["landmarks.tsv"]))
    path = bundle / "expression.npz"
    if sha256(path) != stamp["files"][path.name]:
        raise ValueError("The X predictor expression package changed after staging")
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != set(ARRAYS):
            raise ValueError("X predictor can only load treatment fitting and fit/select DMSO arrays")
        arrays = {name: source[name] for name in ARRAYS}
    if any(x.shape != (len(tables[name]), len(genes)) or x.dtype != np.float32 or not np.isfinite(x).all()
           for name, x in arrays.items()):
        raise ValueError("An X predictor array changed its physical axes")
    return SimpleNamespace(genes=genes, tables=tables, arrays=arrays, receipt=sha256(bundle / "receipt.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    basal = actions.add_parser("basal")
    predict = actions.add_parser("stage")
    for name in ("matrix", "profiles", "basal-reference", "landmarks", "out"):
        basal.add_argument(f"--{name}", type=Path, required=True)
    for name in ("access", "pin", "out"):
        predict.add_argument(f"--{name}", type=Path, required=True)
    predict.add_argument("--slice", required=True)
    args = parser.parse_args()
    print((basal_features if args.action == "basal" else stage_predict)(args))
