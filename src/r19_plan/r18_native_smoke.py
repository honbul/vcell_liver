"""Exercise the native R17 CPU and GPU fits through R18 using generated inputs."""

import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from rdkit import Chem

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY
from src.r15_lincs.design import fold_id
from src.r19_plan.r18_access_plan import TARGET_FIELDS
from src.r19_plan.r18_input_bundle import EXPRESSION_TABLES, check_predictor_scope
from src.r19_plan.r18_score_contract import check_slice_id, inspect_forecast, role_names


CASES = ("C2/B/f1-B", "C3_24h_HEPG2_only/A/f2-C")


def chemicals(fold):
    raw = ("C" * n + end for n in range(2, 40) for end in ("", "O", "N", "F", "Cl"))
    parents = sorted({Chem.MolToSmiles(Chem.MolFromSmiles(name)) for name in raw})
    held = [p for p in parents if fold_id(p, "R15-outer", 3) == fold]
    fit = sorted(set(parents) - set(held))
    kept = [p for p in fit if fold_id(p, f"R15-inner-{fold}", 5) != 0][:12]
    selected = [p for p in fit if fold_id(p, f"R15-inner-{fold}", 5) == 0][:5]
    return kept, selected, held[:4]


def designs(slice_id):
    case, scenario, fold, fit_role = check_slice_id(slice_id)
    training, validation, held = chemicals(int(fold))
    releases = ("GSE70138", "CMap2020") if case == "C2" else ("GSE92742",)
    labels, query = [], []
    for study in releases:
        for phase, parents in (("inner_fit", training), ("select", validation), ("query", held)):
            for i, parent in enumerate(parents):
                plate = f"{phase}-{i % 3 if phase == 'inner_fit' else 0}"
                line = "HUH7" if phase == "query" and case.startswith("C3") else "HEPG2"
                dose = .5 if case == "C2" else 3.
                row = (study, f"{phase}-t{i}", parent, line, 24, dose, plate, f"A{i + 1:02d}")
                if phase == "query":
                    query.append(row)
                else:
                    labels.append((*row, phase))
    label_rows = pd.DataFrame(labels, columns=[*TARGET_FIELDS, "inner_phase"])
    query_rows = pd.DataFrame(query, columns=TARGET_FIELDS)
    context = pd.concat((label_rows, query_rows), ignore_index=True)
    _, select, _ = role_names(fit_role)

    def controls(rows, role):
        plates = rows[["study", "rna_plate"]].drop_duplicates().copy()
        return plates.assign(inst_id=plates.rna_plate + f"-{role}", role=role,
                             rna_well=f"H{'ABC'.index(role) + 1:02d}")

    donor_fields = ["study", "inst_id", "parent", "hours", "cell_id", "plate", "cohort",
                    "CONFIRM_R18_B_source_donor_allowed"]
    donor_control_fields = ["study", "inst_id", "plate", "half"]
    donors, vehicles = [], []
    if scenario == "B":
        for study in releases:
            for i, parent in enumerate(training + validation + held):
                for cell in ("A549", "HT29"):
                    donors.append((study, f"donor-{cell}-{i}", parent, 24, cell, "source", case, "True"))
            vehicles.extend((study, f"source-{half}", "source", half) for half in ("1", "2"))
    return {
        "inner_and_outer_labels": label_rows,
        "query_features": query_rows,
        "fit_controls": controls(context, fit_role),
        "select_controls": controls(label_rows.loc[label_rows.inner_phase.eq("select")], select),
        "B_donor_inputs": pd.DataFrame(donors, columns=donor_fields),
        "B_donor_controls": pd.DataFrame(vehicles, columns=donor_control_fields),
    }


def structure(root, parents, gene_map):
    assets = root / "structure"
    features = assets / "features"
    metadata = assets / "metadata"
    features.mkdir(parents=True)
    metadata.mkdir()
    parents = np.array(sorted(parents), dtype=str)
    copyfile(gene_map, metadata / "genes.tsv")
    pd.DataFrame({ENTITY: parents, "smiles": parents}).to_csv(features / "entities.tsv", sep="\t", index=False)
    simulated = np.random.default_rng(182017).normal(0, .2, (len(parents), 384)).astype(np.float32)
    np.savez_compressed(features / "chemberta.npz", parents=parents, embedding=simulated)
    write_json(features / "receipt.json", {"chemberta_sha256": sha256(features / "chemberta.npz")})
    write_json(assets / "receipt.json", dict(
        synthetic_structure_fixture=True, r18_expression_read=False, fitting=False,
        files={str(file.relative_to(assets)): sha256(file) for file in assets.rglob("*") if file.is_file()}))


def build(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    gene_pin = read_json(args.d0_reference)["gene_axis_sha256"]
    if set(gene_pin.values()) != {sha256(args.landmarks)}:
        raise ValueError("Use the R17 landmark metadata for the synthetic model test")
    genes = read_json(args.d0_reference)["landmarks"]
    sources, parents, slices = {}, set(), {}
    for slice_id in CASES:
        case, scenario, _, role = check_slice_id(slice_id)
        tables = designs(slice_id)
        check_predictor_scope(tables, case, scenario, role)
        parents.update(tables["inner_and_outer_labels"].parent)
        parents.update(tables["query_features"].parent)
        bundle = args.out / "bundles" / slice_id
        plan = args.out / "predict-plan" / slice_id
        plan.mkdir(parents=True)
        bundle.mkdir(parents=True)
        for name, rows in tables.items():
            file = f"{name}.tsv"
            rows.to_csv(bundle / file, sep="\t", index=False)
            copyfile(bundle / file, plan / file)
        sources[slice_id] = {"predict": {file.name: sha256(file) for file in plan.iterdir()}}
        rng = np.random.default_rng(18004 + len(slices))
        arrays = {name: rng.normal(6., .3, (len(tables[name]), genes)).astype(np.float32)
                  for name in EXPRESSION_TABLES}
        np.savez_compressed(bundle / "expression.npz", **arrays)
        copyfile(args.landmarks, bundle / "landmarks.tsv")
        slices[slice_id] = (bundle, {name: list(value.shape) for name, value in arrays.items()})
    access = args.out / "access.json"
    write_json(access, dict(synthetic=True, metadata_only=True, expression_read=False, files=sources))
    for slice_id, (bundle, shapes) in slices.items():
        write_json(bundle / "receipt.json", dict(
            slice=slice_id, access_sha256=sha256(access), predict_plan_sha256=sources[slice_id]["predict"],
            arrays=shapes, files={file.name: sha256(file) for file in bundle.iterdir()}))
    structure(args.out, parents, args.gene_map)


def verify(args):
    fixture = args.fixture
    access = read_json(fixture / "access.json")
    if not access["synthetic"] or set(access["files"]) != set(CASES):
        raise ValueError("The native model test accepts only its two simulated cases")
    d0 = read_json(args.d0_reference)
    axis, = set(d0["gene_axis_sha256"].values())
    manifest = {}
    for slice_id in CASES:
        path = args.forecasts / slice_id
        manifest[slice_id] = inspect_forecast(
            path, slice_id, access["files"][slice_id]["predict"], fixture / "predict-plan",
            sha256(args.protocol), axis, d0["landmarks"])
    write_json(args.out, dict(synthetic=True, r18_expression_read=False, model_fit=True, scored=False,
                             cases=manifest, landmark_sha256=axis, code_sha256=sha256(__file__)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    jobs = cli.add_subparsers(dest="action", required=True)
    for action, fields in (("build", ("landmarks", "gene-map", "d0-reference", "out")),
                           ("verify", ("fixture", "forecasts", "protocol", "d0-reference", "out"))):
        job = jobs.add_parser(action)
        for field in fields:
            job.add_argument(f"--{field}", type=Path, required=True)
    args = cli.parse_args()
    {"build": build, "verify": verify}[args.action](args)
