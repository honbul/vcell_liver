"""Freeze four additional public VC predictions for the unmeasured 96-parent pilot."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.design import fold_id
from src.r19_plan.hcc_lincs_24h_access import CONTEXT
from src.r19_plan.hcc_lincs_24h_transfer import effect, group_means
from src.r19_plan.hcc_off_position_pilot import SLOTS
from src.r19_plan.hcc_off_position_pilot_freeze import old_panels, references as pilot_references
from src.r19_plan.hcc_off_position_pilot_score import frozen_arrays
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import checked_array
from src.r19_plan.jhh5_off_position import SCALAR
from src.r19_plan.jhh7_donor_transfer import controls_for
from src.r19_plan.jhh7_off_position_native import MODELS, donor_means, native
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.t4_workflow import scalar


def references(args):
    # The original source checks validate the original forecast protocol, not this supplementary protocol.
    original_args = argparse.Namespace(**(vars(args) | {"protocol": args.forecast_protocol}))
    old, ref, genes, cols, development, audit, orig_stamp = pilot_references(original_args)
    original = args.forecast / "freeze.json"
    if sha256(original) != sha256(args.forecast_pin):
        raise ValueError("The base prospective forecast differs from its registered freeze")
    frozen = read_json(args.forecast_pin)
    access = args.forecast / "access" / "receipt.json"
    if frozen["target_measured"] or sha256(access) != frozen["access_sha256"] or any(frozen[k] != v for k,v in orig_stamp.items()):
        raise ValueError("The base pilot freeze or predictor authority changed")
    plan = read_json(access)
    if plan["target_measured"] or plan["expression_read"] or not plan["metadata_only"]:
        raise ValueError("Prospective model fitting requires outcome-free access manifests")
    stamp = dict(original_freeze_sha256=sha256(args.forecast_pin), original_access_sha256=sha256(access),
        protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__))
    return old, ref, genes, cols, development, frozen, plan, stamp


def role_inputs(args, plan, role):
    root = args.forecast / "access" / role
    return {name.removesuffix(".tsv"): read_pinned(root, name, h) for name,h in plan["role_files"][role].items()}


def prepare(args, old, ref, genes, cols, development, frozen, plan, stamp):
    role = args.role
    select = "ABC"[("ABC".index(role)+1)%3]
    our = role_inputs(args, plan, role)
    other = role_inputs(args, plan, select)
    slots = read_pinned(args.forecast / "access", "slots.tsv", plan["files"]["slots.tsv"])
    fit, ctl = our["fit"], our["fit_controls"]
    if (tuple(slots.parent_slot) != SLOTS or set(fit.parent) & set(slots.parent) or set(fit.scaffold) & set(slots.scaffold_unit)
            or not fit.cell_id.eq("JHH5").all() or not fit.actual_dose_uM.eq("10").all()):
        raise ValueError("A prospectively held compound, scaffold or context has crossed the fitting boundary")
    valid = fit.parent.map(lambda p: fold_id(p, "R19-JHH7-native-select", 5)).eq(0)
    inner_rows, select_rows = fit.loc[~valid], fit.loc[valid]
    valid_ctl = controls_for(other["fit_controls"], select_rows, select)
    if set(ctl.inst_id) & set(valid_ctl.inst_id):
        raise ValueError("Fit and select share a physical vehicle")
    inner, y_inner = group_means(effect(inner_rows, controls_for(ctl, inner_rows, role), ref, genes, cols, role), inner_rows, ["parent"])
    validation, y_select = group_means(effect(select_rows, valid_ctl, ref, genes, cols, select), select_rows, ["parent"])
    train, y_outer = group_means(effect(fit, ctl, ref, genes, cols, role), fit, ["parent"])
    control_values = extract_matrix(ref["matrices"]["controls"], ctl, cols["controls"], genes)
    ordered, dmso = group_means(control_values, ctl, CONTEXT)
    inner_dmso = dmso[ordered.rna_plate.isin(inner_rows.rna_plate)]
    donors, panels = old_panels(args, ref, genes, development, role)
    positions = donors.get_indexer(train.parent)
    if (positions < 0).any() or len(train) != 277 or len(inner) < 100 or len(validation) < 20:
        raise ValueError("The fixed predictor and nested selection lineage differs")
    m = panels[positions].mean(axis=1)
    ix = pd.Index(train.parent)
    source = our["source"]
    new_means = donor_means(source, effect(source, our["source_controls"], ref, genes, cols, role))
    x_query = np.stack([new_means[p] for p in slots.parent])
    # Check the reconstructed scalar independently against the already frozen fit plan.
    _, original = frozen_arrays(args, frozen)
    a = scalar(train, m, y_outer)
    np.testing.assert_array_equal((a*x_query).astype("float32"), original[SCALAR]["ABC".index(role)])
    dest = args.run / "prepared" / role
    dest.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".pilot-native-input-") as scratch:
        stage = Path(scratch)
        for name, table in (("inner", inner), ("validation", validation), ("training", train), ("query", slots[["parent_slot", "parent"]])):
            table.to_csv(stage / (name+".tsv"), sep="\t", index=False)
        np.savez_compressed(stage / "input.npz", x_inner=m[ix.get_indexer(inner.parent)], x_select=m[ix.get_indexer(validation.parent)],
            x_outer=m, x_query=x_query, y_inner=y_inner, y_select=y_select, y_outer=y_outer, dmso_inner=inner_dmso, dmso_outer=dmso)
        write_json(stage / "receipt.json", dict(stamp, fit_role=role, select_role=select, query_expression_read=False,
            support=[len(inner),len(validation),len(train),len(slots)], anchor_verified=True,
            files={file.name:sha256(file) for file in stage.iterdir()}))
        stage.rename(dest)


def freeze(args, old, ref, genes, cols, development, frozen, plan, stamp):
    receipts = {}
    for role in "ABC":
        root = args.run / "prepared" / role
        receipt = read_json(root / "receipt.json")
        query = read_pinned(root, "query.tsv", receipt["files"]["query.tsv"])
        if tuple(query.parent_slot) != SLOTS or receipt["query_expression_read"] or not receipt["anchor_verified"]:
            raise ValueError("The prospective slot order or source anchor changed")
        receipts[role] = {}
        for model in MODELS:
            folder = args.run / "forecasts" / role / model
            record = read_json(folder / "receipt.json")
            payload = folder / "prediction.npy"
            if (record["input_sha256"] != sha256(root / "receipt.json") or record["output_sha256"] != sha256(payload)
                    or record["role"] != role or record["model"] != model or any(record[k]!=v for k,v in stamp.items())):
                raise ValueError("A public comparator differs from the prepared authority")
            checked_array(np.load(payload, allow_pickle=False), (96, len(genes)), model)
            receipts[role][model] = sha256(folder / "receipt.json")
    write_json(args.run / "freeze.json", dict(stamp, models=list(MODELS), roles=receipts, slots=list(SLOTS),
        genes=[str(g) for g in genes], target_measured=False,
        prepared={role:sha256(args.run / "prepared" / role / "receipt.json") for role in "ABC"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "predict", "freeze"))
    for key in ("run", "protocol", "forecast", "forecast-pin", "forecast-protocol", "previous", "previous-access-pin",
                "previous-freeze-pin", "previous-protocol", "pilot", "pilot-pin", "gene-map", "prnet"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--role", choices=tuple("ABC"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    data = references(args)
    if args.phase == "predict":
        old, ref, genes, cols, development, _, _, stamp = data
        native(args, old, ref, genes, cols, development, stamp)
    else:
        (prepare if args.phase == "prepare" else freeze)(args, *data)
