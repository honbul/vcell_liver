"""Score matched public VC adaptations on the original six-plate JHH-5 cohort."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.design import fold_id
from src.r19_plan.hcc_lincs_24h_access import CONTEXT
from src.r19_plan.hcc_lincs_24h_transfer import context, effect, group_means
from src.r19_plan.hcc_off_position_pilot_freeze import old_panels
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import checked_array, checked_rows
from src.r19_plan.jhh5_off_position import CALIBRATED, SCALAR, STATE
from src.r19_plan.jhh7_donor_transfer import controls_for
from src.r19_plan.jhh7_off_position_native import MODELS, REFERENCES, best_comparison, native
from src.r19_plan.jhh_low_dose_transfer import load_plan
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.t4_position import compare
from src.r19_plan.t4_workflow import axes, scalar


def references(args):
    old = SimpleNamespace(access=args.previous / "access", access_pin=args.access_pin, protocol=args.previous_protocol)
    ref, genes, cols, lineage = context(old)
    freeze = read_json(args.previous_pin)
    if (sha256(args.previous / "freeze.json") != sha256(args.previous_pin)
            or any(freeze[k] != v for k, v in lineage.items())
            or freeze["support"] != {"parents": 305, "plates": 6, "wells": 309}):
        raise ValueError("The original JHH-5 cohort or freeze changed")
    stamp = dict(previous_freeze_sha256=sha256(args.previous_pin), access_sha256=lineage["access_sha256"],
        protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
        adapter_code_sha256=sha256(Path(__file__).with_name("jhh7_off_position_native.py")))
    return old, ref, genes, cols, freeze, stamp


def previous_prediction(args, frozen, key, shape):
    path = args.previous / "forecasts" / key[:4]
    receipt = path / "receipt.json"
    if sha256(receipt) != frozen["groups"][key[:4]]:
        raise ValueError("Original JHH-5 forecast receipt changed")
    data = path / (key + ".npz")
    if sha256(data) != read_json(receipt)["files"][key]:
        raise ValueError("Original JHH-5 forecasts changed")
    with np.load(data, allow_pickle=False) as cells:
        return {arm: checked_array(cells[arm], shape, arm) for arm in (CALIBRATED, SCALAR, STATE)}


def prepare(args, old, ref, genes, cols, frozen, stamp, key, names, means):
    fold, fit_role, select_role, _, plate = axes(key)
    other = f"f{fold}-{select_role}-{plate}"
    plan = load_plan(old, "predict", key, ref["slices"][key]["predict"])
    select_plan = load_plan(old, "predict", other, ref["slices"][other]["predict"])
    fit, controls, query = (plan[name] for name in ("fit.tsv", "fit_controls.tsv", "query.tsv"))
    if (not fit.equals(select_plan["fit.tsv"]) or not query.equals(select_plan["query.tsv"])
            or set(fit.parent) & set(query.parent) or set(fit.scaffold) & set(query.scaffold)
            or fit.fold.eq(str(fold)).any() or fit.rna_plate.eq(plate).any()
            or not query.fold.eq(str(fold)).all() or not query.rna_plate.eq(plate).all()
            or "level3_row_index" in query):
        raise ValueError("The nested adapter fit intersects the outer held fold, scaffold or plate")
    valid = fit.parent.map(lambda parent: fold_id(parent, "R19-JHH7-native-select", 5)).eq(0)
    inner_rows, select_rows = fit.loc[~valid], fit.loc[valid]
    valid_ctl = controls_for(select_plan["fit_controls.tsv"], select_rows, select_role)
    if set(controls.inst_id) & set(valid_ctl.inst_id):
        raise ValueError("Fit and selection share physical DMSO")
    inner_ctl = controls_for(controls, inner_rows, fit_role)
    inner, y_inner = group_means(effect(inner_rows, inner_ctl, ref, genes, cols, fit_role), inner_rows, ["parent"])
    valid, y_select = group_means(effect(select_rows, valid_ctl, ref, genes, cols, select_role), select_rows, ["parent"])
    train, y_outer = group_means(effect(fit, controls, ref, genes, cols, fit_role), fit, ["parent"])
    dmso = extract_matrix(ref["matrices"]["controls"], controls, cols["controls"], genes)
    ctr_axis, ctr = group_means(dmso, controls, CONTEXT)
    ctr_inner = ctr[ctr_axis.rna_plate.isin(inner_rows.rna_plate)]
    if set(inner.parent) & set(valid.parent) or len(inner) < 90 or len(valid) < 20:
        raise ValueError("Nested parent support differs from the preflight")

    def donor(rows):
        idx = names.get_indexer(rows.parent)
        if (idx < 0).any():
            raise ValueError("A locked parent lacks its off-position donor profile")
        return means[idx]

    x_query, x_outer = donor(query), donor(train)
    original = previous_prediction(args, frozen, key, x_query.shape)
    np.testing.assert_array_equal((scalar(train, x_outer, y_outer)*x_query).astype("float32"), original[SCALAR])
    dest = args.run / "prepared" / key
    dest.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".jhh5-native-") as scratch:
        stage = Path(scratch)
        for name, table in (("inner", inner), ("validation", valid), ("training", train), ("query", query)):
            table.to_csv(stage / (name + ".tsv"), sep="\t", index=False)
        np.savez_compressed(stage / "input.npz", x_inner=donor(inner), x_select=donor(valid), x_outer=x_outer,
            x_query=x_query, y_inner=y_inner, y_select=y_select, y_outer=y_outer, dmso_inner=ctr_inner, dmso_outer=ctr)
        write_json(stage / "receipt.json", dict(stamp, fit_role=key, fit_dmso_role=fit_role, select_dmso_role=select_role,
            query_expression_read=False, anchor_verified=True, support=[len(inner), len(valid), len(train), len(query)],
            files={path.name: sha256(path) for path in stage.iterdir()}))
        stage.rename(dest)


def collect(args, old, ref, genes, frozen, stamp):
    queries, predictions, selections, inputs, receipts = {}, {}, [], {}, {}
    for key, entry in sorted(ref["slices"].items()):
        root = args.run / "prepared" / key
        prepared = root / "receipt.json"
        details = read_json(prepared)
        inputs[key] = sha256(prepared)
        query = read_pinned(root, "query.tsv", details["files"]["query.tsv"])
        shape = (len(query), len(genes))
        if (sha256(old.access / "predict-plan" / key / "query.tsv") != entry["predict"]["query.tsv"]
                or not query.equals(read_pinned(old.access / "predict-plan" / key, "query.tsv", entry["predict"]["query.tsv"]))
                or details["query_expression_read"] or not details["anchor_verified"]):
            raise ValueError("Prepared forecasts differ from the permitted query manifest")
        queries[key] = query
        predictions[key] = previous_prediction(args, frozen, key, shape)
        receipts[key] = {}
        for model in MODELS:
            folder = args.run / "forecasts" / key / model
            receipt = read_json(folder / "receipt.json")
            array = folder / "prediction.npy"
            if (receipt["role"] != key or receipt["model"] != model
                    or any(receipt[k] != v for k, v in stamp.items())
                    or receipt["input_sha256"] != inputs[key] or receipt["output_sha256"] != sha256(array)):
                raise ValueError("The public forecast differs from its prepared input or its receipt")
            predictions[key][model] = checked_array(np.load(array, allow_pickle=False), shape, model)
            receipts[key][model] = sha256(folder / "receipt.json")
            selections.append({k: receipt[k] for k in (
                "role", "model", "epoch", "residual_gain", "selection_loss", "inner_anchor_gain", "outer_anchor_gain")})
    rows = checked_rows(pd.concat([query.assign(role=axes(key)[1]) for key, query in queries.items()], ignore_index=True), None)
    support = dict(parents=rows.parent.nunique(), wells=rows.inst_id.nunique(), plates=rows.rna_plate.nunique())
    if support != frozen["support"] or not rows.groupby("inst_id").role.nunique().eq(3).all():
        raise ValueError("The public forecasts do not cover the original physical support")
    manifest = dict(stamp, support=support, inputs=inputs, forecasts=receipts)
    return queries, rows, predictions, selections, manifest


def finish(args, old, ref, genes, cols, frozen, stamp):
    path = args.run / "freeze.json"
    if args.phase == "score":
        if sha256(path) != args.digest or (args.run / "score").exists():
            raise ValueError("Separate scoring must use the pre-existing forecast freeze")
    elif path.exists():
        raise FileExistsError(path)
    queries, rows, predictions, selections, manifest = collect(args, old, ref, genes, frozen, stamp)
    if args.phase == "freeze":
        write_json(path, manifest)
        return
    if read_json(path) != manifest:
        raise ValueError("The benchmark forecast manifest changed")
    targets = []
    for key, query in queries.items():
        tables = load_plan(old, "score", key, ref["slices"][key]["score"])
        target = tables["query.tsv"]
        if not target.drop(columns="level3_row_index").equals(query):
            raise ValueError("The original score wells and predictor query differ")
        targets.append(effect(target, tables["score_controls.tsv"], ref, genes, cols, axes(key)[3]))

    y = np.concatenate(targets)
    arrays = {arm: np.concatenate([values[arm] for values in predictions.values()])
        for arm in (CALIBRATED, SCALAR, *REFERENCES)}
    arrays["B0"] = np.zeros_like(y)
    best, frequency = best_comparison(rows, y, arrays, "all")
    pairs = tuple((CALIBRATED, arm) for arm in (SCALAR, *REFERENCES, "B0"))
    tables = dict(best=best, winner_frequency=frequency, individual=compare(rows, y, arrays, pairs, "all"),
        selection=selections)
    scopes = {"folds": [(i, rows.fold.eq(i)) for i in range(3)],
              "leave_plate": [(plate, rows.rna_plate.ne(plate)) for plate in sorted(rows.rna_plate.unique())]}
    for name, subsets in scopes.items():
        tables[name] = []
        for label, keep in subsets:
            best_subset, _ = best_comparison(rows.loc[keep], y[keep], {arm: a[keep] for arm, a in arrays.items()}, label)
            tables[name].extend(best_subset)
    with TemporaryDirectory(dir=args.run, prefix=".jhh5-native-score-") as scratch:
        stage = Path(scratch)
        for name, table in tables.items():
            pd.DataFrame(table).to_csv(stage / (name + ".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(manifest, freeze_sha256=args.digest,
            files={file.name: sha256(file) for file in stage.iterdir()}))
        stage.rename(args.run / "score")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "predict", "freeze", "score"))
    for name in ("run", "protocol", "previous", "previous-protocol", "previous-pin", "access-pin", "gene-map", "prnet"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--group")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--digest")
    args = parser.parse_args()
    old, ref, genes, cols, frozen, stamp = references(args)
    if args.phase in ("prepare", "predict"):
        keys = [key for key in sorted(ref["slices"]) if key.startswith(args.group + "-")]
        if len(keys) != 6:
            raise ValueError("The group does not include all six original plate slices")
        if args.phase == "prepare":
            names, panels = old_panels(args, ref, genes, frozen, args.group[-1])
            for key in keys:
                prepare(args, old, ref, genes, cols, frozen, stamp, key, names, panels.mean(axis=1))
        else:
            for key in keys:
                args.role = key
                native(args, old, ref, genes, cols, frozen, stamp)
    else:
        finish(args, old, ref, genes, cols, frozen, stamp)
