"""Write separate R18 extraction allow-lists from pinned metadata, without reading expression."""

import argparse
from pathlib import Path

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r19_plan.r18_matrix_index import STUDIES
from src.r19_plan.r18_preflight import checked_tables

WELL = ["study", "inst_id"]
PLATE = ["study", "rna_plate"]
SOURCE_KEY = ["study", "parent", "hours"]
TARGET_FIELDS = ["study", "inst_id", "parent", "cell_id", "hours", "dose", "rna_plate", "rna_well"]


def keys(rows, fields=WELL):
    return pd.MultiIndex.from_frame(rows[fields])


def at(rows, requested, fields):
    return rows.loc[keys(rows, fields).isin(keys(requested, fields))].copy()


def add_indices(rows, indices, allowed):
    wanted = indices[["study", "inst_id", "use", "matrix", "level3_row_index"]]
    merged = rows.merge(wanted, on=WELL, how="left", validate="one_to_one")
    if not merged.use.eq(allowed).all() or merged.level3_row_index.isna().any():
        raise ValueError(f"Missing or misclassified Level 3 input for {allowed}")
    return merged.drop(columns="use")


def control_receipt(controls, plates, role, indices):
    chosen = at(controls.loc[controls.role.eq(role)], plates, PLATE)
    if not keys(plates, PLATE).isin(keys(chosen, PLATE)).all():
        raise ValueError("A registered target plate lacks the requested DMSO role")
    return add_indices(chosen, indices, "target_role_DMSO")


def donor_lists(donors, donor_controls, scope, cohort, scenario, indices):
    relevant = donors.loc[donors.cohort.eq(cohort)]
    if scenario == "A":
        return add_indices(relevant.iloc[:0], indices, "B_donor"), add_indices(
            donor_controls.iloc[:0], indices, "B_donor_DMSO")
    wanted = scope[SOURCE_KEY].drop_duplicates()
    donor = at(relevant, wanted, SOURCE_KEY)
    if not keys(wanted, SOURCE_KEY).isin(keys(donor, SOURCE_KEY)).all():
        raise ValueError("B donor measurements do not cover every registered fit and query row")
    donor = add_indices(donor, indices, "B_donor")
    control = at(donor_controls, donor, ["study", "plate"])
    required = donor[["study", "plate"]].drop_duplicates()
    for half in ("1", "2"):
        have = control.loc[control.half.eq(half)]
        if not keys(required, ["study", "plate"]).isin(keys(have, ["study", "plate"])).all():
            raise ValueError("A registered B donor lacks a requested control half")
    return donor, add_indices(control, indices, "B_donor_DMSO")


def verify_domains(fit, score, predictor_wells, scorer_wells):
    if set(fit.parent) & set(score.parent):
        raise ValueError("A scored parent entered predictor target labels")
    if len(predictor_wells.intersection(scorer_wells)):
        raise ValueError("A score-only physical well entered predictor expression inputs")


def slice_plan(block, rotation, controls, donors, source_control, indices):
    groups = {phase: data[TARGET_FIELDS].copy() for phase, data in block.groupby("phase", sort=False)}
    fitting = groups["outer_fit"]
    query = groups["score"]
    inner, selection = groups["inner_fit"], groups["select"]
    inner_wells, selection_wells, outer_wells = keys(inner), keys(selection), keys(fitting)
    if (len(inner_wells.intersection(selection_wells)) or set(inner.parent) & set(selection.parent)
        or not outer_wells.is_unique or not inner_wells.is_unique or not selection_wells.is_unique
        or set(outer_wells) != set(inner_wells) | set(selection_wells)):
        raise ValueError("Inner splits do not partition outer-fit physical wells")
    labels = add_indices(fitting, indices, "target_treatment")
    labels["inner_phase"] = pd.Series(pd.NA, index=labels.index, dtype="string")
    labels.loc[keys(labels).isin(keys(inner)), "inner_phase"] = "inner_fit"
    labels.loc[keys(labels).isin(keys(selection)), "inner_phase"] = "select"
    if labels.inner_phase.isna().any():
        raise ValueError("An outer-fit treatment label has no inner phase")

    context = pd.concat([fitting, query], ignore_index=True)
    fit_control = control_receipt(controls, context, rotation.fit, indices)
    select_control = control_receipt(controls, selection, rotation.select, indices)
    score_control = control_receipt(controls, query, rotation.score, indices)
    cohort = rotation.case.split("_")[0]
    donor, donor_control = donor_lists(donors, source_control, context, cohort, rotation.scenario, indices)
    score_labels = add_indices(query, indices, "target_treatment")
    allowed = pd.concat([labels, fit_control, select_control, donor, donor_control], ignore_index=True)
    prohibited = pd.concat([score_labels, score_control], ignore_index=True)
    verify_domains(fitting, query, keys(allowed), keys(prohibited))
    return {
        "inner_and_outer_labels": labels, "query_features": query, "fit_controls": fit_control,
        "select_controls": select_control, "B_donor_inputs": donor, "B_donor_controls": donor_control
    }, {"score_labels": score_labels, "score_controls": score_control}


def append_index(index_root, reference_root, split_ref, donor_ref):
    axes, controls = [], []
    for study in STUDIES:
        folder = index_root / study
        stamp = checked_tables(folder, reference_root / study / "receipt.json")
        if (not stamp["metadata_only"] or stamp["expression_read"] or
            stamp["lock_sha256"] != split_ref or stamp["audited_donor_sha256"] != donor_ref):
            raise ValueError(f"Wrong metadata-only index lineage for {study}")
        axes.append(table(folder / "level3_rows.tsv"))
        controls.append(table(folder / "donor_controls.tsv"))
    index = pd.concat(axes, ignore_index=True)
    if index.duplicated(WELL).any():
        raise ValueError("A locked well occurs twice on the raw index axis")
    return index, pd.concat(controls, ignore_index=True)


def dump(root, identifier, tables):
    folder = root / identifier
    folder.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        rows.to_csv(folder / f"{name}.tsv", sep="\t", index=False)
    return {path.name: sha256(path) for path in sorted(folder.iterdir())}


def run(args):
    checked_tables(args.lock, args.lock_reference)
    registered = checked_tables(args.registered, args.registered_reference)
    lock_sha = sha256(args.lock_reference)
    indices, donor_controls = append_index(
        args.index, args.index_reference, lock_sha, registered["files"]["r18_B_donor_exceptions.tsv"])
    rows = table(args.lock / "split_rows.tsv")
    controls = table(args.lock / "control_assignments.tsv")
    donors = table(args.registered / "r18_B_donor_exceptions.tsv")
    rotations = table(args.lock / "rotations.tsv")
    groups = rows.groupby(["case", "scenario", "fold"], sort=False)
    out = args.out
    out.mkdir(exist_ok=False, parents=True)
    receipt = {}
    support = []
    for rotation in rotations.itertuples(index=False):
        if {rotation.fit, rotation.select, rotation.score} != set("ABC"):
            raise ValueError("Target-role compartments are not disjoint")
        slice_id = f"{rotation.case}/{rotation.scenario}/f{rotation.fold}-{rotation.fit}"
        fit, score = slice_plan(groups.get_group((rotation.case, rotation.scenario, rotation.fold)),
                               rotation, controls, donors, donor_controls, indices)
        prediction_hashes = dump(out / "predict-plan", slice_id, fit)
        score_hashes = dump(out / "score-plan", slice_id, score)
        receipt[slice_id] = dict(predict=prediction_hashes, score=score_hashes)
        support.append(dict(slice=slice_id, **{key: len(frame) for key, frame in (fit | score).items()}))
    pd.DataFrame(support).to_csv(out / "support.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", dict(lock_sha256=lock_sha, registered_sha256=sha256(args.registered_reference),
        metadata_only=True, expression_read=False, fit=False, score=False, slices=len(receipt),
        support_sha256=sha256(out / "support.tsv"), manifest_sha256=sha256(__file__), files=receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "lock-reference", "registered", "registered-reference", "index", "index-reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
