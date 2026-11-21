"""Freeze position-disjoint prospective HCC forecasts without prospective outcomes."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_donor_panel_size import choices
from src.r19_plan.hcc_lincs_24h_access import CONTEXT, assigned_controls
from src.r19_plan.hcc_lincs_24h_transfer import context, effect, group_means
from src.r19_plan.hcc_off_position_pilot import SLOTS, verify_layout
from src.r19_plan.i_preflight import read_pinned, write_plan
from src.r19_plan.i_score import checked_array
from src.r19_plan.jhh5_donor_attribution import shrink_consensus
from src.r19_plan.jhh5_off_position import ARMS, CALIBRATED, COUNT, MEAN, SCALAR, SHRINK, STATE, WRONG, WRONG_SCALAR
from src.r19_plan.jhh7_donor_transfer import controls_for
from src.r19_plan.jhh_low_dose_transfer import load_plan
from src.r19_plan.r18_matrix_index import metadata_index
from src.r19_plan.t4_scaffold_transfer import scaffold
from src.r19_plan.t4_state import state_forecast
from src.r19_plan.t4_workflow import axes, scalar
from src.r19_plan.x_inputs import check_unique


def references(args):
    old = SimpleNamespace(access=args.previous / "access", access_pin=args.previous_access_pin, protocol=args.previous_protocol)
    ref, genes, cols, lineage = context(old)
    if sha256(args.previous / "freeze.json") != sha256(args.previous_freeze_pin):
        raise ValueError("The source JHH-5 development run is different")
    frozen = read_json(args.previous_freeze_pin)
    if any(frozen[k] != v for k, v in lineage.items()):
        raise ValueError("The JHH-5 source axes differ from its freeze")
    if sha256(args.pilot / "receipt.json") != sha256(args.pilot_pin):
        raise ValueError("The prospective layout and donor plan differ from the pinned audit")
    pilot = read_json(args.pilot_pin)
    if not pilot["metadata_only"] or pilot["expression_read"] or pilot["measured"] or pilot["selected"] != 96 or pilot["actual_treatment_slots"] != 576:
        raise ValueError("The full prospective roster and 576 verified target positions are required")
    stamp = dict(previous_freeze_sha256=sha256(args.previous_freeze_pin), pilot_sha256=sha256(args.pilot_pin),
                 protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__))
    return old, ref, genes, cols, frozen, pilot, stamp


def wrong_assignment(layout, positions):
    used = layout.loc[layout.kind.eq("treated")].groupby("parent_slot").well.agg(set)
    rng = np.random.default_rng(191198)
    cost = rng.random((len(SLOTS), len(SLOTS)))
    for i, a in enumerate(SLOTS):
        for j, b in enumerate(SLOTS):
            if a == b or used[a] & positions[b]:
                cost[i, j] = np.inf
    ii, jj = linear_sum_assignment(cost)
    if len(ii) != len(SLOTS) or not np.isfinite(cost[ii, jj]).all():
        raise ValueError("A position-disjoint wrong-compound permutation is unavailable")
    return pd.DataFrame(dict(parent_slot=[SLOTS[i] for i in ii], wrong_slot=[SLOTS[j] for j in jj]))


def access(args, old, ref, genes, cols, frozen, pilot, stamp):
    read = lambda name: read_pinned(args.pilot, name, pilot["files"][name])
    slots = read("slots.tsv").sort_values("parent_slot", ignore_index=True)
    sources = read("donor_wells.tsv").assign(study="CMap2020")
    controls = read("donor_controls.tsv").assign(study="CMap2020")
    layout = pd.read_csv(args.pilot / "plate_layout.tsv", sep="\t", keep_default_na=True)
    if sha256(args.pilot / "plate_layout.tsv") != pilot["files"]["plate_layout.tsv"]:
        raise ValueError("The prospective coordinate map differs from its registered digest")
    layout[["parent_slot", "vehicle_role"]] = layout[["parent_slot", "vehicle_role"]].fillna("")
    positions = sources.groupby("parent_slot").rna_well.agg(set).to_dict()
    verify_layout(layout, positions)
    if (not sources.actual_dose_uM.eq("10").all() or not sources.groupby("parent").cell_id.nunique().eq(COUNT).all()
            or sources.groupby("parent").rna_plate.nunique().lt(COUNT).any() or tuple(slots.parent_slot) != SLOTS
            or not slots.parent.is_unique or not slots.scaffold_unit.is_unique
            or not slots.scaffold_unit.eq(slots.parent.map(scaffold)).all()):
        raise ValueError("Prospective identities or molecular scaffolds differ from the chemical audit")
    link = wrong_assignment(layout, positions)
    matrices = ref["matrices"]
    sources, _, trt = metadata_index(Path(matrices["treated"]["path"]), sources, genes)
    controls, _, ctl = metadata_index(Path(matrices["controls"]["path"]), controls, genes)
    if dict(treated=trt, controls=ctl) != matrices:
        raise ValueError("The prospective donor indices belong to a different matrix release")
    controls = assigned_controls(controls)
    with TemporaryDirectory(dir=args.run, prefix=".pilot-access-") as temporary:
        table = Path(temporary)
        details, support = {}, []
        for role in "ABC":
            splits = [(key, entry) for key, entry in ref["slices"].items() if axes(key)[1] == role]
            # The complete, already-scored development cohort can be used for prospective fitting.
            # The union of its earlier scaffold-purged fitting slices can omit some parents.
            train = pd.concat([read_pinned(old.access / "score-plan" / key, "query.tsv", entry["score"]["query.tsv"])
                               for key, entry in splits], ignore_index=True).drop_duplicates()
            full_controls = pd.concat([load_plan(old, "predict", key, entry["predict"])["fit_controls.tsv"]
                                       for key, entry in splits], ignore_index=True).drop_duplicates()
            if train.parent.nunique() != frozen["support"]["parents"] or not train.cell_id.eq("JHH5").all():
                raise ValueError("The historical fitting union does not reproduce its frozen scope")
            fit = train.loc[~train.parent.isin(slots.parent) & ~train.scaffold.isin(slots.scaffold_unit)]
            if fit.parent.nunique() < 100:
                raise ValueError("The prospective scaffold-purged training support is insufficient")
            fit_ctl, source_ctl = (controls_for(frame, rows, role) for frame, rows in ((full_controls, fit), (controls, sources)))
            check_unique(pd.concat((fit, fit_ctl, sources, source_ctl), ignore_index=True))
            details[role] = write_plan(table / role, {"fit.tsv": fit, "fit_controls.tsv": fit_ctl,
                "source.tsv": sources, "source_controls.tsv": source_ctl})
            support.append(dict(role=role, fit_parents=fit.parent.nunique(), fit_wells=len(fit), fit_plates=fit.rna_plate.nunique(),
                                source_wells=len(sources), source_plates=sources.rna_plate.nunique()))
        for name, rows in (("slots", slots), ("wrong", link), ("support", pd.DataFrame(support))):
            rows.to_csv(table / (name + ".tsv"), sep="\t", index=False)
        write_json(table / "receipt.json", dict(stamp, metadata_only=True, expression_read=False, target_measured=False,
            role_files=details, files={file.name: sha256(file) for file in table.glob("*.tsv")}))
        table.rename(args.run / "access")


def trusted_access(args, stamp):
    root = args.run / "access"
    access = read_json(root / "receipt.json")
    if any(access[k] != v for k, v in stamp.items()) or not access["metadata_only"] or access["expression_read"] or access["target_measured"]:
        raise ValueError("The predictor access metadata are inconsistent")
    return root, access, dict(stamp, access_sha256=sha256(root / "receipt.json"))


def old_panels(args, ref, genes, frozen, role):
    old = args.previous / "forecasts" / f"f0-{role}" / "receipt.json"
    if sha256(old) != frozen["groups"][f"f0-{role}"]:
        raise ValueError("The prior development role receipt changed")
    folder = args.previous / "donors" / role
    if sha256(folder / "receipt.json") != read_json(old)["donor_sha256"]:
        raise ValueError("The prior eight-line donor arrays changed")
    data = folder / "panels.npz"
    if sha256(data) != read_json(folder / "receipt.json")["data_sha256"]:
        raise ValueError("The prior donor matrix is no longer the frozen input")
    with np.load(data, allow_pickle=False) as blocks:
        return pd.Index(blocks["parents"]), checked_array(blocks["panels"], (ref["target_parents"], COUNT, len(genes)), "development inputs")


def predict(args, old, ref, genes, cols, frozen, pilot, stamp):
    from src.goal_tracks.t2_state import UPSTREAM, load_network

    root, lock, stamp = trusted_access(args, stamp)
    role = args.role
    plan = {name[:-4]: read_pinned(root / role, name, h) for name, h in lock["role_files"][role].items()}
    slots, wrong = [read_pinned(root, name, lock["files"][name]) for name in ("slots.tsv", "wrong.tsv")]
    fit, source = plan["fit"], plan["source"]
    if set(fit.parent) & set(slots.parent) or set(fit.scaffold) & set(slots.scaffold_unit):
        raise ValueError("A prospective identity or scaffold appears in fitting")
    labels, y = group_means(effect(fit, plan["fit_controls"], ref, genes, cols, role), fit, ["parent"])
    parents, known = old_panels(args, ref, genes, frozen, role)
    i = parents.get_indexer(labels.parent)
    if (i < 0).any():
        raise ValueError("A development parent lacks its positional donor panel")
    train = known[i]
    values = effect(source, plan["source_controls"], ref, genes, cols, role)
    plates, vectors = group_means(values, source, ["parent_slot", "parent", "cell_id", "rna_plate"])
    cells, vectors = group_means(vectors, plates, ["parent_slot", "parent", "cell_id"])
    target, names = [], []
    for key, indexes in cells.groupby(["parent_slot", "parent"], sort=True).indices.items():
        labels8 = cells.cell_id.iloc[indexes].tolist()
        if len(labels8) != COUNT:
            raise ValueError("A forecast lacks eight registered donor vectors")
        names.append(key[0])
        target.append(vectors[indexes[choices(key[1], labels8, 0)]])
    if tuple(names) != SLOTS or not wrong.parent_slot.eq(slots.parent_slot).all():
        raise ValueError("The prospective forecast order differs from its model inputs")
    target = np.stack(target)
    wrong_order = pd.Index(names).get_indexer(wrong.wrong_slot)
    if (wrong_order < 0).any() or len(set(wrong_order)) != len(SLOTS) or np.any(wrong_order == np.arange(len(SLOTS))):
        raise ValueError("Invalid fixed wrong-compound control")
    mfit, mnew = train.mean(axis=1), target.mean(axis=1)
    sfit = shrink_consensus(train, [COUNT]*len(train), mfit)
    snew = shrink_consensus(target, [COUNT]*len(target), mnew)
    a, b = scalar(labels, mfit, y), scalar(labels, sfit, y)
    output = {MEAN:mnew, SHRINK:snew, SCALAR:a*mnew, CALIBRATED:b*snew, WRONG:b*snew[wrong_order],
              WRONG_SCALAR:a*mnew[wrong_order], "B0":np.zeros_like(mnew),
              STATE:state_forecast(load_network(args.state), (labels, train, y), target)}
    folder = args.run / "forecasts" / role
    folder.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=folder.parent, prefix=".pilot-forecast-") as temporary:
        stage = Path(temporary)
        data = stage / "predictions.npz"
        np.savez_compressed(data, slots=np.asarray(names), genes=genes,
            **{arm: checked_array(arr, (96, len(genes)), arm).astype("float32") for arm,arr in output.items()})
        write_json(stage / "receipt.json", dict(stamp, role=role, array_sha256=sha256(data), fit_parents=len(labels),
            scalar_gain=a, shrink_gain=b, state_upstream=UPSTREAM,
            state_model_sha256=sha256(args.state / "src/state/tx/models/state_transition.py"), prospective_labels_read=False))
        stage.rename(folder)


def freeze(args, old, ref, genes, cols, frozen, pilot, stamp):
    _, access, stamp = trusted_access(args, stamp)
    roles = {}
    for role in "ABC":
        folder = args.run / "forecasts" / role
        receipt = read_json(folder / "receipt.json")
        data = folder / "predictions.npz"
        if (receipt["role"] != role or receipt["prospective_labels_read"] or any(receipt[k] != v for k,v in stamp.items())
                or sha256(data) != receipt["array_sha256"]):
            raise ValueError("A prospective array differs from its prediction receipt")
        with np.load(data, allow_pickle=False) as values:
            if tuple(values["slots"]) != SLOTS or not np.array_equal(values["genes"], genes):
                raise ValueError("The forecast slot or landmark ordering differs")
            for arm in (*ARMS, "B0"):
                checked_array(values[arm], (96, len(genes)), arm)
        roles[role] = sha256(folder / "receipt.json")
    path = args.run / "freeze.json"
    if path.exists():
        raise FileExistsError(path)
    write_json(path, dict(stamp, roles=roles, slots=96, genes=len(genes), target_measured=False,
                          planned_lines=pilot["planned_lines"], wrong_sha256=access["files"]["wrong.tsv"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("access", "predict", "freeze"))
    for key in ("run", "protocol", "previous", "previous-access-pin", "previous-freeze-pin", "previous-protocol", "pilot", "pilot-pin", "state"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--role", choices=tuple("ABC"))
    args = parser.parse_args()
    {"access": access, "predict": predict, "freeze": freeze}[args.phase](args, *references(args))
