"""Evaluate equal eight-context JHH-5 models using only off-position donors."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_donor_panel_size import choices
from src.r19_plan.hcc_lincs_24h_access import CELL, CONTEXT, VIEW, register
from src.r19_plan.hcc_lincs_24h_support import evaluate_support
from src.r19_plan.hcc_lincs_24h_transfer import context, effect, group_means
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import checked_array
from src.r19_plan.jhh5_donor_attribution import shrink_consensus, wrong_parent
from src.r19_plan.jhh_dose_bridge import audited
from src.r19_plan.jhh_low_dose_transfer import finish, load_plan
from src.r19_plan.t4_position import compare
from src.r19_plan.t4_state import state_forecast
from src.r19_plan.t4_workflow import axes, scalar
from src.r19_plan.x_inputs import check_unique


DOSE, COUNT = "10", 8
MEAN, SHRINK, SCALAR, STATE = "mean", "shrink", "scalar", "eight_token_State"
CALIBRATED, WRONG, WRONG_SCALAR = "calibrated_shrink", "wrong_calibrated_shrink", "wrong_scalar"
ARMS = (MEAN, SHRINK, SCALAR, STATE, CALIBRATED, WRONG, WRONG_SCALAR)
PAIRS = tuple((CALIBRATED, ref) for ref in (SCALAR, STATE, WRONG, "B0", MEAN, SHRINK)) + tuple(
    (SCALAR, ref) for ref in (STATE, WRONG_SCALAR, "B0"))


def selected_cells(source):
    counts = source.groupby("parent").agg(cells=("cell_id", "nunique"), plates=("rna_plate", "nunique"))
    enough = counts.index[counts.cells.ge(COUNT) & counts.plates.ge(COUNT)]
    panels = {}
    for parent, group in source.loc[source.parent.isin(enough)].groupby("parent", sort=True):
        names = sorted(group.cell_id.unique())
        order = [names[i] for i in choices(parent, names, 0)[:COUNT]]
        panels[parent] = tuple(order)
    return panels


def access(args):
    tables = audited(args.audit, args.audit_pin)
    target = tables["targets"].loc[lambda rows: rows.cell_id.eq(CELL) & rows.actual_dose_uM.eq(DOSE)].copy()
    all_source = tables["donors"].loc[lambda rows: rows.view.eq(VIEW) & rows.actual_dose_uM.eq(DOSE)
        & rows.parent.isin(target.parent) & ~rows.rna_plate.isin(target.rna_plate)]
    locations = ["parent", "rna_well"]
    forbidden = pd.MultiIndex.from_frame(target[locations].drop_duplicates())
    off = all_source.loc[~pd.MultiIndex.from_frame(all_source[locations]).isin(forbidden)]
    panels = selected_cells(off)
    selected = pd.DataFrame([(p, cell) for p, cells in panels.items() for cell in cells], columns=["parent", "cell_id"])
    source = off.merge(selected, on=["parent", "cell_id"], validate="many_to_one")
    if source.groupby("parent").rna_plate.nunique().lt(COUNT).any():
        raise ValueError("A selected eight-cell panel has fewer than eight physical RNA plates")
    target = target.loc[target.parent.isin(panels)].copy()
    target["fold"] = target.fold.astype(int)
    target["b1_seen"] = target.b1_seen.eq("True")
    figures, partitions, _ = evaluate_support(target, source, CELL, VIEW)
    info = next(item for item in figures if item["scope"] == "donor_covered")
    if not info["eligible"] or min(split["score_parents"] for split in partitions) < 3:
        raise ValueError(f"The registered off-position cohort did not meet size requirements: {info}")

    pos_cols = ["cell_id", "rna_plate", "rna_well"]
    keyed = all_source[pos_cols + ["parent"]].rename(columns={"parent": "same_pos_parent"})
    potential = off[["parent", "cell_id", "rna_plate"]].drop_duplicates().merge(
        target[["parent", "rna_well", "inst_id"]], on="parent", validate="many_to_many")
    matched = potential.merge(keyed, on=pos_cols, validate="many_to_one")
    matched = matched.loc[matched.parent.ne(matched.same_pos_parent)]
    same_position_support = matched.groupby(["parent", "inst_id"]).agg(cells=("cell_id", "nunique"), plates=("rna_plate", "nunique"))
    contexts = pd.concat((target[CONTEXT], source[CONTEXT])).drop_duplicates()
    controls = tables["controls"].merge(contexts, on=CONTEXT, validate="many_to_one")
    check_unique(pd.concat((target, source, controls), ignore_index=True))
    detail = dict(eligible_stats={key: int(info[key]) for key in ("fold0", "fold1", "fold2", "fit_parent_min", "b1_exact_wells")},
        off_donor_wells=len(off), total_donor_wells=len(all_source), panel_cells=COUNT,
        position_matched_control_parents={str(n): int(same_position_support.loc[
            same_position_support.cells.ge(n) & same_position_support.plates.ge(n)].index.get_level_values("parent").nunique())
            for n in (2, 4, 8)})
    target["fold"] = target.fold.astype(str)
    for split in partitions:
        split["fold"] = str(split["fold"])
    register(args, target, source, controls, pd.DataFrame(partitions), detail)


def donor(args, ref, genes, cols, stamp, *, dose=DOSE):
    role = args.role
    tables = {name: read_pinned(args.access / "donor-plan" / role, name, digest)
              for name, digest in ref["donors"][role].items()}
    rows = tables["source.tsv"]
    if not rows.actual_dose_uM.eq(dose).all() or not rows.view.eq(VIEW).all() or len(rows) != ref["donor_wells"]:
        raise ValueError("The eight-context donor inventory changed")
    values = effect(rows, tables["controls.tsv"], ref, genes, cols, role)
    plates, v = group_means(values, rows, ["parent", "cell_id", "rna_plate"])
    cells, v = group_means(v, plates, ["parent", "cell_id"])
    names, panels = [], []
    for parent, idx in cells.groupby("parent", sort=True).indices.items():
        line_names = cells.iloc[idx].cell_id.tolist()
        if len(line_names) != COUNT:
            raise ValueError("A parent lacks its locked eight donor cell lines")
        names.append(parent)
        panels.append(v[idx[choices(parent, line_names, 0)]])
    if len(names) != ref["target_parents"]:
        raise ValueError("The off-position panel lost a target parent")
    out = args.run / "donors" / role
    out.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=".off-donor-") as temporary:
        stage = Path(temporary)
        data = stage / "panels.npz"
        np.savez_compressed(data, parents=np.asarray(names), panels=np.stack(panels))
        write_json(stage / "receipt.json", dict(stamp, role=role, data_sha256=sha256(data)))
        stage.rename(out)


def predict(args, ref, genes, cols, stamp):
    from src.goal_tracks.t2_state import UPSTREAM, load_network

    role = args.group[-1]
    donor_root = args.run / "donors" / role
    receipt = read_json(donor_root / "receipt.json")
    data = donor_root / "panels.npz"
    if receipt["role"] != role or any(receipt[k] != v for k,v in stamp.items()) or sha256(data) != receipt["data_sha256"]:
        raise ValueError("The off-position donor panel changed after extraction")
    with np.load(data, allow_pickle=False) as archive:
        names = pd.Index(archive["parents"])
        panels = checked_array(archive["panels"], (ref["target_parents"], COUNT, len(genes)), "donor panels")
    mean = panels.mean(axis=1)
    shrink = shrink_consensus(panels, [COUNT]*len(panels), mean)
    network = load_network(args.state)
    out = args.run / "forecasts" / args.group
    out.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=".off-predict-") as temporary:
        stage = Path(temporary)
        files, fits = {}, []
        for key, entry in sorted(ref["slices"].items()):
            if not key.startswith(args.group + "-"):
                continue
            fold, _, _, _, plate = axes(key)
            plan = load_plan(args, "predict", key, entry["predict"])
            fit, controls, query = (plan[name] for name in ("fit.tsv", "fit_controls.tsv", "query.tsv"))
            if (set(fit.parent) & set(query.parent) or set(fit.scaffold) & set(query.scaffold)
                    or fit.fold.eq(str(fold)).any() or fit.rna_plate.eq(plate).any() or not query.fold.eq(str(fold)).all()
                    or not query.rna_plate.eq(plate).all() or "level3_row_index" in query):
                raise ValueError("Fit intersects the evaluation parent, scaffold, or plate")
            labels, y = group_means(effect(fit, controls, ref, genes, cols, role), fit, ["parent"])
            permuted = wrong_parent(query, fold, plate)
            i, j, wrong = (names.get_indexer(column) for column in (labels.parent, query.parent, permuted))
            if any((idx < 0).any() for idx in (i, j, wrong)) or (j == wrong).any():
                raise ValueError("A query has a missing donor or an invalid wrong-parent link")
            original_gain = scalar(labels, mean[i], y)
            shrink_gain = scalar(labels, shrink[i], y)
            predictions = {MEAN: mean[j], SHRINK: shrink[j], SCALAR: original_gain*mean[j],
                           STATE: state_forecast(network, (labels, panels[i], y), panels[j]),
                           CALIBRATED: shrink_gain*shrink[j], WRONG: shrink_gain*shrink[wrong], WRONG_SCALAR: original_gain*mean[wrong]}
            dest = stage / (key + ".npz")
            np.savez_compressed(dest, **{arm: checked_array(v, (len(query), len(genes)), arm).astype("float32")
                                        for arm,v in predictions.items()})
            files[key] = sha256(dest)
            fits.append(dict(slice=key, fit_parents=len(labels), query_wells=len(query), scalar_gain=original_gain,
                             shrink_gain=shrink_gain))
        if not files:
            raise ValueError("No registered off-position slice belongs to this group")
        write_json(stage / "receipt.json", dict(stamp, group=args.group, files=files, fits=fits,
            donor_sha256=sha256(donor_root / "receipt.json"), state_upstream=UPSTREAM,
            state_model_sha256=sha256(args.state / "src/state/tx/models/state_transition.py")))
        stage.rename(out)


def score_pairs(rows, truth, outcomes, cell, scope):
    return [dict(x, cell=cell) for x in compare(rows, truth, outcomes, PAIRS, scope)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("access", "donor", "predict", "freeze", "score"))
    for key in ("access", "access-pin", "protocol", "run", "audit", "audit-pin", "cmap-pin", "landmarks", "out", "state"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--role", choices=tuple("ABC"))
    parser.add_argument("--group")
    parser.add_argument("--digest")
    args = parser.parse_args()
    if args.phase == "access":
        access(args)
    else:
        ref, genes, cols, stamp = context(args)
        stamp["off_position_code_sha256"] = sha256(__file__)
        if args.phase in ("donor", "predict"):
            (donor if args.phase == "donor" else predict)(args, ref, genes, cols, stamp)
        else:
            finish(args, ref, genes, cols, stamp, arms=ARMS, score_pairs=score_pairs, panel_count=0)
