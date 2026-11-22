"""Transfer the eight-donor off-position JHH-5 predictor to prior JHH-7 wells."""

import argparse
from hashlib import sha256 as hash_text
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_donor_panel_size import choices
from src.r19_plan.hcc_lincs_24h_access import CONTEXT, VIEW, assigned_controls
from src.r19_plan.hcc_lincs_24h_transfer import effect, group_means
from src.r19_plan.i_score import checked_array, checked_rows
from src.r19_plan.jhh5_donor_attribution import shrink_consensus, wrong_parent
from src.r19_plan.jhh5_off_position import ARMS, CALIBRATED, COUNT, DOSE, MEAN, PAIRS, SCALAR, SHRINK, STATE, WRONG, WRONG_SCALAR, selected_cells
from src.r19_plan.jhh7_donor_transfer import read_plan
from src.r19_plan.jhh7_low_dose import provenance, register
from src.r19_plan.jhh_dose_bridge import audited
from src.r19_plan.t4_position import compare
from src.r19_plan.t4_state import state_forecast
from src.r19_plan.t4_workflow import scalar


def choose_panels(eligible, original, donors):
    keys = ["parent", "rna_well"]
    blocked = pd.MultiIndex.from_frame(original[keys].drop_duplicates())
    usable = donors.loc[donors.parent.isin(eligible.parent)
                        & ~pd.MultiIndex.from_frame(donors[keys]).isin(blocked)]
    selections = selected_cells(usable)
    cells = pd.DataFrame([(parent, c) for parent, chosen in selections.items() for c in chosen], columns=["parent", "cell_id"])
    panel = usable.merge(cells, on=["parent", "cell_id"], validate="many_to_one")
    if panel.groupby("parent").rna_plate.nunique().lt(COUNT).any():
        raise ValueError("A selected donor panel has fewer than eight RNA plates")
    return eligible.loc[eligible.parent.isin(selections)], panel, usable


def access(args):
    tables = audited(args.audit, args.audit_pin)
    target, fit = (tables["targets"].loc[lambda r: r.cell_id.eq(cell) & r.actual_dose_uM.eq(DOSE)] for cell in ("JHH7", "JHH5"))
    candidates = fit.loc[~fit.parent.isin(target.parent) & ~fit.scaffold.isin(target.scaffold)
                         & ~fit.rna_plate.isin(target.rna_plate)]
    for_both = tables["donors"].loc[lambda r: r.view.eq(VIEW) & r.actual_dose_uM.eq(DOSE)
        & ~r.rna_plate.isin(set(fit.rna_plate) | set(target.rna_plate))]
    score, source, off = choose_panels(target, target, for_both)
    training, fit_source, _ = choose_panels(candidates, fit, for_both)
    if (score.parent.nunique() < 80 or score.rna_plate.nunique() != 1 or training.parent.nunique() < 100
            or score.groupby("fold").parent.nunique().reindex(list("012"), fill_value=0).min() < 20):
        raise ValueError("The preregistered cross-cell support is insufficient")
    donors = pd.concat((source, fit_source), ignore_index=True)
    cited = pd.concat((score[CONTEXT], training[CONTEXT], donors[CONTEXT])).drop_duplicates()
    controls = assigned_controls(tables["controls"].merge(cited, on=CONTEXT, validate="many_to_one"))
    if not controls.groupby(CONTEXT).role.nunique().eq(3).all():
        raise ValueError("An admitted context lacks three disjoint vehicle roles")
    paired = off[["parent", "cell_id", "rna_plate"]].drop_duplicates().merge(
        score[["parent", "rna_well"]], on="parent", validate="many_to_many")
    keys = ["cell_id", "rna_plate", "rna_well"]
    paired = paired.merge(for_both[keys + ["parent"]].rename(columns={"parent": "other"}), on=keys, validate="many_to_one")
    paired = paired.loc[paired.parent.ne(paired.other)]
    counts = paired.groupby("parent").agg(cells=("cell_id", "nunique"), plates=("rna_plate", "nunique"))
    details = dict(target_parents=int(score.parent.nunique()), target_wells=len(score), target_plates=1,
        fit_parents=int(training.parent.nunique()), fit_wells=len(training), fit_plates=int(training.rna_plate.nunique()),
        target_controls=int(controls.cell_id.eq("JHH7").sum()), excluded_full_JHH7_parents=int(target.parent.nunique()),
        shared_previous_JHH5_parents=int(score.loc[score.parent.isin(fit.parent)].parent.nunique()),
        b1_target_wells=int(score.b1_seen.eq("True").sum()),
        fold_counts={k: int(v) for k,v in score.groupby("fold").parent.nunique().items()},
        position_matched_support={str(n): int((counts.cells.ge(n) & counts.plates.ge(n)).sum()) for n in (1, 2, 4, 8)})
    register(args, score, training, donors, controls, details)


def context(args):
    ref, genes, cols, stamp = provenance(args)
    stamp["transfer_code_sha256"] = sha256(__file__)
    return ref, genes, cols, stamp


def predict(args, ref, genes, cols, stamp):
    from src.goal_tracks.t2_state import UPSTREAM, load_network

    role = args.role
    plan = read_plan(args, ref, "predict", role)
    fit, query, source = (plan[name] for name in ("fit", "query", "source"))
    if (set(fit.parent) & set(query.parent) or set(fit.scaffold) & set(query.scaffold)
            or set(fit.rna_plate) & set(query.rna_plate) or not fit.cell_id.eq("JHH5").all()
            or not query.cell_id.eq("JHH7").all() or "level3_row_index" in query
            or not source.view.eq(VIEW).all() or not source.actual_dose_uM.eq(DOSE).all()):
        raise ValueError("Prediction access intersects the held JHH-7 scope")
    positions = ["parent", "rna_well"]
    if pd.MultiIndex.from_frame(source[positions]).isin(pd.MultiIndex.from_frame(pd.concat((fit, query))[positions])).any():
        raise ValueError("A donor shares the target RNA well index")
    values = effect(source, plan["source_controls"], ref, genes, cols, role)
    block, values = group_means(values, source, ["parent", "cell_id", "rna_plate"])
    by_cell, values = group_means(values, block, ["parent", "cell_id"])
    donors, vectors = [], []
    for parent, ix in by_cell.groupby("parent", sort=True).indices.items():
        cells = by_cell.cell_id.iloc[ix].tolist()
        if len(cells) != COUNT:
            raise ValueError("Expected exactly eight donor cells in the original recipe")
        donors.append(parent)
        vectors.append(values[ix[choices(parent, cells, 0)]])
    names, vectors = pd.Index(donors), np.stack(vectors)
    mean = vectors.mean(axis=1)
    shrink = shrink_consensus(vectors, [COUNT]*len(vectors), mean)
    fit_rows, y = group_means(effect(fit, plan["fit_controls"], ref, genes, cols, role), fit, ["parent"])
    wrong = np.empty(len(query), dtype=object)
    for (fold, plate), indexes in query.groupby(["fold", "rna_plate"]).indices.items():
        wrong[indexes] = wrong_parent(query.iloc[indexes], fold, plate)
    i, j, k = (names.get_indexer(col) for col in (fit_rows.parent, query.parent, wrong))
    if any((idx < 0).any() for idx in (i, j, k)) or (j == k).any():
        raise ValueError("The frozen source opportunity differs by model")
    gain, gain_shrink = scalar(fit_rows, mean[i], y), scalar(fit_rows, shrink[i], y)
    output = {MEAN: mean[j], SHRINK: shrink[j], SCALAR: gain*mean[j], CALIBRATED: gain_shrink*shrink[j],
              WRONG: gain_shrink*shrink[k], WRONG_SCALAR: gain*mean[k],
              STATE: state_forecast(load_network(args.state), (fit_rows, vectors[i], y), vectors[j])}
    dest = args.run / "forecasts" / role
    dest.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".jhh7-off-") as temporary:
        stage = Path(temporary)
        path = stage / "predictions.npz"
        np.savez_compressed(path, **{arm: checked_array(arr, (len(query), len(genes)), arm).astype("float32") for arm, arr in output.items()})
        write_json(stage / "receipt.json", dict(stamp, role=role, array_sha256=sha256(path),
            fit_parents=len(fit_rows), scalar_gain=gain, shrink_gain=gain_shrink, state_upstream=UPSTREAM,
            state_model_sha256=sha256(args.state / "src/state/tx/models/state_transition.py"),
            wrong_parent_sha256=hash_text("\n".join(wrong).encode()).hexdigest()))
        stage.rename(dest)


def finish(args, ref, genes, cols, stamp):
    path = args.run / "freeze.json"
    scoring = args.phase == "score"
    if scoring and (sha256(path) != args.digest or (args.run / "score").exists()):
        raise ValueError("The cross-cell scorer requires its independent prediction freeze")
    if not scoring and path.exists():
        raise FileExistsError(path)
    pieces, receipts, shuffles, fits = {}, {}, set(), []
    for role in "ABC":
        root = args.run / "forecasts" / role
        receipt = read_json(root / "receipt.json")
        matrix = root / "predictions.npz"
        if (receipt["role"] != role or any(receipt[k] != v for k,v in stamp.items()) or sha256(matrix) != receipt["array_sha256"]
                or receipt["fit_parents"] != ref["fit_parents"]):
            raise ValueError("A role forecast or its lineage changed")
        q = read_plan(args, ref, "predict", role)["query"]
        with np.load(matrix, allow_pickle=False) as packed:
            arrays = {key: checked_array(packed[key], (len(q), len(genes)), key) for key in ARMS}
        pieces[role] = (q, arrays)
        shuffles.add(receipt["wrong_parent_sha256"])
        receipts[role] = sha256(root / "receipt.json")
        fits.append({k: receipt[k] for k in ("role", "fit_parents", "scalar_gain", "shrink_gain")})
    rows = checked_rows(pd.concat([q.assign(role=role) for role, (q, _) in pieces.items()], ignore_index=True), None)
    if (rows.inst_id.nunique() != ref["target_wells"] or rows.parent.nunique() != ref["target_parents"]
            or rows.rna_plate.nunique() != 1 or len(shuffles) != 1 or not rows.groupby("inst_id").role.nunique().eq(3).all()):
        raise ValueError("Prediction support or within-fold control changed across roles")
    frozen = dict(stamp, roles=receipts, parents=ref["target_parents"], wells=ref["target_wells"], fit_parents=ref["fit_parents"])
    if not scoring:
        write_json(path, frozen)
        return
    if read_json(path) != frozen:
        raise ValueError("The scorer received something other than the frozen predictions")

    targets = []
    for i, (role, (q, _)) in enumerate(pieces.items()):
        table = read_plan(args, ref, "score", role)
        if not table["query"].drop(columns="level3_row_index").equals(q):
            raise ValueError("Scored target order differs from the frozen query")
        targets.append(effect(table["query"], table["controls"], ref, genes, cols, "ABC"[(i+2)%3]))
    y = np.concatenate(targets)
    arrays = {key: np.concatenate([a[key] for _, a in pieces.values()]) for key in ARMS}
    arrays["B0"] = np.zeros_like(y)
    per_fold = []
    for fold in range(3):
        keep = rows.fold.eq(fold)
        per_fold.extend(compare(rows.loc[keep].reset_index(drop=True), y[keep], {k:v[keep] for k,v in arrays.items()}, PAIRS, fold))
    with TemporaryDirectory(dir=args.run, prefix=".jhh7-off-score-") as temporary:
        stage = Path(temporary)
        for name, records in (("paired", compare(rows, y, arrays, PAIRS, "all")), ("folds", per_fold), ("fits", fits)):
            pd.DataFrame(records).to_csv(stage / (name + ".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(frozen, freeze_sha256=args.digest,
            files={file.name: sha256(file) for file in stage.iterdir()}))
        stage.rename(args.run / "score")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("access", "predict", "freeze", "score"))
    for key in ("run", "protocol", "audit", "audit-pin", "access-pin", "cmap-pin", "landmarks", "state"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--role", choices=tuple("ABC"))
    parser.add_argument("--digest")
    args = parser.parse_args()
    if args.phase == "access":
        access(args)
    else:
        (predict if args.phase == "predict" else finish)(args, *context(args))
