"""Extend the frozen off-position JHH-7 comparison with four public VC adapters."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, LINE
from src.r15_lincs.design import fold_id
from src.r15_lincs.estimators import GAINS, loss
from src.r19_plan.hcc_donor_panel_size import choices
from src.r19_plan.hcc_lincs_24h_access import CONTEXT
from src.r19_plan.hcc_lincs_24h_transfer import effect, group_means
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import BOOTSTRAPS, checked_array, checked_rows, parent_summaries, squared_rows
from src.r19_plan.jhh5_off_position import CALIBRATED, COUNT, SCALAR, STATE
from src.r19_plan.jhh7_donor_transfer import controls_for, read_plan
from src.r19_plan.jhh7_off_position import context
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.t4_position import compare
from src.r19_plan.t4_workflow import scalar


MODELS = ("XPert_morgan", "PRnet-well-adaptation", "chemCPA_P0", "TranSiGen_Tahoe")
REFERENCES = (STATE, *MODELS)
PRNET = "PRnet-well-adaptation"


def references(args):
    previous = SimpleNamespace(run=args.previous, protocol=args.previous_protocol, access_pin=args.access_pin)
    ref, genes, cols, lineage = context(previous)
    if sha256(args.previous / "freeze.json") != sha256(args.previous_pin):
        raise ValueError("The score cohort's historical forecast freeze differs")
    frozen = read_json(args.previous_pin)
    if any(frozen[k] != v for k, v in lineage.items()):
        raise ValueError("The original JHH-7 score cohort, axes, or method changed")
    stamp = dict(cohort_sha256=lineage["access_sha256"], previous_freeze_sha256=sha256(args.previous_pin),
                 protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__))
    return previous, ref, genes, cols, frozen, stamp


def donor_means(data, values):
    plate, vectors = group_means(values, data, ["parent", "cell_id", "rna_plate"])
    cells, vectors = group_means(vectors, plate, ["parent", "cell_id"])
    output = {}
    for parent, idx in cells.groupby("parent", sort=True).indices.items():
        names = cells.cell_id.iloc[idx].tolist()
        if len(names) != COUNT:
            raise ValueError("The model must receive the locked eight-cell donor composition")
        output[parent] = vectors[idx[choices(parent, names, 0)]].mean(axis=0)
    return output


def prepare(args, previous, ref, genes, cols, frozen, stamp):
    role = args.role
    select = "ABC"[("ABC".index(role)+1)%3]
    plan = read_plan(previous, ref, "predict", role)
    select_plan = read_plan(previous, ref, "predict", select)
    fit, query = plan["fit"], plan["query"]
    if (set(fit.parent) & set(query.parent) or set(fit.scaffold) & set(query.scaffold) or not fit.cell_id.eq("JHH5").all()
            or not query.cell_id.eq("JHH7").all() or "level3_row_index" in query):
        raise ValueError("Model preparation encountered a held parent, scaffold, or expression row")
    valid = fit.parent.map(lambda p: fold_id(p, "R19-JHH7-native-select", 5)).eq(0)
    valid_rows = fit.loc[valid]
    valid_ctl = controls_for(select_plan["fit_controls"], valid_rows, select)
    if set(valid_ctl.inst_id) & set(plan["fit_controls"].inst_id):
        raise ValueError("Fit and selection share a physical DMSO")
    inner, y_inner = group_means(effect(fit.loc[~valid], controls_for(plan["fit_controls"], fit.loc[~valid], role),
        ref, genes, cols, role), fit.loc[~valid], ["parent"])
    validation, y_select = group_means(effect(valid_rows, valid_ctl, ref, genes, cols, select), valid_rows, ["parent"])
    training, y_outer = group_means(effect(fit, plan["fit_controls"], ref, genes, cols, role), fit, ["parent"])
    source = plan["source"]
    means = donor_means(source, effect(source, plan["source_controls"], ref, genes, cols, role))
    control = plan["fit_controls"]
    dmso_matrix = extract_matrix(ref["matrices"]["controls"], control, cols["controls"], genes)
    ctr_axis, ctr = group_means(dmso_matrix, control, CONTEXT)
    ctr_inner = ctr[ctr_axis.rna_plate.isin(fit.loc[~valid, "rna_plate"])]
    if (set(inner.parent) & set(validation.parent) or len(training) != ref["fit_parents"]
            or len(inner) < 100 or len(validation) < 20):
        raise ValueError("Inner parent split differs from the registered training support")
    def mapped(rows):
        return np.stack([means[p] for p in rows.parent])

    dest = args.run / "prepared" / role
    dest.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".native-input-") as scratch:
        stage = Path(scratch)
        inner.to_csv(stage / "inner.tsv", sep="\t", index=False)
        validation.to_csv(stage / "validation.tsv", sep="\t", index=False)
        training.to_csv(stage / "training.tsv", sep="\t", index=False)
        query.to_csv(stage / "query.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "input.npz", x_inner=mapped(inner), x_select=mapped(validation), x_outer=mapped(training),
            x_query=mapped(query), y_inner=y_inner, y_select=y_select, y_outer=y_outer, dmso_inner=ctr_inner, dmso_outer=ctr)
        write_json(stage / "receipt.json", dict(stamp, fit_role=role, select_role=select, query_expression_read=False,
            support=[len(inner), len(validation), len(training), len(query)], files={f.name: sha256(f) for f in stage.iterdir()}))
        stage.rename(dest)


def adapted_fit(work, genes, train, query, xt, xq, y, dmso):
    a = scalar(train, xt, y)
    basal = dmso.mean(axis=0)
    r11 = work / "features_input"
    (r11 / "features").mkdir(parents=True)
    (r11 / "metadata").mkdir()
    structures = sorted(set(train.parent) | set(query.parent))
    pd.DataFrame({ENTITY: structures, "smiles": structures}).to_csv(r11 / "features/entities.tsv", sep="\t", index=False)
    (r11 / "metadata/genes.tsv").symlink_to(work.parent / "gene-map.tsv")
    def block(rows, control, treated=None):
        return SimpleNamespace(rows=rows.assign(**{ENTITY: rows.parent, LINE: "JHH5"}), control=control, treated=treated)
    fit = SimpleNamespace(source=SimpleNamespace(r11=r11, dose=10.), panel=genes, weight=np.ones(len(train)),
        train=block(train, basal+a*xt, basal+y), query=block(query, basal+a*xq),
        fit_control=dmso, fit_units=[(str(i), "JHH5") for i in range(len(dmso))])
    return fit, a*xq, a


def native(args, previous, ref, genes, cols, frozen, stamp):
    import torch
    from src.goal_tracks.t2_prnet import fit_model as prnet
    from src.goal_tracks.t2_upstream import EPOCHS, SEED, upstream

    root = args.run / "prepared" / args.role
    reference = read_json(root / "receipt.json")
    if (any(reference[k] != v for k, v in stamp.items()) or reference["fit_role"] != args.role
            or reference["query_expression_read"] or sha256(root / "input.npz") != reference["files"]["input.npz"]):
        raise ValueError("Predictor-only donor receipt changed")
    names = ("inner", "validation", "training", "query")
    inner, val, full, query = [read_pinned(root, n+".tsv", reference["files"][n+".tsv"]) for n in names]
    c = np.load(root / "input.npz", allow_pickle=False)
    dest = args.run / "forecasts" / args.role / args.model
    dest.parent.mkdir(parents=True, exist_ok=True)
    epochs = (100,) if args.model == PRNET else EPOCHS[args.model]

    def sweep(fit, work, checkpoint, evaluate=None):
        torch.manual_seed(SEED)
        torch.set_num_threads(4)
        if args.model == PRNET:
            out, detail = prnet(fit, 10., args.prnet)
            return {100: evaluate(out)} if evaluate is not None else out, detail
        return upstream(args.model)(fit, work, checkpoint, evaluate)

    with TemporaryDirectory(dir=dest.parent, prefix=".native-fit-") as scratch:
        work = Path(scratch)
        (work / "gene-map.tsv").symlink_to(args.gene_map)
        initial, anchor, inner_gain = adapted_fit(work / "inner", genes, inner, val, c["x_inner"], c["x_select"], c["y_inner"], c["dmso_inner"])
        def evaluate(values):
            return {gain: loss(val, c["y_select"], anchor+gain*values) for gain in GAINS}

        options, _ = sweep(initial, work / "inner", epochs, evaluate)
        candidates = [(round(float(value), 9), gain, epoch) for epoch, choices in options.items() for gain, value in choices.items()
                      if np.isfinite(value)]
        if not candidates:
            raise ValueError("The registered inner validation produced no finite comparison")
        _, gain, epoch = min(candidates)
        final, anchor, outer_gain = adapted_fit(work / "outer", genes, full, query, c["x_outer"], c["x_query"], c["y_outer"], c["dmso_outer"])
        residual, model = sweep(final, work / "outer", (epoch,))
        forecast = anchor + gain*residual
        payload = work / "result"
        payload.mkdir()
        np.save(payload / "prediction.npy", checked_array(forecast, (len(query), len(genes)), args.model).astype("float32"), allow_pickle=False)
        write_json(payload / "receipt.json", dict(stamp, role=args.role, model=args.model,
            input_sha256=sha256(root / "receipt.json"), output_sha256=sha256(payload / "prediction.npy"),
            residual_gain=gain, epoch=epoch, selection_loss=float(options[epoch][gain]),
            inner_anchor_gain=inner_gain, outer_anchor_gain=outer_gain, upstream=model, gene_map_sha256=sha256(args.gene_map)))
        payload.rename(dest)
    c.close()


def best_comparison(rows, y, arrays, scope):
    all_arms = (CALIBRATED, *REFERENCES)
    data = [parent_summaries(rows, y.shape[1], squared_rows(y), squared_rows(y, arrays[arm]), squared_rows(y, arrays["B0"])) for arm in all_arms]
    rms, sse, energy = np.column_stack([v[0][:, 0] for v in data]), np.column_stack([v[1][:, 0] for v in data]), data[0][2]
    draws = np.random.default_rng(150104).integers(len(energy), size=(BOOTSTRAPS, len(energy)))
    selected_rmse = rms[draws].mean(axis=1)
    selected_r2 = 1-sse[draws].sum(axis=1)/energy[draws].sum(axis=1, keepdims=True)
    sizes = dict(scope=scope, parents=len(energy), wells=rows.inst_id.nunique())
    summaries, frequency = [], []
    for name, point, selected, choose in (("rmse", rms.mean(axis=0), selected_rmse, np.argmin),
        ("r2_pooled", 1-sse.sum(axis=0)/energy.sum(), selected_r2, np.argmax)):
        point_idx = 1+int(choose(point[1:]))
        draw_idx = 1+choose(selected[:, 1:], axis=1)
        baseline = selected[np.arange(BOOTSTRAPS), draw_idx]
        lo, hi = np.quantile(selected[:, 0]-baseline, (.005,.995))
        rel = (1-point[0]/point[point_idx]) if name == "rmse" else np.nan
        rl, rh = np.quantile(1-selected[:, 0]/baseline, (.005,.995)) if name == "rmse" else (np.nan,np.nan)
        summaries.append(dict(sizes, arm=CALIBRATED, reference="best_VC", metric=name, best_at_point=all_arms[point_idx],
            arm_value=point[0], ref_value=point[point_idx], difference=point[0]-point[point_idx], low99=lo, high99=hi,
            relative_gain=rel, relative_low99=rl, relative_high99=rh))
        frequency.extend(dict(sizes, metric=name, model=arm, wins=int(np.count_nonzero(draw_idx==i))) for i,arm in enumerate(all_arms[1:], 1))
    return summaries, frequency


def finish(args, previous, ref, genes, cols, old, stamp):
    freeze = args.run / "freeze.json"
    scoring = args.phase == "score"
    if scoring and (sha256(freeze) != args.digest or (args.run / "score").exists()):
        raise ValueError("Extended VC scoring requires the independently frozen outputs")
    if not scoring and freeze.exists():
        raise FileExistsError(freeze)
    grouped, inputs, summaries = {}, {}, []
    for role in "ABC":
        root = args.run / "prepared" / role
        inputs[role] = sha256(root / "receipt.json")
        query = read_plan(previous, ref, "predict", role)["query"]
        inherited = args.previous / "forecasts" / role
        if (sha256(inherited / "receipt.json") != old["roles"][role]
                or sha256(inherited / "predictions.npz") != read_json(inherited / "receipt.json")["array_sha256"]):
            raise ValueError("The candidate or original State prediction differs from its freeze")
        with np.load(inherited / "predictions.npz", allow_pickle=False) as a:
            current = {arm: checked_array(a[arm], (len(query), len(genes)), arm) for arm in (CALIBRATED, SCALAR, STATE)}
        for model in MODELS:
            folder = args.run / "forecasts" / role / model
            receipt = read_json(folder / "receipt.json")
            path = folder / "prediction.npy"
            if (receipt["role"] != role or receipt["model"] != model or any(receipt[k] != v for k,v in stamp.items())
                    or receipt["input_sha256"] != inputs[role] or receipt["output_sha256"] != sha256(path)):
                raise ValueError("The new model forecast or its role differs")
            current[model] = checked_array(np.load(path, allow_pickle=False), (len(query), len(genes)), model)
            summaries.append({k:receipt[k] for k in ("role", "model", "epoch", "residual_gain", "selection_loss", "inner_anchor_gain", "outer_anchor_gain")})
        grouped[role] = (query, current)
    manifest = dict(stamp, input_shas=inputs, roles={role:{model:sha256(args.run / "forecasts" / role / model / "receipt.json")
        for model in MODELS} for role in "ABC"})
    if not scoring:
        write_json(freeze, manifest)
        return
    if read_json(freeze) != manifest:
        raise ValueError("Comparison outputs changed after freeze")
    rows = checked_rows(pd.concat([q.assign(role=role) for role,(q,_) in grouped.items()], ignore_index=True), None)
    outcomes = []
    for i,(role,(q,_)) in enumerate(grouped.items()):
        plan = read_plan(previous, ref, "score", role)
        if not plan["query"].drop(columns="level3_row_index").equals(q):
            raise ValueError("The new benchmark must retain original evaluation wells and roles")
        outcomes.append(effect(plan["query"], plan["controls"], ref, genes, cols, "ABC"[(i+2)%3]))
    y = np.concatenate(outcomes)
    arrays = {arm:np.concatenate([d[arm] for _,d in grouped.values()]) for arm in (CALIBRATED, SCALAR, *REFERENCES)}
    arrays["B0"] = np.zeros_like(y)
    pair_choices = tuple((CALIBRATED, arm) for arm in (SCALAR, *REFERENCES, "B0"))
    best, frequency = best_comparison(rows, y, arrays, "all")
    folds = []
    for n in range(3):
        mask = rows.fold.eq(n)
        group, _ = best_comparison(rows.loc[mask], y[mask], {k:a[mask] for k,a in arrays.items()}, n)
        folds.extend(group)
    with TemporaryDirectory(dir=args.run, prefix=".native-score-") as scratch:
        stage = Path(scratch)
        for name, table in (("individual", compare(rows, y, arrays, pair_choices, "all")), ("best", best),
                            ("winner_frequency", frequency), ("folds", folds), ("selection", summaries)):
            pd.DataFrame(table).to_csv(stage / (name+".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(manifest, freeze_sha256=args.digest, score_parents=rows.parent.nunique(),
            score_wells=rows.inst_id.nunique(), files={file.name:sha256(file) for file in stage.iterdir()}))
        stage.rename(args.run / "score")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("prepare", "predict", "freeze", "score"))
    for field in ("run", "protocol", "previous", "previous-pin", "previous-protocol", "access-pin", "gene-map", "prnet"):
        cli.add_argument("--"+field, type=Path, required=True)
    cli.add_argument("--role", choices=tuple("ABC"))
    cli.add_argument("--model", choices=MODELS)
    cli.add_argument("--digest")
    args = cli.parse_args()
    {"prepare":prepare, "predict":native, "freeze":finish, "score":finish}[args.phase](args, *references(args))
