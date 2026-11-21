"""Cross-fit scalar and diagonal calibration with only nonhepatic sources."""

import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.design import fold_id
from src.r15_lincs.estimators import loss
from src.r19_plan.dev_selection import PhysicalControls, ResponseAxis, checked_array
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.r18_predict import vehicle_means
from src.r19_plan.shift_calibration import (
    EffectDeltas, RegisteredControls, control_variance, fit_diagonal, fit_source_scalar,
)
from src.r19_plan.x_inputs import (
    check_unique, control_axis, matrix_axes, read_plan, reference, slice_axes, validate_predict,
)
from src.r19_plan.x_workflow import score_delta

TARGET = "HEPG2"
CONTRACT = "GSE92742-L1000-Level3-physical-well-delta"
ARMS = ("B0", "T0", "T1", "T2", "T3")
PAIRS = tuple((a, "B0") for a in ARMS[1:]) + (
    ("T1", "T0"), ("T2", "T0"), ("T2", "T1"), ("T3", "T1"), ("T3", "T2"),
)
SETTINGS = tuple(product((.25, .5), (.1, .5), np.log([1.25, 1.5])))


def responses(rows, values, controls, control_values):
    normal = vehicle_means(controls, control_values, "rna_plate")
    return values - np.stack([normal[plate] for plate in rows.rna_plate])


def inputs(args, ref):
    target, _, fold, role, next_role, _ = slice_axes(args.slice, ref["cells"])
    _, _, plan = read_plan(args.access, args.pin, args.slice, "predict")
    validate_predict(plan, args.slice, ref["cells"])
    if target != TARGET or ref["parents"] != 493:
        raise ValueError("T1–T3 require the pinned 493-parent HEPG2 target")
    cells = sorted(set(ref["cells"]) - {target})
    source, query, fit = (plan[n] for n in ("labels", "query", "fit_controls"))
    donors = []
    for cell in cells:
        key = f"{cell}-f{fold}-{role}"
        _, _, donor_plan = read_plan(args.access, args.pin, key, "score")
        _, _, donor_query = read_plan(args.access, args.pin, key, "predict")
        rows = donor_plan["labels"]
        if (not rows.cell_id.eq(cell).all() or set(rows.parent) != set(query.parent)
                or not rows.drop(columns="level3_row_index").equals(donor_query["query"])):
            raise ValueError("T1–T3 B donors differ from the registered nonhepatic held metadata")
        donors.append(rows)
    donor = pd.concat(donors, ignore_index=True)
    _, _, rotated = read_plan(args.access, args.pin, f"{TARGET}-f{fold}-{next_role}", "predict")
    select = rotated["fit_controls"].loc[rotated["fit_controls"].cell_id.isin(cells)]
    control_axis(select, source, next_role)
    check_unique(pd.concat((source, query, donor, fit, select), ignore_index=True))
    genes_frame, cols = matrix_axes(args.access, ref)
    genes = tuple(genes_frame.pr_gene_id)
    tables = (source, donor, fit, select)
    raw = extract_matrix(ref["matrix"], pd.concat(tables, ignore_index=True), cols, np.asarray(genes))
    earlier, held, controls, select_controls = np.split(raw, np.cumsum([len(t) for t in tables[:-1]]))
    return cells, query, source, donor, fit, genes, (
        responses(source, earlier, fit, controls), responses(source, earlier, select, select_controls),
        responses(donor, held, fit, controls), controls,
    )


def indexed_means(rows, values):
    groups = rows.groupby(["cell_id", "parent"], sort=True).indices
    axis = pd.MultiIndex.from_tuples(list(groups), names=["cell_id", "parent"])
    return axis, np.stack([values[idx].mean(axis=0) for idx in groups.values()])


def uniform(means, parents, cells):
    axis, values = means
    indices = axis.get_indexer(pd.MultiIndex.from_product((cells, parents)))
    if (indices < 0).any():
        raise ValueError("A matched donor cell and parent are absent in T1–T3")
    return values[indices].reshape(len(cells), len(parents), values.shape[1]).mean(axis=0)


def make_controls(rows, data, genes, cells):
    mask = rows.cell_id.isin(cells).to_numpy()
    selected = rows.loc[mask]
    values = data[mask].astype(np.float64)
    for positions in selected.groupby("cell_id").indices.values():
        values[positions] -= values[positions].mean(axis=0)
    wells = tuple(selected[["rna_plate", "rna_well"]].itertuples(index=False, name=None))
    return RegisteredControls(PhysicalControls(wells, genes, values), wells, CONTRACT)


def rule_factors(fit, data, genes, cells, held):
    src = make_controls(fit, data, genes, cells)
    dest = make_controls(fit, data, genes, [held])
    base = max(float(np.median(control_variance(src, target=False))), 1e-8)
    return {params: fit_diagonal(src, dest, alpha=params[0], floor=params[1]*base,
                                 clip=params[2]).factors for params in SETTINGS}


def gains(parts, genes, settings):
    rows = pd.concat([p[0] for p in parts], ignore_index=True)
    ids = tuple(rows[["study", "inst_id"]].itertuples(index=False, name=None))
    axis = ResponseAxis(ids, genes)
    truth = EffectDeltas(axis, CONTRACT, np.concatenate([p[2] for p in parts]))
    forecasts = {None: np.concatenate([p[1] for p in parts])}
    forecasts.update({s: np.concatenate([p[1] * p[3][s] for p in parts]) for s in settings})
    return {s: fit_source_scalar(EffectDeltas(axis, CONTRACT, values), truth).gain for s, values in forecasts.items()}


def calibration(cells, query, source, donor, fit, genes, values, fold):
    source_fit, source_select, donor_values, controls = values
    means = indexed_means(source, source_fit)
    parent_group = source.parent.map(lambda p: fold_id(p, f"T123-inner-{fold}", 3)).to_numpy()

    def pseudo(candidates, target, parent_mask):
        test = source.cell_id.eq(target).to_numpy() & parent_mask
        rows = source.loc[test]
        forecast = uniform(means, rows.parent, candidates)
        directions = rule_factors(fit, controls, genes, candidates, target)
        if len(candidates) < 2 or rows.parent.nunique() < 20 or set(rows.parent) & set(query.parent):
            raise ValueError("Pseudo-target has inadequate source domains or disjoint parents")
        return rows, forecast, source_select[test], directions

    inner_losses = {setting: [] for setting in SETTINGS}
    for hold in cells:
        inner = [cell for cell in cells if cell != hold]
        for inner_fold in range(3):
            train = [pseudo([c for c in inner if c != cell], cell, parent_group != inner_fold) for cell in inner]
            validation = pseudo(inner, hold, parent_group == inner_fold)
            settings = gains(train, genes, SETTINGS)
            rows, prediction, truth, factors = validation
            for s in SETTINGS:
                inner_losses[s].append(loss(rows, truth, settings[s] * factors[s] * prediction))
    objectives = {s: float(np.mean(observations)) for s, observations in inner_losses.items()}
    if not np.isfinite(list(objectives.values())).all():
        raise ValueError("Nonfinite T1–T3 source-only independent selection loss")
    selected = min(SETTINGS, key=lambda s: (round(objectives[s], 9), s))
    outer = [pseudo([c for c in cells if c != cell], cell, np.ones(len(source), bool)) for cell in cells]
    fitted = gains(outer, genes, [selected])
    raw = uniform(indexed_means(donor, donor_values), query.parent, cells)
    adjusted = raw * rule_factors(fit, controls, genes, cells, TARGET)[selected]
    forecasts = dict(B0=np.zeros_like(raw), T0=raw, T1=fitted[None]*raw, T2=adjusted, T3=fitted[selected]*adjusted)
    details = dict(alpha=selected[0], floor_multiplier=selected[1], log_clip=selected[2], loss=objectives[selected],
                   T1_gain=fitted[None], T3_gain=fitted[selected], source_parents=source.parent.nunique(),
                   source_wells=len(source), query_donor_wells=len(donor), target_fit_dmsos=int(fit.cell_id.eq(TARGET).sum()),
                   target_fit_plates=fit.loc[fit.cell_id.eq(TARGET), "rna_plate"].nunique())
    return forecasts, details


def stamp(args):
    return dict(access_sha256=sha256(args.pin), protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
                estimator_sha256=sha256(Path(__file__).with_name("shift_calibration.py")))


def run_predict(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    ref = reference(args.pin)
    *data, values = inputs(args, ref)
    fold = slice_axes(args.slice, ref["cells"])[2]
    forecasts, setting = calibration(*data, values, fold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".t123-predict-") as temp:
        stage = Path(temp)
        np.savez_compressed(stage / "forecasts.npz", **{k: v.astype("float32") for k, v in forecasts.items()})
        write_json(stage / "selection.json", setting)
        write_json(stage / "receipt.json", dict(stamp(args), slice=args.slice,
            query_sha256=ref["slices"][args.slice]["predict"]["query.tsv"],
            files={f.name: sha256(f) for f in stage.iterdir()}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


def comparison_rows(rows, truth, forecasts):
    energy = squared_rows(truth)
    errors = {arm: squared_rows(truth, values) for arm, values in forecasts.items()}
    results = []
    for scope in ("ALL", 0, 1, 2):
        mask = np.ones(len(rows), bool) if scope == "ALL" else rows.fold.eq(scope).to_numpy()
        for arm, baseline in PAIRS:
            values = parent_summaries(rows.loc[mask], truth.shape[1], energy[mask], errors[arm][mask], errors[baseline][mask])
            points, ci, draws = paired_effects(*values)
            low, high, _ = relative_interval(values[0])
            for metric, (a, b), (lo, hi), n in zip(METRICS, points, ci, draws, strict=True):
                results.append(dict(scope=scope, arm=arm, reference=baseline, metric=metric, arm_value=a, reference_value=b,
                    difference=a-b, low99=lo, high99=hi, relative_gain=1-a/b if metric=="rmse" and b>0 else np.nan,
                    relative_low99=low if metric=="rmse" else np.nan, relative_high99=high if metric=="rmse" else np.nan,
                    parents=rows.loc[mask].parent.nunique(), wells=rows.loc[mask].inst_id.nunique(), valid_draws=int(n)))
    return pd.DataFrame(results)


def evaluate(args):
    scoring = args.action == "score"
    if scoring and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("Scoring requires the independent T1–T3 freeze digest")
    if not scoring and args.freeze.exists():
        raise FileExistsError(args.freeze)
    ref = reference(args.pin)
    genes_frame, cols = matrix_axes(args.access, ref)
    genes = genes_frame.pr_gene_id.to_numpy()
    keys = sorted(key for key in ref["slices"] if key.startswith(TARGET+"-"))
    lineage = stamp(args)
    queries, receipts, options, arrays = {}, {}, [], {arm: [] for arm in ARMS}
    for key in keys:
        _, _, plan = read_plan(args.access, args.pin, key, "predict")
        path = args.forecasts / key
        receipt = read_json(path / "receipt.json")
        if (any(receipt[k] != h for k, h in lineage.items()) or receipt["slice"] != key
                or receipt["query_sha256"] != ref["slices"][key]["predict"]["query.tsv"]
                or set(receipt["files"]) != {"forecasts.npz", "selection.json"}
                or any(sha256(path / name) != h for name, h in receipt["files"].items())):
            raise ValueError("T1–T3 forecast or selection lineage changed")
        queries[key] = plan["query"]
        with np.load(path / "forecasts.npz", allow_pickle=False) as data:
            if set(data.files) != set(ARMS):
                raise ValueError("T1–T3 forecast arm inventory changed")
            for arm in ARMS:
                values = checked_array(data[arm], (len(queries[key]), len(genes)), arm)
                if scoring:
                    arrays[arm].append(values)
        receipts[key] = sha256(path / "receipt.json")
        options.append(dict(slice=key, **read_json(path / "selection.json")))
    rows = checked_rows(pd.concat([query.assign(fold=slice_axes(k, ref["cells"])[2],
                        role=slice_axes(k, ref["cells"])[3]) for k, query in queries.items()], ignore_index=True), None)
    if rows.parent.nunique() != ref["parents"] or not rows.groupby("inst_id").role.nunique().eq(3).all():
        raise ValueError("T1–T3 forecasts lost a held parent or DMSO rotation")
    locked = dict(lineage, slices=receipts)
    if not scoring:
        write_json(args.freeze, locked)
        return sha256(args.freeze)
    if read_json(args.freeze) != locked:
        raise ValueError("T1–T3 predictions changed after freeze")
    truth = np.concatenate([score_delta(args, k, queries[k], genes_frame, cols) for k in keys])
    results = comparison_rows(rows, truth, {k: np.concatenate(v) for k, v in arrays.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".t123-score-") as temp:
        stage = Path(temp)
        results.to_csv(stage / "paired.tsv", sep="\t", index=False, na_rep="NA")
        pd.DataFrame(options).to_csv(stage / "selection.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(locked, freeze_sha256=args.expected_sha256,
            parents=rows.parent.nunique(), wells=rows.inst_id.nunique(),
            files={f.name: sha256(f) for f in stage.iterdir()}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "freeze", "score"))
    for name in ("access", "pin", "protocol", "out", "forecasts", "freeze"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--slice")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print((run_predict if args.action == "predict" else evaluate)(args))
