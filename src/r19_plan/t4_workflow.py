"""Predict, freeze and score registered matched LINCS time experiments."""

import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.design import fold_id
from src.r15_lincs.estimators import loss
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.i_stage import parse_tsv, pinned_bytes
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.r18_predict import vehicle_means
from src.r19_plan.t4_access import CELL, COHORTS, CONDITION, PHH_RANK, PHH_RIDGE, PREDICT, SCORE, condition_index
from src.r19_plan.time_operator import RANKS, RIDGES, fit_time_operator
from src.r19_plan.x_inputs import check_unique, control_axis, matrix_axes

ARMS = ("B0", "identity6", "scalar6", "operator6", "direct24", "logdose6")
PHH_PROTOCOL_SHA256 = "590b915275e15b44f6ccef203a784685a4003d6f6144976bd71528cf2e960c16"
PAIRS = (
    ("identity6", "B0"), ("scalar6", "B0"), ("operator6", "B0"), ("direct24", "B0"),
    ("logdose6", "identity6"), ("operator6", "identity6"), ("operator6", "scalar6"),
    ("operator6", "direct24"), ("scalar6", "identity6"),
)


def reference(args):
    data = read_json(args.pin)
    cell = data.get("cell", CELL)
    if (sha256(args.access / "receipt.json") != sha256(args.pin) or not data["metadata_only"]
            or data["expression_read"] or (data["parents"], data["conditions"]) != COHORTS.get(cell)
            or (cell == "PHH" and ((data.get("fixed_rank"), data.get("fixed_ridge")) != (PHH_RANK, PHH_RIDGE)
                                  or sha256(args.protocol) != PHH_PROTOCOL_SHA256))):
        raise ValueError("T4 requires its independently pinned metadata-only access receipt")
    return data


def plans(args, ref, key, phase, names):
    allowed = {name + ".tsv" for name in names}
    files = ref["slices"][key][phase]
    if set(files) != allowed:
        raise ValueError("Unexpected T4 data role inventory")
    root = args.access / (phase + "-plan") / key
    return {name: parse_tsv(pinned_bytes(root / (name + ".tsv"), files[name + ".tsv"])) for name in names}


def axes(key):
    fold, role, plate = key.split("-", 2)
    role_index = "ABC".index(role)
    return int(fold.removeprefix("f")), role, "ABC"[(role_index+1)%3], "ABC"[(role_index+2)%3], plate


def validate_predict(tables, key, cell=CELL):
    fold, role, select_role, _, plate = axes(key)
    source = [tables[name] for name in ("source6", "source24")]
    donors = [tables[name] for name in ("donor6", "direct24")]
    query, fit, select = [tables[name] for name in ("query", "fit_controls", "select_controls")]
    training = pd.concat(source, ignore_index=True)
    donor = pd.concat(donors, ignore_index=True)
    selection = tables["source24"].loc[tables["source24"].phase.eq("select")]
    parent_fold = lambda rows: rows.parent.map(lambda p: fold_id(p, "R15-outer", 3))
    nested = source[1].parent.map(lambda p: cell != "PHH" and fold_id(p, f"R15-inner-{fold}", 5) == 0)
    if ("level3_row_index" in query or not query.rna_plate.eq(plate).all() or not query.hours.eq("24").all()
            or not all(rows.hours.eq(hours).all() for rows, hours in zip((*source, *donors), ("6", "24", "6", "24")))
            or not parent_fold(query).eq(fold).all() or parent_fold(training).eq(fold).any()
            or set(donor.parent) != set(query.parent) or not nested.eq(source[1].phase.eq("select")).all()
            or not source[1].phase.isin(["select", "inner_fit"]).all()
            or (cell == "PHH" and (not select.empty or source[1].parent.nunique() <= PHH_RANK))
            or any(rows.rna_plate.eq(plate).any() for rows in (*source, *donors, fit, select))):
        raise ValueError("T4 prediction violates the independent parent, time or physical plate role")
    active = pd.concat((training, donor), ignore_index=True)
    combined = pd.concat((active, query, fit, select), ignore_index=True)
    if not combined.cell_id.eq(cell).all():
        raise ValueError("Only the registered context is allowed for T4")
    check_unique(combined)
    control_axis(fit, active, role)
    if cell != "PHH":
        control_axis(select, selection, select_role)


def deltas(rows, values, control_rows, controls):
    means = vehicle_means(control_rows, controls, "rna_plate")
    return values - np.stack([means[plate] for plate in rows.rna_plate])


def average_conditions(rows, values):
    groups = rows.groupby(CONDITION, sort=True).indices
    keys = pd.MultiIndex.from_tuples(list(groups), names=CONDITION)
    return keys, np.stack([values[indices].mean(axis=0) for indices in groups.values()])


def lookup(candidate, rows):
    keys, values = candidate
    positions = keys.get_indexer(condition_index(rows))
    if (positions < 0).any():
        raise ValueError("No matched parent–exact dose measurement in the supplied time role")
    return values[positions]


def pairs(source6, target, outcome):
    condition, later = average_conditions(target, outcome)
    rows = condition.to_frame(index=False).assign(study="GSE92742", cell_id=target.cell_id.iloc[0])
    rows["dose_uM"] = rows.actual_dose_uM
    return rows, lookup(source6, rows), later


def scalar(rows, earlier, later):
    w = 1. / rows.groupby("parent").parent.transform("size").to_numpy()
    energy = np.einsum("i,ij,ij->", w, earlier, earlier, dtype=np.float64)
    numerator = np.einsum("i,ij,ij->", w, earlier, later, dtype=np.float64)
    if not energy > 0:
        raise ValueError("No source temporal contrast energy for T4 scalar fitting")
    return float(np.clip(numerator / energy, 0, 2))


def logdose(candidate, rows):
    keys, values = candidate
    parents, doses = keys.get_level_values("parent"), keys.get_level_values("actual_dose_uM").astype(float)
    logged = np.log2(doses)
    result = []
    for parent, dose in rows[CONDITION].itertuples(index=False, name=None):
        same = parents == parent
        weight = np.exp(-.5 * (logged[same] - np.log2(float(dose))) ** 2)
        result.append(np.average(values[same], axis=0, weights=weight))
    return np.stack(result)


def selected_settings(data, effects, genes, fit, source6, held):
    outer = data["source24"]
    select_mask = outer.phase.eq("select").to_numpy()
    select = outer.loc[select_mask]
    inner = pairs(source6, outer.loc[~select_mask], effects["source24"][~select_mask])
    select6 = lookup(source6, select)
    options = []
    for rank, ridge in product(RANKS, RIDGES):
        if rank >= min(inner[0].parent.nunique(), fit[0].parent.nunique()):
            continue
        try:
            model = fit_time_operator(*inner, genes, rank=rank, ridge=ridge, held_parents=held | set(select.parent))
        except ValueError as exc:
            if str(exc).startswith("Insufficient numerical support"):
                continue
            raise
        objective = loss(select, effects["select24"], model.predict(select6, genes))
        if not np.isfinite(objective):
            raise ValueError("Nonfinite T4 inner independent selection loss")
        options.append((round(objective, 9), rank, ridge, objective))
    if not options:
        raise ValueError("No rank is supported by the T4 source and selection pairs")
    _, rank, ridge, objective = min(options)
    return dict(rank=rank, ridge=ridge, select_loss=objective, inner_parents=inner[0].parent.nunique(),
                select_parents=select.parent.nunique())


def run_model(data, effects, genes):
    source6 = average_conditions(data["source6"], effects["source6"])
    donor = average_conditions(data["donor6"], effects["donor6"])
    direct = average_conditions(data["direct24"], effects["direct24"])
    outer, query = data["source24"], data["query"]
    fit = pairs(source6, outer, effects["source24"])
    query6 = lookup(donor, query)
    held = set(query.parent)
    settings = (dict(rank=PHH_RANK, ridge=PHH_RIDGE, select_loss=None, inner_parents=0, select_parents=0)
                if query.cell_id.eq("PHH").all() else selected_settings(data, effects, genes, fit, source6, held))
    model = fit_time_operator(*fit, genes, rank=settings["rank"], ridge=settings["ridge"], held_parents=held)
    gain = scalar(*fit)
    forecasts = dict(B0=np.zeros_like(query6), identity6=query6, scalar6=gain*query6,
                     operator6=model.predict(query6, genes), direct24=lookup(direct, query), logdose6=logdose(donor, query))
    return forecasts, dict(settings, scalar_gain=gain, fit_parents=model.fit_parent_count)


def lineage(args):
    return dict(access_sha256=sha256(args.pin), protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
                operator_sha256=sha256(Path(__file__).with_name("time_operator.py")))


def predict(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    ref = reference(args)
    data = plans(args, ref, args.slice, "predict", PREDICT)
    cell = ref.get("cell", CELL)
    validate_predict(data, args.slice, cell)
    genes_frame, cols = matrix_axes(args.access, ref)
    genes = genes_frame.pr_gene_id.to_numpy()
    names = [n for n in PREDICT if n != "query"]
    admitted = pd.concat([data[n] for n in names], ignore_index=True)
    raw = extract_matrix(ref["matrix"], admitted, cols, genes)
    values = dict(zip(names, np.split(raw, np.cumsum([len(data[n]) for n in names[:-1]])), strict=True))
    effects = {n: deltas(data[n], values[n], data["fit_controls"], values["fit_controls"]) for n in names[:4]}
    if cell != "PHH":
        mask = data["source24"].phase.eq("select").to_numpy()
        effects["select24"] = deltas(data["source24"].loc[mask], values["source24"][mask],
                                     data["select_controls"], values["select_controls"])
    forecasts, selection = run_model(data, effects, genes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".t4-predict-") as tmp:
        stage = Path(tmp)
        np.savez_compressed(stage / "forecasts.npz", **{name: values.astype("float32") for name, values in forecasts.items()})
        write_json(stage / "selection.json", selection)
        write_json(stage / "receipt.json", dict(lineage(args), slice=args.slice,
                   query_sha256=ref["slices"][args.slice]["predict"]["query.tsv"],
                   files={f.name: sha256(f) for f in stage.iterdir()}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


def score_delta(args, ref, key, query, cols, genes):
    data = plans(args, ref, key, "score", SCORE)
    rows, controls = data["query"], data["controls"]
    if not rows.drop(columns="level3_row_index").equals(query):
        raise ValueError("T4 score values have a different registered query row order")
    check_unique(pd.concat((rows, controls), ignore_index=True))
    control_axis(controls, rows, axes(key)[3])
    raw = extract_matrix(ref["matrix"], pd.concat((rows, controls), ignore_index=True), cols, genes)
    return deltas(rows, raw[:len(rows)], controls, raw[len(rows):])


def paired_results(rows, truth, forecasts):
    energy = squared_rows(truth)
    errors = {arm: squared_rows(truth, values) for arm, values in forecasts.items()}
    parts = {"ALL": np.ones(len(rows), dtype=bool)}
    parts.update({f"fold_{fold}": rows.fold.eq(fold).to_numpy() for fold in range(3)})
    parts.update({f"{dose}_uM": rows.actual_dose_uM.eq(dose).to_numpy() for dose in ("10", "2.5")})
    results = []
    for scope, mask in parts.items():
        if not mask.any():
            continue
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
        raise ValueError("T4 scoring requires a new output and the independent freeze digest")
    if not scoring and args.freeze.exists():
        raise FileExistsError(args.freeze)
    ref = reference(args)
    genes_frame, cols = matrix_axes(args.access, ref)
    genes = genes_frame.pr_gene_id.to_numpy()
    expected = lineage(args)
    queries, receipts, options, arrays = {}, {}, [], {arm: [] for arm in ARMS}
    for key, manifest in sorted(ref["slices"].items()):
        path = args.forecasts / key
        receipt = read_json(path / "receipt.json")
        query_digest = manifest["predict"]["query.tsv"]
        if (any(receipt[k] != h for k, h in expected.items()) or receipt["slice"] != key
                or receipt["query_sha256"] != query_digest or set(receipt["files"]) != {"forecasts.npz", "selection.json"}
                or any(sha256(path / name) != h for name, h in receipt["files"].items())):
            raise ValueError("T4 forecast, slice, or prediction lineage changed")
        queries[key] = parse_tsv(pinned_bytes(args.access/"predict-plan"/key/"query.tsv", query_digest))
        with np.load(path / "forecasts.npz", allow_pickle=False) as file:
            if set(file.files) != set(ARMS):
                raise ValueError("T4 forecast arm inventory changed")
            for arm in ARMS:
                values = checked_array(file[arm], (len(queries[key]), len(genes)), arm)
                if scoring:
                    arrays[arm].append(values)
        receipts[key] = sha256(path / "receipt.json")
        options.append(dict(slice=key, **read_json(path / "selection.json")))
    stamp = dict(expected, slices=receipts)
    rows = checked_rows(pd.concat([query.assign(fold=axes(key)[0], role=axes(key)[1])
                      for key, query in queries.items()], ignore_index=True), None)
    by_condition = rows.groupby(CONDITION)
    if (rows.parent.nunique() != ref["parents"] or by_condition.ngroups != ref["conditions"]
            or not by_condition.rna_plate.nunique().eq(1).all()
            or not rows.groupby("inst_id").role.nunique().eq(3).all()):
        raise ValueError("T4 forecasts do not cover the full cohort with one plate and all DMSO rotations")
    if not scoring:
        write_json(args.freeze, stamp)
        return sha256(args.freeze)
    if read_json(args.freeze) != stamp:
        raise ValueError("T4 predictions changed after freeze")
    truth = np.concatenate([score_delta(args, ref, key, query, cols, genes) for key, query in queries.items()])
    result = paired_results(rows, truth, {arm: np.concatenate(values) for arm, values in arrays.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".t4-score-") as tmp:
        stage = Path(tmp)
        result.to_csv(stage / "paired.tsv", sep="\t", index=False, na_rep="NA")
        pd.DataFrame(options).to_csv(stage / "selection.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(stamp, freeze_sha256=args.expected_sha256,
            parents=rows.parent.nunique(), conditions=by_condition.ngroups, wells=rows.inst_id.nunique(),
            plates=rows.rna_plate.nunique(), files={f.name: sha256(f) for f in stage.iterdir()}))
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
    print((predict if args.action == "predict" else evaluate)(args))
