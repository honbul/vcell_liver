"""Transfer a fixed LINCS time operator across contexts with held parent folds."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.design import fold_id
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.t4_access import CELL, PHH_RANK, PHH_RIDGE, PREDICT
from src.r19_plan.t4_workflow import (average_conditions, axes, deltas, pairs, plans, reference,
                                      scalar, score_delta, validate_predict)
from src.r19_plan.time_operator import fit_time_operator
from src.r19_plan.x_inputs import check_unique, control_axis, matrix_axes


OLD = dict(identity6="identity6", local_scalar="scalar6", local_operator="operator6", direct24="direct24")
NEW = ("cross_scalar", "cross_operator")
CONTRASTS = tuple(("cross_operator", ref) for ref in (*OLD, "cross_scalar"))


def inputs(args):
    scopes = {}
    stamp = dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__), inputs={})
    for cell, root, prefix, protocol, access_pin in (
        (CELL, args.hepg2, "t4", "T4_PROTOCOL.md", "access.json"),
        ("PHH", args.phh, "t4_phh", "T4_PHH_PROTOCOL.md", "access/receipt.json"),
    ):
        public = args.reports / "data" / prefix
        scope = SimpleNamespace(access=root / "access", pin=public / access_pin, protocol=args.reports / protocol)
        ref = reference(scope)
        pin = public / "freeze.json"
        frozen = read_json(pin)
        if (ref.get("cell", CELL) != cell or sha256(root / "freeze.json") != sha256(pin)
                or frozen["access_sha256"] != sha256(scope.pin) or frozen["protocol_sha256"] != sha256(scope.protocol)
                or set(frozen["slices"]) != set(ref["slices"])):
            raise ValueError(f"The registered {cell} input and freeze differ")
        axis, cols = matrix_axes(scope.access, ref)
        scopes[cell] = SimpleNamespace(scope=scope, ref=ref, frozen=frozen, forecasts=root / "forecasts",
                                       genes=axis.pr_gene_id.to_numpy(), cols=cols)
        stamp["inputs"][cell] = dict(access_sha256=sha256(scope.pin), freeze_sha256=sha256(pin))
    a, b = scopes.values()
    if a.ref["matrix"] != b.ref["matrix"] or not np.array_equal(a.genes, b.genes) or not np.array_equal(a.cols, b.cols):
        raise ValueError("Cross-context T4 needs an identical matrix and gene order")
    return scopes[args.target], scopes["PHH" if args.target == CELL else CELL], stamp


def group_plans(cohort, group):
    data = {}
    for key in sorted(cohort.ref["slices"]):
        if key.startswith(group + "-"):
            data[key] = plans(cohort.scope, cohort.ref, key, "predict", PREDICT)
    if not data:
        raise ValueError(f"Unregistered fold and role: {group}")
    for key, value in data.items():
        validate_predict(value, key, cohort.ref.get("cell", CELL))
    return data


def original(cohort, key, query):
    path = cohort.forecasts / key
    receipt = read_json(path / "receipt.json")
    packed = path / "forecasts.npz"
    if (sha256(path / "receipt.json") != cohort.frozen["slices"][key]
            or sha256(packed) != receipt["files"][packed.name]
            or receipt["query_sha256"] != cohort.ref["slices"][key]["predict"]["query.tsv"]):
        raise ValueError("The original input vector or comparator differs from its freeze")
    with np.load(packed, allow_pickle=False) as file:
        return {arm: checked_array(file[name], (len(query), len(cohort.genes)), arm) for arm, name in OLD.items()}


def source_fit(source, source_plans, queries, target_plates, fold, role):
    earlier, later, controls = (
        pd.concat([group[name] for group in source_plans.values()], ignore_index=True).drop_duplicates()
        for name in ("source6", "source24", "fit_controls")
    )
    train = pd.concat((earlier, later), ignore_index=True)
    controls = controls.loc[controls.rna_plate.isin(train.rna_plate)]
    admitted = pd.concat((train, controls), ignore_index=True)
    check_unique(admitted)
    control_axis(controls, train, role)
    if (set(admitted.rna_plate) & target_plates or set(train.parent) & set(queries.parent)
            or train.parent.map(lambda p: fold_id(p, "R15-outer", 3)).eq(fold).any()
            or later.parent.nunique() <= PHH_RANK):
        raise ValueError("Insufficient independent source parent, plate or time-pair support")
    raw = extract_matrix(source.ref["matrix"], admitted, source.cols, source.genes)
    values = deltas(train, raw[:len(train)], controls, raw[len(train):])
    triplet = pairs(average_conditions(earlier, values[:len(earlier)]), later, values[len(earlier):])
    model = fit_time_operator(*triplet, source.genes, rank=PHH_RANK, ridge=PHH_RIDGE, held_parents=set(queries.parent))
    return model, scalar(*triplet), len(triplet[0]), train.rna_plate.nunique()


def predict(args):
    target, source, stamp = inputs(args)
    data = group_plans(target, args.group)
    fold, role, *_ = axes(next(iter(data)))
    query = pd.concat([x["query"] for x in data.values()], ignore_index=True)
    target_plates = {plate for group in data.values() for rows in group.values() for plate in rows.rna_plate}
    fit, gain, conditions, plates = source_fit(source, group_plans(source, args.group), query, target_plates, fold, role)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    with TemporaryDirectory(dir=args.out.parent, prefix=".t4-context-") as tmp:
        stage = Path(tmp)
        files = {}
        for key, table in data.items():
            early = original(target, key, table["query"])["identity6"]
            path = stage / f"{key}.npz"
            np.savez_compressed(path, cross_scalar=(gain * early).astype("float32"),
                                cross_operator=fit.predict(early, target.genes).astype("float32"))
            files[key] = sha256(path)
        write_json(stage / "receipt.json", dict(stamp, slices=files, target=args.target, group=args.group,
                   source_parents=fit.fit_parent_count, source_conditions=conditions, source_plates=plates, scalar=gain))
        stage.rename(args.out)


def metrics(rows, truth, arrays):
    energy = squared_rows(truth)
    errors = {arm: squared_rows(truth, array) for arm, array in arrays.items()}
    result = []
    for arm, ref in CONTRASTS:
        moments = parent_summaries(rows, truth.shape[1], energy, errors[arm], errors[ref])
        points, intervals, draws = paired_effects(*moments)
        rel_low, rel_high, _ = relative_interval(moments[0])
        for metric, (a, b), (lo, hi), n in zip(METRICS, points, intervals, draws, strict=True):
            result.append(dict(arm=arm, reference=ref, metric=metric, arm_value=a, reference_value=b,
                               difference=a-b, low99=lo, high99=hi, draws=n,
                               relative_gain=1-a/b if metric == "rmse" else np.nan,
                               relative_low99=rel_low if metric == "rmse" else np.nan,
                               relative_high99=rel_high if metric == "rmse" else np.nan))
    return pd.DataFrame(result)


def evaluate(args):
    scoring = args.action == "score"
    if scoring and (args.out.exists() or sha256(args.freeze) != args.digest):
        raise ValueError("Independent scoring requires the freeze digest and a new output path")
    if not scoring and args.freeze.exists():
        raise FileExistsError(args.freeze)
    target, _, stamp = inputs(args)
    groups = sorted({key[:4] for key in target.ref["slices"]})
    manifest = dict(stamp, target=args.target, groups={})
    queries, settings, arrays = {}, [], {arm: [] for arm in (*OLD, *NEW)}
    for group in groups:
        path = args.forecasts / group
        receipt = read_json(path / "receipt.json")
        keys = {key for key in target.ref["slices"] if key.startswith(group + "-")}
        if (any(receipt[k] != v for k, v in stamp.items()) or receipt["target"] != args.target
                or receipt["group"] != group or set(receipt["slices"]) != keys):
            raise ValueError("Unexpected context-transfer forecast inventory")
        manifest["groups"][group] = sha256(path / "receipt.json")
        settings.append({k: receipt[k] for k in ("group", "source_parents", "source_conditions", "source_plates", "scalar")})
        for key, digest in sorted(receipt["slices"].items()):
            query = queries[key] = plans(target.scope, target.ref, key, "predict", PREDICT)["query"]
            previous = original(target, key, query)
            packed = path / f"{key}.npz"
            if sha256(packed) != digest:
                raise ValueError("The candidate differs from its prediction receipt")
            with np.load(packed, allow_pickle=False) as file:
                if set(file.files) != set(NEW):
                    raise ValueError("The forecast changed its registered arms")
                current = {arm: checked_array(file[arm], previous["identity6"].shape, arm) for arm in NEW}
            if scoring:
                for arm, matrix in {**previous, **current}.items():
                    arrays[arm].append(matrix)
    rows = checked_rows(pd.concat([tab.assign(fold=axes(k)[0], role=axes(k)[1]) for k, tab in queries.items()],
                                  ignore_index=True), None)
    if rows.parent.nunique() != target.ref["parents"] or not rows.groupby("inst_id").role.nunique().eq(3).all():
        raise ValueError("The context transfer lost registered parents or role rotations")
    if not scoring:
        write_json(args.freeze, manifest)
        return
    if read_json(args.freeze) != manifest:
        raise ValueError("Candidate inputs differ from the independent forecast freeze")
    truth = np.concatenate([score_delta(target.scope, target.ref, k, q, target.cols, target.genes) for k, q in queries.items()])
    report = metrics(rows, truth, {arm: np.concatenate(parts) for arm, parts in arrays.items()})
    with TemporaryDirectory(dir=args.out.parent, prefix=".t4-context-score-") as tmp:
        stage = Path(tmp)
        report.to_csv(stage / "paired.tsv", sep="\t", index=False, na_rep="NA")
        pd.DataFrame(settings).to_csv(stage / "fitting.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(manifest, freeze_sha256=args.digest, parents=rows.parent.nunique(),
                   wells=rows.inst_id.nunique(), plates=rows.rna_plate.nunique(), files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("predict", "freeze", "score"))
    cli.add_argument("--target", choices=(CELL, "PHH"), required=True)
    for field in ("reports", "hepg2", "phh", "protocol", "forecasts", "freeze", "out"):
        cli.add_argument("--" + field, type=Path)
    cli.add_argument("--group")
    cli.add_argument("--digest")
    args = cli.parse_args()
    (predict if args.action == "predict" else evaluate)(args)
