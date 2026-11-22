"""Isolate source offsets and test anchored time residuals on the locked T4 targets."""

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
from src.r19_plan.t4_access import CELL, PHH_RANK, PHH_RIDGE, PREDICT
from src.r19_plan.t4_context_transfer import group_plans, inputs, original
from src.r19_plan.t4_scaffold_transfer import fit_effects
from src.r19_plan.t4_workflow import average_conditions, axes, pairs, plans, scalar, score_delta
from src.r19_plan.time_operator import fit_time_operator


PRIOR = ("cross_scalar", "cross_operator")
NEW = ("scalar_with_offset", "operator_no_offset", "residual_only", "residual_with_offset")
COMPARISONS = (
    ("residual_only", "cross_scalar"), ("residual_only", "cross_operator"), ("residual_only", "local_scalar"),
    ("cross_operator", "operator_no_offset"), ("scalar_with_offset", "cross_scalar"),
    ("residual_with_offset", "residual_only"), ("operator_no_offset", "cross_scalar"),
)


def lineage(args):
    prior_args = SimpleNamespace(**vars(args))
    prior_args.protocol = args.reports / "T4_CONTEXT_TRANSFER_PROTOCOL.md"
    target, source, source_stamp = inputs(prior_args)
    root = args.previous / args.target
    pin = args.reports / "data/t4_context_transfer" / args.target / "freeze.json"
    frozen = read_json(pin)
    if (sha256(root / "freeze.json") != sha256(pin) or frozen["target"] != args.target
            or any(frozen[key] != value for key, value in source_stamp.items())):
        raise ValueError("The original context transfer differs from its published freeze")
    return target, source, root, frozen, dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
               upstream_freeze_sha256=sha256(pin), target=args.target, access=source_stamp["inputs"])


def previous(root, frozen, group, key, shape):
    path = root / "forecasts" / group
    receipt = read_json(path / "receipt.json")
    packed = path / f"{key}.npz"
    if sha256(path / "receipt.json") != frozen["groups"][group] or sha256(packed) != receipt["slices"][key]:
        raise ValueError("The registered context forecast changed")
    with np.load(packed, allow_pickle=False) as file:
        if set(file.files) != set(PRIOR):
            raise ValueError("The original context experiment has an unexpected arm")
        return {name: checked_array(file[name], shape, name) for name in PRIOR}


def source_pairs(source, data, target, target_data, fold, role):
    earlier, later = (pd.concat([item[name] for item in data.values()], ignore_index=True).drop_duplicates()
                      for name in ("source6", "source24"))
    query = pd.concat([item["query"] for item in target_data.values()], ignore_index=True)
    active = pd.concat((earlier, later), ignore_index=True)
    held_plates = {p for item in target_data.values() for rows in item.values() for p in rows.rna_plate}
    if (source.ref["matrix"] != target.ref["matrix"] or set(active.rna_plate) & held_plates
            or set(active.parent) & set(query.parent)
            or active.parent.map(lambda p: fold_id(p, "R15-outer", 3)).eq(fold).any()):
        raise ValueError("Source pairs intersect target plates or the held parent fold")
    index, values = fit_effects(source, data, role)
    source6 = average_conditions(earlier, values[index.get_indexer(earlier.inst_id)])
    return pairs(source6, later, values[index.get_indexer(later.inst_id)]), set(query.parent)


def model_arms(x, gain, linear, learned, genes):
    baseline = gain * x
    offset = linear.output_mean - linear.input_mean @ linear.weights
    correction = x @ learned.weights
    forecasts = dict(scalar_with_offset=baseline + offset, operator_no_offset=x @ linear.weights,
                     residual_only=baseline + correction,
                     residual_with_offset=baseline + learned.predict(x, genes))
    return forecasts, dict(cross_scalar=baseline, cross_operator=linear.predict(x, genes))


def predict(args):
    target, source, previous_root, frozen, stamp = lineage(args)
    destination = args.forecasts / args.group
    if destination.exists():
        raise FileExistsError(destination)
    target_data = group_plans(target, args.group)
    fold, role, *_ = axes(next(iter(target_data)))
    triplet, held = source_pairs(source, group_plans(source, args.group), target, target_data, fold, role)
    rows, early, later = triplet
    gain = scalar(*triplet)
    fit = dict(rank=PHH_RANK, ridge=PHH_RIDGE, held_parents=held)
    linear = fit_time_operator(*triplet, source.genes, **fit)
    learned = fit_time_operator(rows, early, later - gain * early, source.genes, **fit)
    args.forecasts.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.forecasts, prefix=".t4-residual-") as tmp:
        stage = Path(tmp)
        files, max_error = {}, 0.
        for key, plan in target_data.items():
            x = original(target, key, plan["query"])["identity6"]
            values, replay = model_arms(x, gain, linear, learned, source.genes)
            held_values = previous(previous_root, frozen, args.group, key, x.shape)
            max_error = max(max_error, *(float(np.max(np.abs(replay[name].astype("float32") - held_values[name])))
                                         for name in PRIOR))
            if max_error > 1e-6:
                raise ValueError(f"Original context forecast replay differs by {max_error:g}")
            out = stage / f"{key}.npz"
            np.savez_compressed(out, **{name: pred.astype("float32") for name, pred in values.items()})
            files[key] = sha256(out)
        write_json(stage / "receipt.json", dict(stamp, group=args.group, slices=files,
                   max_replay_error=max_error, source_parents=linear.fit_parent_count,
                   source_conditions=len(rows), source_scalar=gain))
        stage.rename(destination)


def metric_table(rows, truth, forecasts):
    results = []
    energy = squared_rows(truth)
    errors = {name: squared_rows(truth, value) for name, value in forecasts.items()}
    for arm, reference in COMPARISONS:
        moments = parent_summaries(rows, truth.shape[1], energy, errors[arm], errors[reference])
        points, ci, draws = paired_effects(*moments)
        low, high, _ = relative_interval(moments[0])
        for metric, (a, b), (lo, hi), n in zip(METRICS, points, ci, draws, strict=True):
            raw_rmse = metric == "rmse"
            results.append(dict(arm=arm, reference=reference, metric=metric, arm_value=a, reference_value=b,
                difference=a-b, low99=lo, high99=hi, draws=n, relative_gain=1-a/b if raw_rmse else np.nan,
                relative_low99=low if raw_rmse else np.nan, relative_high99=high if raw_rmse else np.nan))
    return pd.DataFrame(results)


def evaluate(args):
    scoring = args.action == "score"
    if scoring and (args.out.exists() or sha256(args.freeze) != args.digest):
        raise ValueError("Residual scoring requires the separate freeze hash")
    if not scoring and args.freeze.exists():
        raise FileExistsError(args.freeze)
    target, _, previous_root, old, stamp = lineage(args)
    groups = sorted({key[:4] for key in target.ref["slices"]})
    queries, digests, fit_support = {}, {}, []
    forecasts = {name: [] for name in (*PRIOR, *NEW, "local_scalar")}
    for group in groups:
        path = args.forecasts / group
        receipt = read_json(path / "receipt.json")
        keys = {key for key in target.ref["slices"] if key.startswith(group + "-")}
        if (any(receipt[k] != value for k, value in stamp.items()) or receipt["group"] != group
                or set(receipt["slices"]) != keys or receipt["max_replay_error"] > 1e-6):
            raise ValueError("A residual forecast differs from its registered group or lineage")
        digests[group] = sha256(path / "receipt.json")
        fit_support.append({k: receipt[k] for k in ("group", "max_replay_error", "source_parents", "source_conditions", "source_scalar")})
        for key, digest in sorted(receipt["slices"].items()):
            query = queries[key] = plans(target.scope, target.ref, key, "predict", PREDICT)["query"]
            shape = (len(query), len(target.genes))
            previous_values = previous(previous_root, old, group, key, shape)
            path_npz = path / f"{key}.npz"
            if sha256(path_npz) != digest:
                raise ValueError("The residual forecast payload changed")
            with np.load(path_npz, allow_pickle=False) as packed:
                if set(packed.files) != set(NEW):
                    raise ValueError("Unexpected model inventory in the locked residual forecast")
                new = {name: checked_array(packed[name], shape, name) for name in NEW}
            if scoring:
                local = original(target, key, query)["local_scalar"]
                for name, value in (previous_values | new | {"local_scalar": local}).items():
                    forecasts[name].append(value)
    rows = checked_rows(pd.concat([q.assign(fold=axes(k)[0], role=axes(k)[1]) for k, q in queries.items()], ignore_index=True), None)
    if (rows.parent.nunique() != target.ref["parents"] or rows.inst_id.nunique() != {"PHH": 105, CELL: 268}[args.target]
            or not rows.groupby("inst_id").role.nunique().eq(3).all()):
        raise ValueError("At least one registered target parent or physical well rotation is missing")
    manifest = dict(stamp, groups=digests)
    if not scoring:
        write_json(args.freeze, manifest)
        return
    if read_json(args.freeze) != manifest:
        raise ValueError("Locked residual predictions changed")
    truth = np.concatenate([score_delta(target.scope, target.ref, key, query, target.cols, target.genes) for key, query in queries.items()])
    scores = metric_table(rows, truth, {name: np.concatenate(parts) for name, parts in forecasts.items()})
    with TemporaryDirectory(dir=args.out.parent, prefix=".t4-residual-score-") as tmp:
        stage = Path(tmp)
        scores.to_csv(stage / "paired.tsv", sep="\t", index=False, na_rep="NA")
        pd.DataFrame(fit_support).to_csv(stage / "fit_support.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(manifest, freeze_sha256=args.digest, parents=rows.parent.nunique(),
            wells=rows.inst_id.nunique(), plates=rows.rna_plate.nunique(), files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("predict", "freeze", "score"))
    cli.add_argument("--target", choices=(CELL, "PHH"), required=True)
    for field in ("reports", "hepg2", "phh", "previous", "protocol", "forecasts", "freeze", "out"):
        cli.add_argument("--" + field, type=Path)
    cli.add_argument("--group")
    cli.add_argument("--digest")
    args = cli.parse_args()
    (predict if args.action == "predict" else evaluate)(args)
