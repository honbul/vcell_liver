"""Compare the frozen early operator with independently calibrated late repeats."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.t4_access import CELL, CONDITION, PREDICT
from src.r19_plan.t4_workflow import axes, deltas, plans, reference, scalar, score_delta, validate_predict
from src.r19_plan.time_operator import fit_time_operator
from src.r19_plan.x_inputs import check_unique, control_axis, matrix_axes


EXISTING = ("operator6", "scalar6", "direct24")
ADDED = ("direct24_scalar", "direct24_rank16")
ARMS = (*EXISTING, *ADDED)
PAIRS = (("operator6", "direct24"), *(("operator6", a) for a in ADDED),
         *((a, "direct24") for a in ADDED), ("direct24_rank16", "direct24_scalar"))


def lineage(args):
    scope = SimpleNamespace(access=args.access, pin=args.pin, protocol=args.original_protocol)
    ref, frozen = reference(scope), read_json(args.original)
    if (sha256(args.original) != sha256(args.published) or frozen["access_sha256"] != sha256(args.pin)
            or frozen["protocol_sha256"] != sha256(args.original_protocol) or set(frozen["slices"]) != set(ref["slices"])):
        raise ValueError("The original T4 predictions changed from the published freeze")
    return ref, frozen, dict(protocol_sha256=sha256(args.protocol), original_sha256=sha256(args.original),
        access_sha256=sha256(args.pin), code_sha256=sha256(__file__))


def fit_pairs(rows, values):
    cases, features, labels = [], [], []
    for (parent, dose), ids in rows.groupby(CONDITION, sort=True).indices.items():
        plates = rows.rna_plate.iloc[ids]
        if plates.nunique() < 2:
            continue
        selected = min(plates.unique(), key=lambda p: (crc32(f"cal24|{parent}|{dose}|{p}".encode()), p))
        score = plates.eq(selected).to_numpy()
        cases.append(dict(parent=parent, dose_uM=dose, study=rows.study.iloc[ids[0]], cell_id=rows.cell_id.iloc[ids[0]]))
        features.append(values[ids[~score]].mean(axis=0))
        labels.append(values[ids[score]].mean(axis=0))
    data = pd.DataFrame(cases)
    if len(data) == 0 or data.parent.nunique() < 5:
        raise ValueError("Fewer than five independent training parents have two late RNA plates")
    return data, np.stack(features), np.stack(labels)


def original_arrays(args, key, frozen, query, genes):
    folder = args.original_forecasts/key
    receipt = read_json(folder/"receipt.json")
    packed = folder/"forecasts.npz"
    if sha256(folder/"receipt.json") != frozen["slices"][key] or sha256(packed) != receipt["files"]["forecasts.npz"]:
        raise ValueError("An original time forecast has a different hash")
    with np.load(packed, allow_pickle=False) as file:
        return {a: checked_array(file[a], (len(query), len(genes)), a) for a in EXISTING}


def predict(args):
    ref, frozen, stamp = lineage(args)
    cell = ref.get("cell", CELL)
    genes_frame, columns = matrix_axes(args.access, ref)
    genes = genes_frame.pr_gene_id.to_numpy()
    keys = sorted(key for key in ref["slices"] if key.startswith(args.group+"-"))
    if not keys or args.out.exists():
        raise ValueError("The repeat calibration requires a new registered fold-rotation group")
    data = {k: plans(args, ref, k, "predict", PREDICT) for k in keys}
    for k in keys:
        validate_predict(data[k], k, cell)
    train = pd.concat([d["source24"] for d in data.values()], ignore_index=True).drop_duplicates()
    controls = pd.concat([d["fit_controls"] for d in data.values()], ignore_index=True).drop_duplicates()
    controls = controls.loc[controls.rna_plate.isin(train.rna_plate)]
    control_axis(controls, train, axes(keys[0])[1])
    check_unique(pd.concat((train, controls), ignore_index=True))
    raw = extract_matrix(ref["matrix"], pd.concat((train, controls), ignore_index=True), columns, genes)
    train_values = deltas(train, raw[:len(train)], controls, raw[len(train):])
    index = pd.Index(train.inst_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".repeat-calibration-") as tmp:
        stage = Path(tmp)
        records, settings = {}, []
        for key, plan in data.items():
            fit, query = plan["source24"], plan["query"]
            if fit.rna_plate.eq(axes(key)[4]).any() or set(fit.parent) & set(query.parent):
                raise ValueError("A held parent or late plate entered repeat calibration")
            fit_rows, inputs, labels = fit_pairs(fit, train_values[index.get_indexer(fit.inst_id)])
            old = original_arrays(args, key, frozen, query, genes)
            gain = scalar(fit_rows, inputs, labels)
            simple = gain * old["direct24"]
            model = None
            if fit_rows.parent.nunique() >= 17:
                try:
                    model = fit_time_operator(fit_rows, inputs, labels, genes, rank=16, ridge=1., held_parents=set(query.parent))
                except ValueError as exc:
                    if not str(exc).startswith("Insufficient numerical support"):
                        raise
            complex_arm = simple if model is None else model.predict(old["direct24"], genes)
            path = stage/(key+".npz")
            np.savez_compressed(path, **old, direct24_scalar=simple.astype("float32"), direct24_rank16=complex_arm.astype("float32"))
            records[key] = sha256(path)
            settings.append(dict(slice=key, fitting_parents=fit_rows.parent.nunique(), fitting_conditions=len(fit_rows),
                                 scalar_gain=gain, rank16_fit=model is not None))
        write_json(stage/"receipt.json", dict(stamp, slices=records, settings=settings))
        stage.rename(args.out)


def summary(rows, truth, forecasts):
    records = []
    energy = squared_rows(truth)
    errors = {name: squared_rows(truth, value) for name, value in forecasts.items()}
    for arm, reference_name in PAIRS:
        moments = parent_summaries(rows, truth.shape[1], energy, errors[arm], errors[reference_name])
        points, intervals, draws = paired_effects(*moments)
        rel_low, rel_high, _ = relative_interval(moments[0])
        for metric, (a, b), (lo, hi), n in zip(METRICS, points, intervals, draws, strict=True):
            records.append(dict(arm=arm, reference=reference_name, metric=metric, arm_value=a, reference_value=b,
                difference=a-b, low99=lo, high99=hi, valid_draws=n,
                relative_gain=1-a/b if metric == "rmse" else np.nan,
                relative_low99=rel_low if metric == "rmse" else np.nan,
                relative_high99=rel_high if metric == "rmse" else np.nan))
    return pd.DataFrame(records)


def evaluate(args):
    scoring = args.action == "score"
    if scoring and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("The repeat comparison requires the independent freeze digest")
    if not scoring and args.freeze.exists():
        raise FileExistsError(args.freeze)
    ref, old, stamp = lineage(args)
    frame, columns = matrix_axes(args.access, ref)
    genes = frame.pr_gene_id.to_numpy()
    groups = sorted({"-".join(k.split("-", 2)[:2]) for k in ref["slices"]})
    queries, hashes, selections, forecasts = {}, {}, [], {arm: [] for arm in ARMS}
    for group in groups:
        path = args.forecasts/group
        receipt = read_json(path/"receipt.json")
        required = {k for k in ref["slices"] if k.startswith(group+"-")}
        if any(receipt[key] != h for key, h in stamp.items()) or set(receipt["slices"]) != required:
            raise ValueError("A repeat prediction group is incomplete or changed since fitting")
        hashes[group] = sha256(path/"receipt.json")
        selections.extend(receipt["settings"])
        for key, digest in sorted(receipt["slices"].items()):
            packed = path/(key+".npz")
            if sha256(packed) != digest:
                raise ValueError("A repeat prediction array changed after fitting")
            query = queries[key] = plans(args, ref, key, "predict", PREDICT)["query"]
            originals = original_arrays(args, key, old, query, genes)
            with np.load(packed, allow_pickle=False) as values:
                if set(values.files) != set(ARMS):
                    raise ValueError("The repeat model inventory changed")
                for arm in ARMS:
                    matrix = checked_array(values[arm], (len(query), len(genes)), arm)
                    if arm in originals and not np.array_equal(matrix, originals[arm]):
                        raise ValueError("The original comparison forecast changed")
                    if scoring:
                        forecasts[arm].append(matrix)
    rows = checked_rows(pd.concat([q.assign(fold=axes(k)[0], role=axes(k)[1]) for k, q in queries.items()], ignore_index=True), None)
    if rows.parent.nunique() != ref["parents"] or not rows.groupby("inst_id").role.nunique().eq(3).all():
        raise ValueError("The repeat arms lost a parent or rotation from the original scope")
    freeze = dict(stamp, groups=hashes)
    if not scoring:
        write_json(args.freeze, freeze)
        return
    if read_json(args.freeze) != freeze:
        raise ValueError("The fit plans or predictions differ from the frozen record")
    truth = np.concatenate([score_delta(args, ref, key, query, columns, genes) for key, query in queries.items()])
    scores = summary(rows, truth, {arm: np.concatenate(parts) for arm, parts in forecasts.items()})
    with TemporaryDirectory(dir=args.out.parent, prefix=".repeat-calibration-score-") as tmp:
        stage = Path(tmp)
        scores.to_csv(stage/"paired.tsv", sep="\t", index=False, na_rep="NA")
        pd.DataFrame(selections).to_csv(stage/"selection.tsv", sep="\t", index=False)
        write_json(stage/"receipt.json", dict(freeze, freeze_sha256=args.expected_sha256, parents=rows.parent.nunique(),
            wells=rows.inst_id.nunique(), files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "freeze", "score"))
    for name in ("protocol", "original-protocol", "access", "pin", "original", "published", "original-forecasts",
                 "forecasts", "freeze", "out"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--group")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    (predict if args.action == "predict" else evaluate)(args)
