"""Freeze matched D1/D2 predictions before reading score-role controls and outcomes."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.forecast import LEGACY
from src.r19_plan.d1_inputs import access_reference, physical_subset
from src.r19_plan.d12_predict import POLICIES
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_score import METRICS, PHYSICAL, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.i_stage import (
    SCORE, ExpressionCache, check_role, check_wells, checked_query, parse_tsv, pinned_bytes, plan_receipt,
    read_tables, slice_roles,
)
from src.r19_plan.i_workflow import registered_axes
from src.r19_plan.r18_input_bundle import extract_matrix, indices
from src.r19_plan.r18_predict import vehicle_means

PAIRS = {"D1_independent": "D1_shared", "D2_joint": "D2_legacy"}
ARMS = set(POLICIES) | {"B_pos"}
FILES = {"rows.tsv", "landmarks.tsv", "forecasts.npz", "selection.json"}


def read_predictions(root, model, slice_id, query, genes, access_sha, protocol_sha):
    path = root / model / slice_id
    receipt = path / "receipt.json"
    ref = read_json(receipt)
    if (ref["slice"] != slice_id or ref["model"] != model or ref["protocol_sha256"] != protocol_sha
            or ref["inputs"]["access"] != access_sha or set(ref["arms"]) != ARMS or set(ref["files"]) != FILES
            or ref["code_sha256"] != sha256(Path(__file__).with_name("d12_predict.py"))):
        raise ValueError("D1/D2 forecast receipt changed its registered slice, model, arm, or input")
    for name, digest in ref["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"A D1/D2 prediction artifact changed: {model}/{slice_id}/{name}")
    if ((path / "rows.tsv").read_bytes() != query.to_csv(sep="\t", index=False).encode()
            or (path / "landmarks.tsv").read_bytes() != genes.to_csv(sep="\t", index=False).encode()):
        raise ValueError("D1/D2 frozen well or gene axes differ from the original approved query")
    with np.load(path / "forecasts.npz", allow_pickle=False) as data:
        if set(data.files) != ARMS:
            raise ValueError("D1/D2 forecast arrays differ from the registered arms")
        arrays = {name: checked_array(data[name], (len(query), len(genes)), name) for name in ARMS}
    if not np.array_equal(arrays["D2_joint"], arrays["D1_independent"]):
        raise ValueError("D1/D2 equal-input joint choices must produce identical forecasts")
    return sha256(receipt), arrays


def score_truth(args, original, balanced, raw_queries, genes):
    expected = {name + ".tsv" for name in SCORE}
    cache = ExpressionCache(args.cache, args.vehicles, original)
    axis = parse_tsv(pinned_bytes(args.access / "gene_indices.tsv", balanced["files"]["gene_indices.tsv"]))
    identifiers = genes.pr_gene_id.to_numpy()
    if not cache.landmarks.equals(genes) or not np.array_equal(indices(axis.gene_id), identifiers):
        raise ValueError("D1/D2 score landmark order changed")
    result = []
    for slice_id, query in raw_queries.items():
        files = balanced["slice_files"][slice_id]["score"]
        if set(files) != expected:
            raise ValueError("D1/D2 has an incomplete score-only plan")
        path = args.access / "score-plan" / slice_id
        tables = {name: parse_tsv(pinned_bytes(path / f"{name}.tsv", files[f"{name}.tsv"])) for name in SCORE}
        labels, controls = tables["score_labels"], tables["score_controls"]
        old, _ = read_tables(args.original_plan, original, slice_id, "score", SCORE)
        if not labels.equals(old["score_labels"]) or not checked_query(labels.drop(columns="cache_index")).equals(query):
            raise ValueError("D1/D2 score treatment rows differ from the original approved scope")
        physical_subset(controls, old["score_controls"])
        check_role(controls, labels, slice_roles(slice_id)[-1])
        check_wells(pd.concat((labels, controls), ignore_index=True))
        target = cache.take(cache.cache / "target_wells.npy", (cache.target_count, len(genes)), cache.target_indices(labels))
        values = extract_matrix(balanced["matrix"], controls, indices(axis.level3_gene_index), identifiers)
        means = vehicle_means(controls, values, "rna_plate")
        result.append(target - np.stack([means[plate] for plate in labels.rna_plate]))
    return np.concatenate(result)


def metrics(rows, truth, predictions, fit):
    rows = checked_rows(rows, fit)
    shape = len(rows), truth.shape[1]
    exposure = squared_rows(checked_array(truth, shape, "D1/D2 score labels"))
    effects = []
    for model, arrays in predictions.items():
        errors = {arm: squared_rows(truth, checked_array(values, shape, arm)) for arm, values in arrays.items()}
        contrasts = list(PAIRS.items()) + [(arm, "B_pos") for arm in POLICIES]
        for arm, other in contrasts:
            summary = parent_summaries(rows, shape[1], exposure, errors[arm], errors[other])
            points, intervals, valid = paired_effects(*summary)
            for metric, (a, b), (low, high), draws in zip(METRICS, points, intervals, valid, strict=True):
                effects.append(dict(model=model, arm=arm, reference=other, metric=metric, arm_value=a,
                    reference_value=b, difference=a - b, low99=low, high99=high, valid_bootstrap_draws=int(draws),
                    parents=rows.parent.nunique(), role_rows=len(rows), physical_wells=len(rows.drop_duplicates(PHYSICAL))))
    return pd.DataFrame(effects)


def run(args):
    if args.action == "score" and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("D1/D2 scoring requires the immutable freeze digest and a fresh output path")
    access_sha, protocol_sha = sha256(args.access_reference), sha256(args.protocol)
    balanced = access_reference(args.access_reference, "f0-A")
    original, digest = plan_receipt(args.original_plan, args.original_reference, "f0-A")
    if digest != balanced["source_access_sha256"] or sha256(args.access / "receipt.json") != access_sha:
        raise ValueError("D1/D2 balanced role lock differs from the original approved development split")
    raw, queries, genes, fit = registered_axes(args.original_plan, args.original_reference, args.cache)
    if not args.models or len(set(args.models)) != len(args.models):
        raise ValueError("D1/D2 requires a unique list of registered comparator models")
    all_predictions, receipts = {}, {}
    for model in args.models:
        forecasts = {arm: [] for arm in ARMS}
        receipts[model] = {}
        for (fold, role), query in queries.items():
            slice_id = f"f{fold}-{role}"
            digest, arrays = read_predictions(args.forecasts, model, slice_id, query, genes, access_sha, protocol_sha)
            receipts[model][slice_id] = digest
            for arm in ARMS:
                forecasts[arm].append(arrays[arm])
        if args.action == "score":
            all_predictions[model] = {name: np.concatenate(values) for name, values in forecasts.items()}
    module = Path(__file__)
    lock = dict(protocol_sha256=protocol_sha, access_sha256=access_sha, folds=receipts, models=args.models,
                code_sha256={name: sha256(module.with_name(name)) for name in
                            (module.name, "d12_predict.py", "d1_inputs.py", "dev_selection.py", "i_score.py")})
    if args.action == "freeze":
        write_json(args.freeze, lock)
        return sha256(args.freeze)
    if lock != read_json(args.freeze):
        raise ValueError("D1/D2 changed a registered prediction, model, or code after the freeze")
    truths = score_truth(args, original, balanced, raw, genes)
    score = metrics(pd.concat(queries.values(), ignore_index=True), truths, all_predictions, fit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".d12-score-") as tmp:
        stage = Path(tmp)
        score.to_csv(stage / "paired.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(lock, freeze_sha256=args.expected_sha256,
                                                  files={"paired.tsv": sha256(stage / "paired.tsv")}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    parser.add_argument("--models", nargs="+", choices=LEGACY, required=True)
    for name in ("access", "access-reference", "original-plan", "original-reference", "cache", "protocol",
                 "forecasts", "freeze"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("vehicles", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.action == "score" and any(value is None for value in (args.out, args.expected_sha256, args.vehicles)):
        parser.error("score requires --out, --expected-sha256, and --vehicles")
    print(run(args))
