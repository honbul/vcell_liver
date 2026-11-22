"""Freeze all I5 prediction slices and score them in an independent process."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.goal_tracks.t2_predict import R12_EPOCHS, R12_LRS
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.estimators import GAINS
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i5_run import MODE_ARMS, SOURCE_HASH, VIEWS, source_slice
from src.r19_plan.i5_score import ARMS, i5_cohorts, score_i5
from src.r19_plan.i_stage import load_score_delta
from src.r19_plan.i_workflow import registered_axes

SUPPORT = {"both_phenotypes": 345, "prism_available": 722, "whole_fallback": 1862}
RUN_FILES = {"predictions.npz", "coverage.npz"}


def registered_predictions(args, queries, genes):
    forecasts = {arm: [] for arm in ARMS}
    availability = {view: [] for view in VIEWS}
    receipts = {}
    for (fold, role), query in queries.items():
        slice_id = f"f{fold}-{role}"
        _, _, original, arrays, masks = source_slice(args.source, args.source_protocol, slice_id, query, genes)
        path = args.forecasts / slice_id
        receipt = path / "receipt.json"
        stamp = read_json(receipt)
        if (stamp["slice"] != slice_id or stamp["protocol_sha256"] != sha256(args.protocol)
                or stamp["source_slice_sha256"] != original or set(stamp["files"]) != RUN_FILES
                or stamp["inputs"]["access"] != sha256(args.split_reference)
                or not 0 <= stamp["reconstruction_max_abs"] <= 1e-6
                or set(stamp["choices"]) != set(MODE_ARMS.values())):
            raise ValueError(f"I5 forecast lineage or axes changed in {slice_id}")
        for choice in stamp["choices"].values():
            if (choice["lr"] not in R12_LRS or choice["epoch"] not in R12_EPOCHS
                    or choice["gain"] not in GAINS or not np.isfinite(choice["loss"])):
                raise ValueError(f"I5 checkpoint or gain outside the registered grid in {slice_id}")
        for filename, digest in stamp["files"].items():
            if sha256(path / filename) != digest:
                raise ValueError(f"I5 forecast changed after writing: {slice_id}/{filename}")
        with np.load(path / "coverage.npz", allow_pickle=False) as data:
            if set(data.files) != set(VIEWS):
                raise ValueError(f"Unexpected I5 feature views in {slice_id}")
            for index, view in enumerate(VIEWS):
                array = data[view]
                if array.dtype.kind != "b" or not np.array_equal(array, masks[:, index]):
                    raise ValueError(f"I5 availability changed in {slice_id}/{view}")
                availability[view].append(array)
        with np.load(path / "predictions.npz", allow_pickle=False) as data:
            if set(data.files) != set(MODE_ARMS.values()):
                raise ValueError(f"Unexpected I5 prediction arms in {slice_id}")
            absent = ~masks.any(axis=1)
            for arm in data.files:
                predicted = checked_array(data[arm], (len(query), len(genes)), arm)
                if not np.array_equal(predicted[absent], arrays["I5_kernel"][absent]):
                    raise ValueError(f"I5 no-view fallback changed in {slice_id}/{arm}")
                arrays[arm] = predicted
        for arm in forecasts:
            forecasts[arm].append(arrays[arm])
        receipts[slice_id] = sha256(receipt)
    rows = pd.concat(queries.values(), ignore_index=True)
    available = {view: np.concatenate(parts) for view, parts in availability.items()}
    cohorts = i5_cohorts(rows, available)
    for scope, n in SUPPORT.items():
        if rows.loc[cohorts[scope], "parent"].nunique() != n:
            raise ValueError(f"I5 did not preserve the registered parent support: {scope}")
    return rows, forecasts, available, receipts


def run(args):
    if args.action == "score" and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("I5 scoring requires a fresh output path and the independent freeze digest")
    raw, queries, genes, fit = registered_axes(args.plan, args.split_reference, args.cache)
    rows, forecasts, available, receipts = registered_predictions(args, queries, genes)
    code = Path(__file__)
    stamp = dict(source_freeze_sha256=SOURCE_HASH, protocol_sha256=sha256(args.protocol),
                 split_sha256=sha256(args.split_reference), folds=receipts,
                 code_sha256={name: sha256(code.with_name(name)) for name in
                              ("i5_run.py", "i5_neural.py", "i5_score.py", code.name)})
    if args.action == "freeze":
        write_json(args.freeze, stamp)
        return sha256(args.freeze)
    if read_json(args.freeze) != stamp:
        raise ValueError("I5 score compared an altered prediction or code manifest")
    deltas = [load_score_delta(args.cache, args.vehicles, args.plan, args.split_reference, slice_id, query, genes)
              for slice_id, query in raw.items()]
    score = score_i5(rows, np.concatenate(deltas), {arm: np.concatenate(parts) for arm, parts in forecasts.items()},
                     available, expected_support=SUPPORT, fit_parents_by_fold=fit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".i5-score-") as tmp:
        stage = Path(tmp)
        score.to_csv(stage / "paired.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(stamp, freeze_sha256=args.expected_sha256,
                                                 files={"paired.tsv": sha256(stage / "paired.tsv")}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    for name in ("plan", "split-reference", "cache", "forecasts", "source", "source-protocol", "protocol", "freeze"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("vehicles", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.action == "score" and any(value is None for value in (args.vehicles, args.out, args.expected_sha256)):
        parser.error("score requires --vehicles, --out, and --expected-sha256")
    print(run(args))
