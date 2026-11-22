"""Apply the prescore I-track operational readout to the original frozen forecasts."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.i_evaluate import JoinedPredictions, frozen_registry, merged_coverage, open_forecasts
from src.r19_plan.i_freeze import verify_run
from src.r19_plan.i_operational import operational_points
from src.r19_plan.i_score import checked_array, checked_rows, coverage_arrays
from src.r19_plan.i_stage import load_score_delta
from src.r19_plan.i_workflow import registered_axes


def run(args):
    run_root = args.run
    forecast_path = run_root / "forecasts"
    frozen_digest = (run_root / "freeze.sha256").read_text().strip()
    raw, queries, genes, fit = registered_axes(args.plan, args.split_reference, args.cache)
    freeze = verify_run(run_root / "freeze.json", frozen_digest, forecast_path, queries, genes, args.protocol)
    score = run_root / "score"
    score_receipt = read_json(score / "receipt.json")
    if score_receipt["freeze_sha256"] != frozen_digest or score_receipt["protocol_sha256"] != sha256(args.protocol):
        raise ValueError("The operational supplement is not attached to the registered primary score")
    for name, digest in score_receipt["files"].items():
        if sha256(score / name) != digest:
            raise ValueError(f"The original score artifact changed: {name}")

    order = sorted(queries)
    slices = [f"f{fold}-{role}" for fold, role in order]
    paths = [forecast_path / slice_id for slice_id in slices]
    lengths = [len(queries[key]) for key in order]
    rows = checked_rows(pd.concat([queries[key] for key in order], ignore_index=True), fit)
    registry = frozen_registry(paths, freeze["folds"])
    arms = set(registry) | {"B_R17"}
    with ExitStack() as stack:
        full = open_forecasts(stack, paths, "forecasts.npz", arms)
        counts = open_forecasts(stack, paths, "coverage.npz", set(registry))
        coverage = merged_coverage(counts, registry, lengths)
        coverage_arrays(rows, coverage, arms)
        truth = np.concatenate([
            checked_array(load_score_delta(args.cache, args.vehicles, args.plan, args.split_reference,
                                           slice_id, raw[slice_id], genes),
                          (length, len(genes)), f"held score delta {slice_id}")
            for slice_id, length in zip(slices, lengths, strict=True)
        ])
        points = operational_points(rows, truth, JoinedPredictions(full, lengths, genes), coverage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".i-operational-") as tmp:
        stage = Path(tmp)
        points.to_csv(stage / "operational.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(
            freeze_sha256=frozen_digest, original_score_sha256=sha256(score / "receipt.json"),
            code_sha256=sha256(__file__), statistic_sha256=sha256(Path(__file__).with_name("i_operational.py")),
            protocol_sha256=sha256(args.operational_protocol),
            files={"operational.tsv": sha256(stage / "operational.tsv")},
        ))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "plan", "split-reference", "cache", "vehicles", "protocol", "operational-protocol", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    print(run(parser.parse_args()))
