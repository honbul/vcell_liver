"""Compare serialized CPU CLI outputs with the already-frozen batch predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from vc_build.io import sha256

from .cli import calibration, infer_from_bundle
from .inputs import json_write
from .models import KS
from .prepare import AUXILIARY


def packet(path, ids, observations):
    pd.DataFrame({"compound_id": ids, "response": observations}).to_csv(
        path, sep="\t", index=False, float_format="%.17g")


def command(model, inputs, k, out):
    return [sys.executable, "-m", "nextgen_h1_anchor.cli", "--model", str(model), "--anchors", str(inputs),
            "--k", str(k), "--out", str(out)]


def replay_one(deployed, anchors, forecast, task):
    model = deployed / task
    loaded = calibration(model)
    seal, _, uncertainty = loaded
    path = anchors / f"{task}.npz"
    batch_file = forecast / f"{task}.npz"
    a = dict(np.load(path, allow_pickle=False))
    p = dict(np.load(batch_file, allow_pickle=False))
    bounds = np.array(uncertainty["bounds"])
    support = ((a["x"] >= bounds[:, 0]) & (a["x"] <= bounds[:, 1])).all(axis=1)
    if not support.any():
        if not task.startswith(f"{AUXILIARY}-"):
            raise ValueError(f"No held assay rows within the CLI input envelope: {task}")
        with TemporaryDirectory(dir=deployed.parent, prefix="cli-replay-") as scratch:
            source, result = (Path(scratch) / name for name in ("in.tsv", "out.tsv"))
            k = 5
            for values in a["x"]:
                packet(source, seal["anchors"][:k], values[:k])
                failure = subprocess.run(command(model, source, k, result), check=False, capture_output=True, text=True)
                if failure.returncode != 2 or "outside" not in failure.stderr or result.exists():
                    raise ValueError("A Wong out-of-envelope packet was accepted")
        return task, {"model_sha256": seal["artifacts"]["model.npz"], "batch_score_sha256": sha256(batch_file),
                      "interval_sha256": sha256(model / "intervals.json"), "tested_rows": 0,
                      "rows_outside_cli_envelope": len(a["x"]), "invalid_input_exit_codes": {"out_of_range": 2},
                      "sensitivity_scope": "auxiliary research only; all held packets rejected by CLI"}
    first = int(np.flatnonzero(support)[0])
    deviations = {}
    with TemporaryDirectory(dir=deployed.parent, prefix="cli-replay-") as scratch:
        source, result = (Path(scratch) / name for name in ("in.tsv", "out.tsv"))
        for k in KS:
            ids = seal["anchors"][:k]
            expected = p[f"H1-Anchor-Best/{k}"][support]
            observed = []
            for row in a["x"][support, :k]:
                packet(source, ids, row)
                product = infer_from_bundle(loaded, source, k).set_index("compound_id").loc[seal["query_panel"]]
                observed.append(product.score.to_numpy())
                if (product.interval_lower > product.score).any() or (product.interval_upper < product.score).any():
                    raise ValueError("Uncertainty interval does not contain its forecast")
            deviations[str(k)] = float(np.max(np.abs(np.array(observed) - expected)))
        k = 5
        values = a["x"][first, :k]
        ids = seal["anchors"][:k]
        packet(source, ids, values)
        subprocess.run(command(model, source, k, result), check=True, capture_output=True, text=True)
        emitted = pd.read_csv(result, sep="\t", index_col="compound_id", float_precision="round_trip")
        if not emitted.score.is_monotonic_decreasing or list(emitted["rank"]) != list(range(1, len(emitted) + 1)):
            raise ValueError("Output is not in descending forecast order")
        deviation = float(np.max(np.abs(emitted.loc[seal["query_panel"], "score"] - p[f"H1-Anchor-Best/{k}"][first])))
        original = result.read_bytes()
        rejected = {}
        invalid = [("missing_id", ids[:-1], values[:-1]), ("missing_value", ids, [np.nan, *values[1:]]),
                   ("out_of_range", ids, [bounds[0, 1] + 1, *values[1:]])]
        for label, drugs, observations in invalid:
            packet(source, drugs, observations)
            failure = subprocess.run(command(model, source, k, result), check=False, capture_output=True, text=True)
            if failure.returncode != 2 or result.read_bytes() != original:
                raise ValueError(f"CLI did not reject {label} cleanly")
            rejected[label] = failure.returncode
    return task, {"model_sha256": seal["artifacts"]["model.npz"], "batch_score_sha256": sha256(batch_file),
                  "interval_sha256": sha256(model / "intervals.json"), "k5_shell_max_abs_diff": deviation,
                  "single_batch_max_abs_diff": deviations, "tested_rows": int(support.sum()),
                  "rows_outside_cli_envelope": int((~support).sum()), "invalid_input_exit_codes": rejected}


def main(deployed, anchors, forecast, out):
    frozen = json.loads((forecast / "forecast_receipt.json").read_text())
    tasks = sorted(frozen["files"])
    for task in tasks:
        if sha256(forecast / f"{task}.npz") != frozen["files"][task]:
            raise ValueError("Replay batch file failed seal")
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = dict(pool.map(lambda t: replay_one(deployed, anchors, forecast, t), tasks))
    maximum = max(max(r["k5_shell_max_abs_diff"], *r["single_batch_max_abs_diff"].values())
                  for r in receipts.values() if r["tested_rows"])
    if maximum >= 1e-6:
        raise ValueError(f"CLI mismatch: {maximum}")
    json_write(out, {"cpu_only": True, "tsv_command_tested": True, "forecast_receipt_sha256": sha256(
        forecast / "forecast_receipt.json"), "max_abs_diff": maximum, "acceptance_threshold": 1e-6,
        "replay_passed": True, "tasks": receipts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deployed", "anchors", "forecast", "out"):
        parser.add_argument(name, type=Path)
    main(**vars(parser.parse_args()))
