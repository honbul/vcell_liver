"""Check the CLI and numerical replay against frozen primary and auxiliary predictions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from nextgen_h1_anchor.inputs import json_write
from vc_build.io import sha256

from .cli import bounds, single
from .data import BUDGETS, STUDY, TASKS
from .freeze import ANCHOR, DUAL, read_package
from .predict import anchors_v1, old_receipts


def domains(args):
    task, freezes, forecasts, auxiliary, work = args
    seal, package = read_package(freezes / task)
    a, _ = old_receipts()
    packet = anchors_v1(task, a)
    study, fold = task.rsplit("-", 1)
    with np.load(STUDY / study / f"{fold}-query.npz", allow_pickle=False) as query:
        rna = query["x"]
    out = work / task
    out.mkdir()
    with np.load(forecasts / f"{task}.npz", allow_pickle=False) as p:
        prediction = dict(p)
    primary = np.full((len(packet["index"]), len(package["panel"])), np.nan)
    position = {x: j for j, x in enumerate(package["panel"])}
    primary[:, [position[x] for x in seal["a8"]]] = packet["x"]
    aux_path = auxiliary / f"{task}.npz"
    aux = {}
    if aux_path.exists():
        with np.load(aux_path, allow_pickle=False) as f:
            aux = dict(f)
        matrix12 = np.full((len(aux["index"]), len(package["panel"])), np.nan)
        matrix12[:, aux["codes"]] = aux["anchors"]
    stats = []
    for arm in (ANCHOR, DUAL):
        if arm not in package["models"]:
            continue
        model = package["models"][arm]
        for k in BUDGETS:
            if k == 12 and not aux:
                continue
            pred, matrix = (aux, matrix12) if k == 12 else (prediction, primary)
            v = matrix[:, model.order[:k]]
            if k:
                interval = np.array(bounds(model, k))
                eligible = np.flatnonzero(((v >= interval[:, 0]) & (v <= interval[:, 1])).all(1))
            else:
                eligible = np.arange(len(v))
            if len(eligible) < 1:
                raise ValueError(f"No in-envelope replay row at {task}/{arm}/{k}")
            indices = eligible[np.linspace(0, len(eligible) - 1, min(len(eligible), 2), dtype=int)]
            errors = []
            for i in indices:
                r = rna[pred["index"][i]] if model.dual else None
                q, fitted = single(package, arm, k, v[i:i + 1], r)
                if not np.array_equal(q, pred["q"]):
                    raise ValueError("Replay changed query support")
                errors.append(float(np.max(np.abs(pred[f"{arm}/{k}"][i] - fitted))))
            stat = dict(arm=arm, k=k, eligible=int(len(eligible)), tested=len(indices), max_abs_diff=max(errors))
            if k == 5:
                i = indices[0]
                inputs = out / f"{arm}-anchors.tsv"
                result = out / f"{arm}-pred.tsv"
                pd.DataFrame({"compound_id": package["panel"][model.order[:k]], "response": v[i]}).to_csv(
                    inputs, sep="\t", index=False, float_format="%.17g")
                cmd = [sys.executable, "-m", "nextgen_h1_v2.cli", "--model", str(freezes / task),
                       "--anchors", str(inputs), "--arm", arm, "--out", str(result)]
                if model.dual:
                    rna_path = out / "rna.npy"
                    np.save(rna_path, rna[pred["index"][i]])
                    cmd += ["--rna", str(rna_path)]
                subprocess.run(cmd, check=True, capture_output=True)
                scores = pd.read_csv(result, sep="\t", float_precision="round_trip").set_index("compound_id").score
                stat["cli_abs_diff"] = float(np.max(np.abs(scores.loc[package["panel"][pred["q"]]]
                                                          - pred[f"{arm}/5"][i])))
                source = pd.read_csv(inputs, sep="\t")
                source.iloc[:-1].to_csv(inputs, index=False, sep="\t")
                missing = subprocess.run(cmd, capture_output=True).returncode
                source.loc[0, "response"] = 1e30
                source.to_csv(inputs, index=False, sep="\t")
                out_of_range = subprocess.run(cmd, capture_output=True).returncode
                if (missing, out_of_range) != (2, 2):
                    raise ValueError("CPU CLI failed to reject invalid inputs")
                stat.update(missing_exit_code=missing, outside_exit_code=out_of_range)
            stats.append(stat)
    return task, stats


def execute(freezes, forecasts, auxiliary, scratch, out):
    scratch.mkdir()
    jobs = [(t, freezes, forecasts, auxiliary, scratch) for t in TASKS]
    with ProcessPoolExecutor(max_workers=6) as pool:
        observations = dict(pool.map(domains, jobs))
    maximum = max(max(r["max_abs_diff"], r.get("cli_abs_diff", 0.))
                  for task in observations.values() for r in task)
    json_write(out, {"max_abs_diff": maximum, "passed": maximum < 1e-6, "threshold": 1e-6,
                     "device": "cpu", "outer_query_labels_opened": False, "tasks": observations})
    if maximum >= 1e-6:
        raise ValueError(f"Replay parity failed: {maximum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freezes", "forecasts", "auxiliary", "scratch", "out"):
        parser.add_argument(key, type=Path)
    execute(**vars(parser.parse_args()))
