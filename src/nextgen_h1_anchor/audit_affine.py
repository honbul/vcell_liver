"""Check the corrected B3 on the previously locked inner splits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from vc_build.io import sha256

from .cli import bundle
from .inputs import json_write
from .metrics import Retrieval
from .models import AFFINE_SHIFT_PENALTY, KS, chemistry
from .tune import get_inputs, inner_splits


def check_task(args):
    data, frozen, task = args
    seal = json.loads((frozen / task / "freeze.json").read_text())
    study, block, _, meta = get_inputs(data, task)
    chem, similarity = chemistry(meta)
    old_recipe = seal["baselines"]["B3"]
    new_recipe = old_recipe | {"shift_penalty": AFFINE_SHIFT_PENALTY}
    rows = []
    for fold, split in enumerate(inner_splits(block, study, chem, similarity)):
        q, good = split.with_policy(seal["policy"])
        anchors = split.fit.anchors
        target = split.held[good][:, q]
        score = Retrieval(target)
        for k in KS:
            packet = split.held[good][:, anchors[:k]]
            index = np.flatnonzero(good)
            original, fixed = (split.fit.predict(packet, k, recipe, rna_index=index)[:, q]
                               for recipe in (old_recipe, new_recipe))
            rank_delta = np.nanmax(np.abs(score(original, full=True) - score(fixed, full=True)))
            if rank_delta > 1e-12:
                raise ValueError(f"{task} fold {fold} K={k} changed an inner retrieval metric")
            original_error, fixed_error = original - target, fixed - target
            rows.append({
                "task": task, "inner_fold": fold, "k": k, "base": old_recipe["base"],
                "a8": "|".join(block["panel"][anchors]), "q_size": len(q),
                "samples": int(good.sum()), "measured": int(np.isfinite(target).sum()),
                "legacy_mae": np.nanmean(np.abs(original_error)), "corrected_mae": np.nanmean(np.abs(fixed_error)),
                "legacy_rmse": np.sqrt(np.nanmean(original_error ** 2)),
                "corrected_rmse": np.sqrt(np.nanmean(fixed_error ** 2)),
                "max_inner_rank_metric_change": rank_delta,
            })
    return rows


def run(data, frozen, out):
    out.mkdir()
    receipt = data / "receipt.json"
    tasks = json.loads(receipt.read_text())["tasks"]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))) as pool:
        rows = [row for result in pool.map(check_task, [(data, frozen, t) for t in tasks]) for row in result]
    table = out / "H1_B3_INNER_CORRECTION.tsv"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False, float_format="%.10g")
    json_write(out / "b3_inner_correction_receipt.json", {
        "data_receipt_sha256": sha256(receipt), "rows": len(rows), "shift_penalty": AFFINE_SHIFT_PENALTY,
        "frozen_b3_base_retained": True, "outer_outcomes_opened": False,
        "freeze_sha256": {task: sha256(frozen / task / "freeze.json") for task in tasks},
        "code_sha256": {name: sha256(Path(__file__).with_name(name))
                        for name in ("audit_affine.py", "metrics.py", "models.py", "tune.py")},
        "table_sha256": sha256(table),
        "max_inner_rank_metric_change": max(r["max_inner_rank_metric_change"] for r in rows),
    })


def audit_forecasts(frozen, anchors, forecasts, out):
    """Compare ranks on signed predictions using only the released anchor packets."""
    stamp = anchors / "anchors_receipt.json"
    packets = json.loads(stamp.read_text())
    path = forecasts / "forecast_receipt.json"
    receipt = json.loads(path.read_text())
    if (receipt["input_sha256"] != sha256(stamp) or packets["evaluation_query_values_exported"]
            or receipt["prediction_has_read_Q_truth"]):
        raise ValueError("Forecast provenance does not verify against the anchor-only export")
    results = {}
    for task, expected in receipt["files"].items():
        record = packets["tasks"][task]
        freeze = frozen / task
        a_path, pred_path = anchors / f"{task}.npz", forecasts / f"{task}.npz"
        if (sha256(a_path), sha256(pred_path), sha256(freeze / "freeze.json")) != (
                record["sha256"], expected, record["freeze"]):
            raise ValueError(f"Hash mismatch for {task}")
        seal, fit = bundle(freeze)
        recipe = seal["baselines"]["B3"]
        corrected = recipe | {"shift_penalty": AFFINE_SHIFT_PENALTY}
        task_metrics = {}
        with np.load(a_path, allow_pickle=False) as a, np.load(pred_path, allow_pickle=False) as saved:
            if any(not np.array_equal(a[anchor_key], saved[key])
                   for anchor_key, key in (("group", "group"), ("index", "index"), ("q", "q"))):
                raise ValueError(f"Frozen group or Q support differs for {task}")
            for k in KS:
                original = saved[f"B3/{k}"]
                packet = a["x"][:, :k]
                the_same = fit.predict(packet, k, recipe, rna_index=a["index"])[:, a["q"]]
                if not np.array_equal(original, the_same):
                    raise ValueError(f"Archived B3 did not replay for {task} K={k}")
                new = fit.predict(packet, k, corrected, rna_index=a["index"])[:, a["q"]]
                unchanged = rankdata(original, axis=1) == rankdata(new, axis=1)
                task_metrics[str(k)] = {
                    "query_drugs": original.shape[1], "samples": len(new),
                    "samples_with_any_rank_change": int((~unchanged.all(axis=1)).sum()),
                    "max_abs_forecast_change": float(np.abs(new - original).max()),
                }
        results[task] = task_metrics
    json_write(out, {"forecast_receipt_sha256": sha256(path), "anchor_receipt_sha256": sha256(stamp),
                     "outer_outcomes_opened": False, "shift_penalty": AFFINE_SHIFT_PENALTY, "tasks": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, fields in (("inner", ("data", "frozen", "out")),
                            ("forecasts", ("frozen", "anchors", "forecasts", "out"))):
        sub = commands.add_parser(command)
        for name in fields:
            sub.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"inner": run, "forecasts": audit_forecasts}[command](**args)
