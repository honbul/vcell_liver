"""Verify package, Q12 forecast paths and metric outputs before opening outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np

from nextgen_h1_anchor.inputs import json_write
from nextgen_h1_anchor.score import Task
from vc_build.io import sha256

from .data import TASKS
from .predict import anchors_v1, controls, models_predict, old_receipts, verify_freeze
from .score import fit_metrics, validate


def execute(frozen, freeze_gate, forecasts, forecast_gate, replay, out):
    validate(forecasts, forecast_gate, frozen, freeze_gate)
    cpu = json.loads(replay.read_text())
    if not cpu["passed"] or set(cpu["tasks"]) != set(TASKS):
        raise ValueError("Primary CPU replay must pass before the scoring claim")
    archive, _ = old_receipts()
    counts = {}
    for task in TASKS:
        info, p = verify_freeze(frozen, freeze_gate, task)
        with np.load(forecasts / f"{task}.npz", allow_pickle=False) as primary:
            published_dual = "v1-RNA+Anchor/5" in primary
            primary_arms = {k.rsplit("/", 1)[0] for k in primary.files if k.endswith("/5")}
        if task.startswith(("GDSC2-", "CTRPv2-", "PRISM20Q2-")) and not published_dual:
            raise ValueError(f"Pan-cancer tissue comparison requires the archived v1 RNA+Anchor arm on {task}")
        ids = np.asarray(p["orders"]["anchor"])
        a = anchors_v1(task, archive)
        model = next(iter(p["models"].values()))
        # Only Q12 medication availability is simulated. No response in Q8/Q12
        # from the held outcome packages is opened here.
        index = a["index"][:min(8, len(a["index"]))]
        fake_packet = np.broadcast_to(model.ctx.mean, (len(p["groups"]), len(p["panel"]))).copy()
        by_key = {v: j for j, v in enumerate(p["panel"])}
        fake_packet[np.ix_(index, [by_key[x] for x in info["a8"]])] = a["x"][:len(index)]
        q = np.setdiff1d(np.arange(len(p["panel"])), ids)
        predictions = models_predict(p, fake_packet, 12, q, index)
        predictions.update(controls(task, fake_packet, 12, q, index, p, {}, published_dual=published_dual))
        if set(predictions) != primary_arms or any(not np.isfinite(value).all() for value in predictions.values()):
            raise ValueError(f"Auxiliary matched arms or finite support changed on {task}")
        fake_y = np.arange(len(index))[:, None] * .03 + np.linspace(.15, .85, len(q))[None]
        t = Task(task, fake_y, predictions, p["groups"][index], np.array(["mock"] * len(index)), None)
        t.drawn_indices = np.vstack((np.arange(len(index)), np.arange(len(index))[::-1]))
        _, metrics, _, counts_fit = fit_metrics(t)
        if set(metrics) != set(predictions) or not all(v.shape == (2, 9) for v in metrics.values()):
            raise ValueError("Metric contract mismatch")
        counts[task] = {"arm_count": len(metrics), "q12": len(q), "groups": len(index),
                        "published_v1_dual": published_dual}
    json_write(out, {"tasks": counts, "holdout_outcomes_opened": False, "prelabel_replay": sha256(replay),
                    "sealed_models": sha256(freeze_gate), "sealed_forecasts": sha256(forecast_gate)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("frozen", "freeze_gate", "forecasts", "forecast_gate", "replay", "out"):
        parser.add_argument(field, type=Path)
    execute(**vars(parser.parse_args()))
