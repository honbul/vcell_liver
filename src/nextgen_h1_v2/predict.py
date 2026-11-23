"""Produce common-Q forecasts using only frozen models and released anchor packets."""
import argparse
import json
from pathlib import Path

import numpy as np

from nextgen_h1_anchor.cli import bundle as v1_package
from nextgen_h1_anchor.inputs import json_write
from nextgen_h1_anchor.outer import derange
from vc_build.io import sha256

from .data import PRIMARY_BUDGETS, TASKS, V1
from .freeze import ANCHOR, DUAL, read_package


def old_receipts():
    receipts = {d: json.loads((V1 / d / f"{d.split('-')[0]}_receipt.json").read_text())
                for d in ("anchors-v1", "forecast-v1")}
    ar, pr = receipts["anchors-v1"], receipts["forecast-v1"]
    if pr["input_sha256"] != sha256(V1 / "anchors-v1/anchors_receipt.json") or ar["evaluation_query_values_exported"]:
        raise ValueError("v1 held anchor provenance mismatch")
    return ar, pr


def anchors_v1(task, stamp):
    file = V1 / f"anchors-v1/{task}.npz"
    if sha256(file) != stamp["tasks"][task]["sha256"]:
        raise ValueError("Archived v1 anchor packet changed")
    with np.load(file, allow_pickle=False) as packed:
        return dict(packed)


def v1_reference(task, stamp):
    file = V1 / f"forecast-v1/{task}.npz"
    if sha256(file) != stamp["files"][task]:
        raise ValueError("Archived v1 forecast changed")
    with np.load(file, allow_pickle=False) as p:
        return dict(p)


def controls(task, packet, k, q, index, package, historical):
    seal, old = v1_package(V1 / f"freeze-v1/{task}")
    a8 = old.anchors
    can_reuse = k in {0, 3, 5, 8}
    simple = package["simple"]
    # RNA-only and drug mean are identical on the matched archived support.
    # The anchor-consuming simple baselines use the v2 anchor-only order.
    old.anchors = np.asarray(package["orders"]["anchor"])
    controls = {name: historical[f"{name}/{k}"] if can_reuse and name in {"B0", "B1"} else
                old.predict(packet[index][:, old.anchors[:k]], k, cfg, rna_index=index)[:, q]
                for name, cfg in simple["configs"].items()}
    controls["Bsimple"] = controls[simple["strongest"]]
    old.anchors = np.r_[a8, package["extension"]] if k == 12 else a8
    if can_reuse:
        controls["v1-Anchor-Best"] = historical[f"H1-Anchor-Best/{k}"]
        controls["v1-Bsimple"] = historical[f"Bsimple/{k}"]
    else:
        controls["v1-Anchor-Best"] = old.predict(packet[index][:, old.anchors[:k]], k, seal["primary"],
                                                rna_index=index)[:, q]
        settings = seal["baselines"][seal["baseline"]]
        controls["v1-Bsimple"] = old.predict(packet[index][:, old.anchors[:k]], k, settings, rna_index=index)[:, q]
    if seal["rna_control"] is not None:
        historical_dual = f"H1-RNA+Anchor-Best/{k}"
        if "H1-RNA+Anchor-Best/5" in historical:
            recipe = seal["rna_secondary"] or seal["primary"]
            controls["v1-RNA+Anchor"] = historical[historical_dual] if can_reuse else old.predict(
                packet[index][:, old.anchors[:k]], k, recipe, rna_index=index)[:, q]
        else:
            controls["v1-RNA+Anchor-ungated"] = old.predict(packet[index][:, old.anchors[:k]],
                k, seal["rna_control"], rna_index=index)[:, q]
    return controls


def models_predict(package, packet, k, q, index):
    out = {}
    for name, model in package["models"].items():
        obs = packet[:, model.order[:k]]
        out[name] = model.predict(obs, k)[index][:, q]
    permutation = np.arange(len(package["groups"]))
    permutation[index] = index[derange(len(index), 101181)]
    for model_name, control in ((ANCHOR, "N1"), (DUAL, "N1-Dual")):
        if model_name in package["models"]:
            model = package["models"][model_name]
            obs = packet[permutation][:, model.order[:k]]
            out[control] = model.predict(obs, k)[index][:, q]
    if DUAL in package["models"]:
        model = package["models"][DUAL]
        out["N2"] = model.predict(packet[:, model.order[:k]], k, scramble=permutation)[index][:, q]
    if not all(np.isfinite(a).all() and a.shape == (len(index), len(q)) for a in out.values()):
        raise ValueError("Prediction failed to cover the shared query")
    return out


def verify_freeze(freezes, sealed, task):
    gate = json.loads(sealed.read_text())
    stamp = freezes / task / "freeze.json"
    if gate["outer_labels_opened"] or gate["packages"][task] != sha256(stamp):
        raise ValueError(f"Freeze gate mismatch for {task}")
    return read_package(freezes / task)


def primary(task, freezes, sealed, out):
    out.mkdir(parents=True, exist_ok=True)
    seal, package = verify_freeze(freezes, sealed, task)
    ar, pr = old_receipts()
    old_anchor, historical = anchors_v1(task, ar), v1_reference(task, pr)
    index, q = old_anchor["index"], package["q8"]
    if any(not np.array_equal(old_anchor[field], historical[field]) for field in ("index", "group", "q")):
        raise ValueError(f"v1 common eligibility does not match the saved forecasts for {task}")
    if not np.array_equal(old_anchor["q"], q) or not np.array_equal(package["groups"][index], old_anchor["group"]):
        raise ValueError(f"v2 support changed for {task}")
    ids = {v: j for j, v in enumerate(package["panel"])}
    cols = [ids[a] for a in seal["a8"]]
    packet = np.full((len(package["groups"]), len(package["panel"])), np.nan)
    packet[np.ix_(index, cols)] = old_anchor["x"]
    outputs = {}
    for k in PRIMARY_BUDGETS:
        observed = models_predict(package, packet, k, q, index)
        observed.update(controls(task, packet, k, q, index, package, historical))
        outputs.update({f"{name}/{k}": value for name, value in observed.items()})
    dest = out / f"{task}.npz"
    with dest.open("xb") as handle:
        np.savez_compressed(handle, **outputs, index=index, q=q, group=package["groups"][index])
    json_write(out / f"{task}.json", {"task": task, "sha256": sha256(dest), "scope": "Q8",
                                     "n": len(index), "q": len(q), "freeze_sha256": sha256(freezes / task / "freeze.json"),
                                     "outer_labels_opened": False})
    print(task, "primary forecasts sealed", flush=True)


def seal_predictions(forecasts, frozen, out):
    stamps = {}
    for task in TASKS:
        path = forecasts / f"{task}.json"
        record = json.loads(path.read_text())
        if (record["task"] != task or record["outer_labels_opened"]
                or record["sha256"] != sha256(forecasts / f"{task}.npz")):
            raise ValueError("Invalid primary forecast stamp")
        stamps[task] = sha256(path)
    json_write(out, {"prediction_receipts": stamps, "frozen_sha256": sha256(frozen), "outer_labels_opened": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    predict = sub.add_parser("run")
    for key in ("freezes", "sealed", "out"):
        predict.add_argument(key, type=Path)
    predict.add_argument("--task")
    finish = sub.add_parser("seal")
    for key in ("forecasts", "frozen", "out"):
        finish.add_argument(key, type=Path)
    args = parser.parse_args()
    if args.stage == "seal":
        seal_predictions(args.forecasts, args.frozen, args.out)
    else:
        from os import environ
        task = args.task if args.task else TASKS[int(environ["SLURM_ARRAY_TASK_ID"])]
        primary(task, args.freezes, args.sealed, args.out)
