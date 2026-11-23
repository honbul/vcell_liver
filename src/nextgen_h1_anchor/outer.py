"""Validate freezes, calibrate intervals, and predict from isolated anchor packets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from vc_build.io import sha256

from .cli import bundle
from .inputs import json_write
from .models import KS, chemistry, config
from .prepare import AUXILIARY, TRANSFER
from .tune import get_inputs, inner_splits


def verify(root, frozen):
    registered = json.loads((root / "receipt.json").read_text())
    freezes = {}
    for task in registered["tasks"]:
        path = frozen / task / "freeze.json"
        seal, fit = bundle(path.parent)
        if seal["task"] != task or seal["data_receipt_sha256"] != sha256(root / "receipt.json"):
            raise ValueError("Freeze was fitted with a different register")
        for name, expected in seal["artifacts"].items():
            if sha256(path.parent / name) != expected:
                raise ValueError(f"{task} freeze mismatch for {name}")
        for name, expected in seal["code_sha256"].items():
            if sha256(Path(__file__).parent / name) != expected:
                raise ValueError(f"{task} fit code changed: {name}")
        _, train, query, _ = get_inputs(root, task)
        if list(train["panel"][fit.anchors]) != seal["anchors"] or len(set(fit.anchors)) != 8:
            raise ValueError("Frozen anchor identity mismatch")
        if list(np.delete(query["panel"], fit.anchors)) != seal["query_panel"]:
            raise ValueError("Frozen Q differs from the A8 complement")
        freezes[task] = sha256(path)
    return registered, freezes


def calibrate_one(args):
    root, frozen, task, out = args
    study, block, _, meta = get_inputs(root, task)
    seal, fit = bundle(frozen / task)
    inputs = []
    chem, similarity = chemistry(meta)
    for split in inner_splits(block, study, chem, similarity):
        q, good = split.with_policy(seal["policy"])
        a = split.fit.anchors
        packet = split.held[good]
        for k in KS:
            prediction = split.fit.predict(packet[:, a[:k]], k, seal["primary"], rna_index=np.flatnonzero(good))
            error = np.abs(prediction - packet)
            error[:, a] = np.nan
            inputs.append((k, error))
    is_transfer = study in {TRANSFER, AUXILIARY}
    a = fit.anchors
    q = np.setdiff1d(np.arange(fit.y.shape[1]), a)
    half_width = {}
    for k in KS:
        e = np.concatenate([values for key, values in inputs if key == k])
        pooled = float(np.nanquantile(e, .90))
        by_drug = [float(np.nanquantile(c, .90)) if np.isfinite(c).sum() >= 12 else pooled for c in e[:, q].T]
        half_width[str(k)] = by_drug
    if is_transfer:
        bounds = [[-0.25, 1.25] for _ in a]
    else:
        lo = np.nanmin(fit.y[:, a], axis=0) - 3 * fit.sd[a]
        hi = np.nanmax(fit.y[:, a], axis=0) + 3 * fit.sd[a]
        bounds = np.column_stack((lo, hi)).tolist()
    path = out / task
    path.mkdir()
    for file in ("model.npz", "freeze.json"):
        (path / file).symlink_to((frozen / task / file).resolve())
    json_write(path / "intervals.json", {"freeze_sha256": sha256(frozen / task / "freeze.json"),
               "model_sha256": seal["artifacts"]["model.npz"], "half_width": half_width, "bounds": bounds,
               "source": "fit-only nested inner absolute error 90th percentile, pooled fallback at <12 measurements",
               "transfer_intervals": "source variation only" if is_transfer else None})
    return task, sha256(path / "intervals.json")


def calibrate(root, frozen, out):
    _, freezes = verify(root, frozen)
    out.mkdir()
    jobs = [(root, frozen, task, out) for task in freezes]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))) as pool:
        receipts = dict(pool.map(calibrate_one, jobs))
    json_write(out / "calibration_receipt.json", {"freezes": freezes, "intervals": receipts})


def unlock(root, frozen, deployed, out):
    reg, freezes = verify(root, frozen)
    calib = json.loads((deployed / "calibration_receipt.json").read_text())
    if calib["freezes"] != freezes:
        raise ValueError("Calibration is not attached to the current freeze")
    if any(sha256(deployed / task / "intervals.json") != expected for task, expected in calib["intervals"].items()):
        raise ValueError("Interval calibration hash changed")
    out.mkdir()
    metadata = {}
    for task, freeze in freezes.items():
        study, _, query, _ = get_inputs(root, task)
        seal, fit = bundle(frozen / task)
        truth_file = root / "labels" / f"{task}.npz"
        if sha256(truth_file) != reg["files"][f"labels/{task}.npz"]:
            raise ValueError("Held outcomes changed after registration")
        truth = dict(np.load(truth_file, allow_pickle=False))
        if not np.array_equal(truth["groups"], query["groups"]) or not np.array_equal(truth["panel"], query["panel"]):
            raise ValueError("Held anchor identities differ from frozen query")
        y = truth["y"]
        a = fit.anchors
        q = np.setdiff1d(np.arange(len(seal["anchors"]) + len(seal["query_panel"])), a)
        ok = np.isfinite(y[:, a]).all(axis=1) & (np.isfinite(y[:, q]).sum(axis=1) >= 15)
        if (study == TRANSFER and not ok.all()) or (study != AUXILIARY and ok.sum() < 2):
            raise ValueError(f"{task} does not have enough complete held anchor packets")
        path = out / f"{task}.npz"
        np.savez_compressed(path, x=y[ok][:, a], group=query["groups"][ok], index=np.flatnonzero(ok), q=q)
        metadata[task] = {"sha256": sha256(path), "freeze": freeze, "eligible": int(ok.sum()),
                          "registered": len(ok), "eligible_query_entries": int(np.isfinite(y[ok][:, q]).sum()),
                          "minimum_measured_q": int(np.isfinite(y[ok][:, q]).sum(axis=1).min()) if ok.any() else None,
                          "q_drugs": len(q)}
    json_write(out / "anchors_receipt.json", {"tasks": metadata, "evaluation_query_values_exported": False,
                                              "calibration_sha256": sha256(deployed / "calibration_receipt.json")})


def derange(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    mapping = np.empty(n, dtype=int)
    mapping[order] = np.roll(order, 1)
    if (mapping == np.arange(n)).any():
        raise ValueError("Shuffled control needs at least two independent groups")
    return mapping


def forecast(frozen, anchors, output):
    receipt = json.loads((anchors / "anchors_receipt.json").read_text())
    output.mkdir()
    files = {}
    for task, record in receipt["tasks"].items():
        if record["eligible"] < 2:
            continue
        path = anchors / f"{task}.npz"
        if sha256(path) != record["sha256"] or sha256(frozen / task / "freeze.json") != record["freeze"]:
            raise ValueError("Changed anchor packet or freeze")
        seal, fit = bundle(frozen / task)
        packed = dict(np.load(path, allow_pickle=False))
        values, indices, q = (packed[k] for k in ("x", "index", "q"))
        permutation_a = derange(len(values), 442181)
        permutation_rna = derange(len(values), 442183)
        predictions = {}
        for k in KS:
            obs = values[:, :k]
            for name, settings in seal["baselines"].items():
                predictions[f"{name}/{k}"] = fit.predict(obs, k, settings, rna_index=indices)[:, q]
            pred = fit.predict(obs, k, seal["primary"], rna_index=indices)[:, q]
            predictions[f"H1-Anchor-Best/{k}"] = pred
            predictions[f"Bsimple/{k}"] = predictions[f"{seal['baseline']}/{k}"]
            predictions[f"N1/{k}"] = fit.predict(obs[permutation_a], k, seal["primary"], rna_index=indices)[:, q]
            if seal["rna_control"] is not None:
                setting = seal["rna_control"]
                predictions[f"N2/{k}"] = fit.predict(obs, k, setting, rna_index=indices[permutation_rna])[:, q]
            if seal["rna_secondary"] is not None:
                predictions[f"H1-RNA+Anchor-Best/{k}"] = fit.predict(
                    obs, k, seal["rna_secondary"], rna_index=indices)[:, q]
            if not np.isfinite(pred).all():
                raise ValueError("Nonfinite point forecast")
        file = output / f"{task}.npz"
        np.savez_compressed(file, **predictions, group=packed["group"], index=indices, q=q)
        files[task] = sha256(file)
    json_write(output / "forecast_receipt.json", {"input_sha256": sha256(anchors / "anchors_receipt.json"),
                                                 "files": files, "prediction_has_read_Q_truth": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for stage, fields in {"calibrate": ("root", "frozen", "out"),
                          "unlock": ("root", "frozen", "deployed", "out"),
                          "forecast": ("frozen", "anchors", "out")}.items():
        cli = sub.add_parser(stage)
        for field in fields:
            cli.add_argument(field, type=Path)
    args = parser.parse_args()
    stage = args.command
    payload = vars(args)
    del payload["command"]
    if stage == "forecast":
        payload["output"] = payload.pop("out")
    {"calibrate": calibrate, "unlock": unlock, "forecast": forecast}[stage](**payload)
