"""Fit isolated inner stages and seal seed-level outcome-free outer forecasts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vc_build.external import native_sensitivity
from vc_build.io import freeze_table, publish_json, sha256
from vc_build.structured_release import fit_kernel
from vc_build.v0plus_cv import validation_roster

from .data import IDENTITY, ROOT, ROSTER, contract, expression, molecular_view, outer_reference
from .data import prior_structure, read_json, read_part, sample_rank, save_table, split_id
from .train import Arm, predict, train


def first_arms():
    return [Arm("G0", "smooth"), Arm("G3", "alternating"),
            *(Arm(f"G1_w{w}", "sum", w) for w in range(3)),
            *(Arm(f"G2_w{w}", "pcgrad", w) for w in range(3))]


def inner_stage(stage, index):
    name = split_id(index)
    files = ROOT / "data/splits" / name
    output = ROOT / f"{stage}/{name}"
    output.mkdir(parents=True, exist_ok=False)
    freeze = contract()
    rna = expression()
    ranked = pd.DataFrame(sample_rank(rna.to_numpy(np.float32)), index=rna.index, columns=rna.columns)
    lock = None if stage == "round1" else read_json(ROOT / f"{stage}-lock.json")
    arms = first_arms() if lock is None else [Arm(**arm) for arm in lock["splits"][name]["arms"]]
    evidence, tables = [], []
    for inner in range(2):
        fit = read_part(files / f"inner{inner}-fit.parquet")
        valid = read_part(files / f"inner{inner}-valid.parquet")
        p = valid[ROSTER].copy().assign(fold=inner)
        p["observed"] = native_sensitivity(valid)
        view = molecular_view(fit, p, rna, ranked=ranked)
        if stage == "round1":
            p["drug_mean"] = p.compound_id.map(view["anchor_series"])
            if inner == 0:
                prior, _ = prior_structure()
            model, info = fit_kernel(rna.loc[view["cells"]], view["aligned"],
                                     view["anchor_series"], *prior)
            query = validation_roster(p)
            base, residual = model.predict(query, rna.loc[sorted(p.cell_id.unique())])
            v = query.assign(structured=base + residual).set_index(["cell_id", "lineage_group", "compound_id"]).structured
            p["structured"] = v.reindex(pd.MultiIndex.from_frame(p[["cell_id", "lineage_group", "compound_id"]])).values
            evidence.append({"inner": inner, "arm": {"name": "structured"}, "kernel": info})
        audits = []
        for arm in arms:
            traced = stage in ("round1", "round2-block")
            fitted, receipt, audit = train(view, arm, freeze, freeze["seeds"][0], device="cuda", diagnostics=traced)
            evidence.append({"inner": inner, **receipt})
            p[arm.name] = predict(fitted, view, "cuda")
            audits.extend({"split": name, "inner": inner, "arm": arm.name, **row} for row in audit)
            print(stage, name, inner, arm.name, round(receipt["wall_seconds"], 2), flush=True)
            if stage == "round1" and index == 0 and inner == 0 and arm.name in {"G0", "G1_w0", "G2_w0", "G3"}:
                device_model, timed, _ = train(view, arm, freeze, freeze["seeds"][0],
                                               device="cuda", diagnostics=False)
                gap = np.max(np.abs(p[arm.name] - predict(device_model, view, "cuda")))
                evidence.append({"inner": inner, "timing_replay": True, "max_forecast_difference": float(gap), **timed})
        if audits:
            save_table(pd.DataFrame(audits), output / f"gradients-inner{inner}.parquet")
        if not np.isfinite(p[[arm.name for arm in arms]].to_numpy()).all():
            raise ValueError("Incomplete inner forecasts")
        tables.append(p)
    location = output / "forecast.parquet"
    save_table(pd.concat(tables, ignore_index=True), location)
    publish_json(output / "receipt.json", {"split": name, "stage": stage, "runs": evidence,
                                           "forecast_sha256": sha256(location)})


def outer(index):
    name = split_id(index)
    files = ROOT / "data/splits" / name
    output = ROOT / f"outer-forecast/{name}"
    output.mkdir(parents=True, exist_ok=False)
    lockfile = ROOT / "final-lock.json"
    lock = read_json(lockfile)
    conf = contract()
    fit = read_part(files / "fit.parquet")
    p = read_part(files / "query.parquet")
    if set(fit.lineage_group) & set(p.lineage_group) or set(fit.source_id) & set(p.source_id):
        raise ValueError("The historical split is no longer isolated")
    view = molecular_view(fit, p, expression())
    results = p[IDENTITY].assign(track=p.source_id + "/" + p.endpoint_id, fold=index % 3)
    parts, evidence = [], []
    references, previous_hash = outer_reference(p, name)
    for method, forecast in references.items():
        parts.append(results.assign(method=method, predicted_value=forecast))
    cache = {}
    for option in lock["folds"][name]["arms"]:
        arm = Arm(**option)
        for seed in conf["seeds"]:
            key = (arm.method, arm.weight, arm.batch, arm.head, seed)
            reused = key in cache
            if not reused:
                fitted, stamp, _ = train(view, arm, conf, seed, device="cuda", task_vectors=arm.name in {"G1_best", "G1_matched", "G2"})
                cache[key] = fitted, stamp, predict(fitted, view, "cuda")
            fitted, stamp, prediction = cache[key]
            stamp = stamp | {"arm": arm.record(), "reused_matched_fit": reused}
            component = f"{arm.name}__{seed}"
            target = output / f"{component}.pt"
            torch.save({"arm": arm.record(), "source": sha256(files / "fit.parquet"), "seed": seed,
                        "state": {k: v.cpu() for k, v in fitted.state_dict().items()},
                        "features": {key: view[key] for key in
                                     ("genes", "center", "spread", "drugs", "scale", "anchor", "drug", "x")}},
                       target)
            parts.append(results.assign(method=component, predicted_value=prediction))
            evidence.append({"checkpoint": target.name, "checkpoint_sha256": sha256(target), **stamp})
            print(name, component, round(stamp["wall_seconds"], 2), flush=True)
    freeze_table(pd.concat(parts, ignore_index=True), output / "forecast.parquet",
                 provenance={"historical_reference_sha256": previous_hash, "lock_sha256": sha256(lockfile),
                             "split": name, "runs": evidence, "labels_loaded": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["round1", "round2-block", "round2-opt", "round2-head", "outer"])
    parser.add_argument("--index", type=int, required=True, choices=range(9))
    args = parser.parse_args()
    if args.stage == "outer":
        outer(args.index)
    else:
        inner_stage(args.stage, args.index)


if __name__ == "__main__":
    main()
