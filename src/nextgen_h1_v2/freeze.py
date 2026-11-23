"""Fit sealed outer-fold packages without opening any held response labels."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from nextgen_h1_anchor.cli import bundle as opener
from nextgen_h1_anchor.inputs import json_write
from vc_build.io import sha256

from .anchors import extend_reserve, nested_contexts, order_policy
from .data import ROOT, STUDY, TASKS, V1, build, drug_features, source
from .search import fit_model

ANCHOR = "H1-v2-AnchorOnly-Best"
DUAL = "H1-v2-Dual-Best"
FAMILIES = ("A", "B", "C", "D")


def experiment_paths(task, result, kernel):
    return {family: (kernel if family == "A" else result) / task / family / "result.json"
            for family in FAMILIES}


def load_results(task, result, kernel, alternative=None):
    files = experiment_paths(task, result, kernel)
    for family, path in files.items():
        replacement = alternative / task / family / "result.json" if alternative else None
        files[family] = replacement if replacement is not None and replacement.exists() else path
    entries = {family: json.loads(path.read_text()) for family, path in files.items()}
    if any(r["outer_labels_opened"] or r["task"] != task or r["family"] != family for family, r in entries.items()):
        raise ValueError(f"Invalid training receipt for {task}")
    return entries, {family: sha256(path) for family, path in files.items()}


def choose(choices):
    return max(choices, key=lambda k: (choices[k]["selected"]["objective"], choices[k]["selected"]["ndcg_at_5"], k))


def fit_package(task, result, kernel, out, alternative=None):
    entries, stamps = load_results(task, result, kernel, alternative)
    out.mkdir(parents=True)
    study, dev, meta, held = source(task, query=True)
    orig, orig_fit = opener(V1 / "freeze-v1" / task)
    receipt = json.loads((V1 / "score-v1/score_receipt.json").read_text())
    old_freeze_hash = sha256(V1 / f"freeze-v1/{task}/freeze.json")
    if old_freeze_hash != receipt["masks"][task]["freeze"] or orig["anchors"] != list(dev["panel"][orig_fit.anchors]):
        raise ValueError("v1 reserve differs from signed v1 support")
    a8 = orig_fit.anchors
    q8 = np.setdiff1d(np.arange(len(meta)), a8)
    if list(dev["panel"][q8]) != orig["query_panel"]:
        raise ValueError("v1 Q8 support was not preserved")
    drug = drug_features(meta)
    full = np.ones(len(dev["groups"]), dtype=bool)
    ctx = build(study, dev, full, held["x"], held["groups"], held["tissue"], drug)
    sub = nested_contexts(study, dev, full, drug)
    extended = extend_reserve(ctx, a8, 4)
    models, choices, configs, orders = {}, {}, {}, {}
    cache = {}
    modes = ("anchor",) if ctx.levels else ("anchor", "dual")
    for mode in modes:
        by_family = {name: entry["modes"][mode] for name, entry in entries.items()}
        policies = {e["policy"] for e in by_family.values()}
        if len(policies) != 1:
            raise ValueError("Model families used different fixed-policy support")
        policy = policies.pop()
        a = order_policy(ctx, sub, a8, policy)
        order = np.r_[a, extended[8:]]
        if set(map(int, a)) != set(map(int, a8)) or len(set(order)) != 12:
            raise ValueError("Invalid selected anchor order or auxiliary reserve")
        orders[mode] = order.tolist()
        champion = choose(by_family)
        choices[mode] = {"champion": champion, "policy": policy,
                         "r3_utility": {f: e["selected"]["objective"] for f, e in by_family.items()}}
        benchmark_mode = "dual" if study in {"GDSC2", "CTRPv2", "PRISM20Q2", "LIMORE"} else "anchor"
        requested = FAMILIES if benchmark_mode == mode else (champion,)
        for family in requested:
            selected = by_family[family]["selected"]
            cfg = selected["config"].copy()
            if family in {"B", "C", "D"}:
                epochs = [f["best_epoch"] for f in selected["folds"]]
                cfg["refit_epochs"] = max(1, int(np.median(epochs)))
            arm = f"Family-{family}" if benchmark_mode == mode else (DUAL if mode == "dual" else ANCHOR)
            model = fit_model(ctx, sub, order, mode == "dual", cfg, "cpu", cache)
            models[arm] = model
            configs[arm] = cfg
            if family == champion:
                key = DUAL if mode == "dual" else ANCHOR
                models[key], configs[key] = model, cfg
    state = out / "package.pkl"
    train_rna = dev["x"] if "dual" in modes else None
    with state.open("xb") as handle:
        pickle.dump(dict(models=models, train_rna=train_rna, panel=dev["panel"], groups=held["groups"],
                         q8=q8, extension=extended[8:], orders=orders, v1=orig), handle, protocol=5)
    json_write(out / "freeze.json", {"task": task, "study": study, "a8": list(dev["panel"][a8]),
                                     "a12": list(dev["panel"][extended]), "q8": list(dev["panel"][q8]),
                                     "q12": list(dev["panel"][np.setdiff1d(np.arange(len(meta)), extended)]),
                                     "choices": choices, "orders": orders, "configs": configs,
                                     "fit_groups": len(ctx.train), "held_group_count": len(held["groups"]),
                                     "package_sha256": sha256(state), "inner_receipts": stamps,
                                     "old_freeze_sha256": old_freeze_hash, "outer_labels_opened": False,
                                     "model_code": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}})
    print(task, choices, flush=True)


def read_package(path):
    freeze = json.loads((path / "freeze.json").read_text())
    state = path / "package.pkl"
    if freeze["package_sha256"] != sha256(state) or freeze["outer_labels_opened"]:
        raise ValueError(f"Invalid frozen package {path}")
    with state.open("rb") as file:
        return freeze, pickle.load(file)


def seal(directory, out):
    packages = {}
    for task in TASKS:
        fold, data = read_package(directory / task)
        if fold["task"] != task or fold["held_group_count"] != len(data["groups"]):
            raise ValueError("Freeze package identity mismatch")
        packages[task] = sha256(directory / task / "freeze.json")
    json_write(out, {"packages": packages, "task_count": len(packages), "v1_study_sha256": sha256(STUDY / "receipt.json"),
                     "anchor_policy_uses_held_outcomes": False, "outer_labels_opened": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("result", type=Path)
    fit.add_argument("kernel", type=Path)
    fit.add_argument("out", type=Path)
    fit.add_argument("--task")
    fit.add_argument("--alternative", type=Path)
    bundle = sub.add_parser("seal")
    bundle.add_argument("directory", type=Path)
    bundle.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.stage == "seal":
        seal(args.directory, args.out)
    else:
        from os import environ
        task = args.task if args.task else TASKS[int(environ["SLURM_ARRAY_TASK_ID"])]
        fit_package(task, args.result, args.kernel, args.out / task, args.alternative)
