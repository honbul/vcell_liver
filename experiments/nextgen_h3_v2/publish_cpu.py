"""Verify CPU-worker forecasts before making them available to the Slurm owner."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from subprocess import check_output

import numpy as np
import pandas as pd

from h3v2.atlas import model_input_digest
from h3v2.data import ROOT, table_hash
from h3v2.fit import fold_fit, options
from h3v2.grid import SEEDS, identify
from h3v2.pipeline import inner_tasks
from nextgen_h2.run import read_json
from vc_build.io import publish_json, sha256

STAGE = "boundary-1"
LOCK = ROOT / "locks/boundary-1-daea-huber.json"
RESULTS = ROOT / "external-cpu/daea-20261125"
FILES = {"receipt.json", "predictions.parquet", "curves.parquet"}


def owner_running(group):
    raw = check_output(["scontrol", "-o", "show", "job", group["owner"]], text=True)
    state = dict(part.split("=", 1) for part in raw.split() if "=" in part)
    return (state["JobState"] == "RUNNING" and state["JobId"] == group["slurm_job_id"]
            and state["NodeList"] == group["owner_node"] and state["JobName"].startswith("ng2-h3-boundary-1-"))


def canonical(group, identity):
    inner = f"inner-{group['inner']}"
    return ROOT / "fits" / STAGE / group["split"] / inner / "gbdt" / identity


def has_lead(group, destination):
    previous = [ROOT / path for path in group["ordered_receipts"]]
    before = previous[:previous.index(destination / "receipt.json")]
    return sum(not path.is_file() for path in before) >= 2


def local_batch(batch):
    if batch < 0:
        raise ValueError("The local CPU batch must be nonnegative")
    suffix = f"-batch-{batch}" if batch else ""
    return LOCK.with_stem(LOCK.stem + suffix), RESULTS.with_name(RESULTS.name + suffix), f"daea-cpu-imports{suffix}.tsv"


def register(path, *, batch=0):
    plan = read_json(path)
    expected = read_json(ROOT / "step0_verification.json")["versions"]["lightgbm_so_sha256"]
    if (plan["worker"] != "daea" or plan["stage"] != STAGE or plan["threads"] != 8
            or plan.get("batch", 0) != batch
            or plan["plan_sha256"] != sha256(ROOT / "locks" / f"{STAGE}.json")
            or plan["lightgbm_so_sha256"] != expected or (ROOT / "score/consumed.json").exists()):
        raise ValueError("The CPU-worker registration differs from the sealed inner run")
    reserve = plan["reserved_cpu_ids"]
    assigned = [cpu for group in plan["groups"] for cpu in group["affinity"]]
    if (len(set(reserve)) < 4 or set(reserve).intersection(assigned) or len(set(assigned)) != len(assigned)
            or any(len(group["affinity"]) != plan["threads"] for group in plan["groups"])):
        raise ValueError("The CPU-worker registration does not leave four CPUs free")
    tasks = inner_tasks(STAGE)["gbdt"]
    seen = set()
    for group in plan["groups"]:
        registered = tasks[group["index"]]
        if [str(file.relative_to(ROOT)) for file in registered] != group["ordered_receipts"] or not owner_running(group):
            raise ValueError("The original Slurm owner or candidate order has changed")
        for identity in group["candidates"]:
            path = canonical(group, identity)
            if (path / "receipt.json" not in registered or path in seen or os.path.lexists(path)
                    or not has_lead(group, path)):
                raise ValueError(f"Expected an unstarted, registered tail candidate: {path}")
            seen.add(path)
    lock, _, _ = local_batch(batch)
    publish_json(lock, plan)
    print("Registered", len(seen), "inner Huber candidates on", plan["worker"], flush=True)


def publish(group_index, identity, *, plan_path=None, staged=None, batch=0):
    lock, local_results, local_ledger = local_batch(batch)
    plan = read_json(plan_path or lock)
    group = plan["groups"][group_index]
    if identity not in group["candidates"] or (ROOT / "score/consumed.json").exists():
        raise ValueError("The candidate was not registered for CPU-worker import")
    destination = canonical(group, identity)
    if (destination / "receipt.json").is_file() or os.path.lexists(destination):
        print("COVERED", identity, flush=True)
        return
    if not owner_running(group) or not has_lead(group, destination):
        print("CLUSTER-OWNER", identity, flush=True)
        return
    staged = staged or local_results / str(group_index) / identity
    if set(path.name for path in staged.iterdir() if path.is_file() and not path.is_symlink()) != FILES:
        raise ValueError("The CPU-worker result does not contain the complete fit artifacts")
    receipt = read_json(staged / "receipt.json")
    atlas = read_json(ROOT / "features" / group["split"] / f"inner-{group['inner']}" / "receipt.json")
    config = {identify(c): c for c in options(STAGE, group["split"], "gbdt", "gbdt")}[identity]
    training = receipt["training"]
    if (receipt["stage"], receipt["split"], receipt["inner"], receipt["family"], receipt["seed"],
            receipt["config_id"], receipt["config"], receipt["query_has_outer_labels"]) != (
        STAGE, group["split"], group["inner"], "gbdt", SEEDS[0], identity, config, False
    ) or config["objective"] != "huber":
        raise ValueError("The CPU-worker fit used a different arm or partition")
    if ((training["device"], training["library"], training["train_rows"]) != ("cpu", "4.6.0", atlas["fit_rows"])
            or receipt["atlas_sha256"] != atlas["atlas_sha256"] or not receipt["lincs_full_unclipped"]
            or receipt["model_input_sha256"] != model_input_digest(atlas["atlas_sha256"])):
        raise ValueError("The CPU-worker fit used different features or hardware backend")
    forecasts = staged / "predictions.parquet"
    curves = staged / "curves.parquet"
    if (sha256(forecasts), sha256(curves)) != (receipt["forecast_sha256"], receipt["curve_sha256"]):
        raise ValueError("The transferred forecast or learning curve changed")
    preds = pd.read_parquet(forecasts)
    if (len(preds) != atlas["held_rows"] or table_hash(preds) != atlas["held_id_sha256"]
            or not np.isfinite(preds.predicted).all()):
        raise ValueError("The CPU-worker forecast differs from the registered inner-validation rows")
    history = pd.read_parquet(curves)
    trace = history.loc[history.branch.eq("huber")]
    if (set(history.branch) != {"huber", "selected"} or not trace.iteration.eq(training["best_iteration"]).any()
            or not np.isfinite(trace[["train_objective", "validation_objective"]]).all().all()):
        raise ValueError("The CPU-worker result is missing the registered Huber learning curve")
    if not has_lead(group, destination):
        print("CLUSTER-OWNER", identity, flush=True)
        return
    try:
        destination.symlink_to(os.path.relpath(staged, destination.parent), target_is_directory=True)
    except FileExistsError:
        print("COVERED", identity, flush=True)
        return
    ledger = local_ledger if plan_path is None else "slurm-cpu-tail-imports.tsv"
    with (ROOT / "logs" / ledger).open("a") as stream:
        print(datetime.now(timezone.utc).isoformat(), group["owner"], identity, sha256(staged / "receipt.json"),
              sep="\t", file=stream)
    print("PUBLISHED", identity, flush=True)


def fit_slurm_tail(plan_path, group_index):
    plan = read_json(plan_path)
    group = plan["groups"][group_index]
    if (plan["stage"] != STAGE or plan["plan_sha256"] != sha256(ROOT / "locks" / f"{STAGE}.json")
            or group["worker_node"] != os.environ["SLURMD_NODENAME"]
            or int(os.environ["SLURM_CPUS_PER_TASK"]) != 8 or (ROOT / "score/consumed.json").exists()):
        raise ValueError("The Slurm worker differs from the admitted CPU tail")
    output = ROOT / "external-cpu/slurm-huber-b1" / str(group_index)
    for identity in group["candidates"]:
        destination = canonical(group, identity)
        if os.path.lexists(destination) or not owner_running(group) or not has_lead(group, destination):
            print("CLUSTER-OWNER", identity, flush=True)
            continue
        print("CPU tail", group["worker_node"], group["split"], group["inner"], identity, flush=True)
        fold_fit(STAGE, group["split"], "gbdt", group["inner"], "gbdt",
                 candidate_ids=(identity,), artifact_root=output)
        staged = output / destination.relative_to(ROOT / "fits")
        publish(group_index, identity, plan_path=plan_path, staged=staged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    registration = actions.add_parser("register")
    registration.add_argument("plan", type=Path)
    command = actions.add_parser("publish")
    command.add_argument("group", type=int)
    command.add_argument("config_id")
    for action in (registration, command):
        action.add_argument("--batch", type=int, default=0)
    tail = actions.add_parser("slurm-tail")
    tail.add_argument("plan", type=Path)
    tail.add_argument("group", type=int, default=None, nargs="?")
    args = parser.parse_args()
    if args.action == "register":
        register(args.plan, batch=args.batch)
    elif args.action == "publish":
        publish(args.group, args.config_id, batch=args.batch)
    else:
        index = args.group if args.group is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        fit_slurm_tail(args.plan, index)
