"""Submit completed local CPU forecasts to G1's guarded importer."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from shlex import join
from subprocess import SubprocessError, check_output, run
from time import sleep

SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
ARTIFACTS = ("receipt.json", "predictions.parquet", "curves.parquet")
FINAL_STATES = {"PUBLISHED", "COVERED", "CLUSTER-OWNER"}


def remote(*command):
    return check_output([*SSH, "g1", join(map(str, command))], text=True, timeout=120).strip()


def deliver(local, root, publisher, group, identity, *, batch=0):
    suffix = f"-batch-{batch}" if batch else ""
    remote_dir = root / f"external-cpu/daea-20261125{suffix}" / str(group) / identity
    remote("mkdir", "-p", remote_dir)
    run(["rsync", "-a", "--ignore-existing", "-e", join(SSH),
         *(str(local / file) for file in ARTIFACTS), f"g1:{remote_dir}/"], check=True, timeout=120)
    snapshot = root / "code/full-lincs-v24"
    batch_args = ("--batch", batch) if batch else ()
    result = remote("env", "SLURM_CPUS_PER_TASK=1", snapshot / "experiments/nextgen_h3_v2/job.sbatch",
                    snapshot, publisher, "publish", group, identity, *batch_args)
    state, verified_id = result.rsplit(maxsplit=1)
    if state not in FINAL_STATES or verified_id != identity:
        raise ValueError(f"Unexpected CPU result disposition: {result}")
    return state


def watch(work, root):
    plan = json.loads((work / "active-plan.json").read_text())
    stage = plan["stage"]
    batch = plan.get("batch", 0)
    unit_prefix = f"ng2-h3-daea-batch-{batch}" if batch else "ng2-h3-daea"
    publisher = root / f"code/daea-huber-{plan['worker_commit'][:7]}/publish_cpu.py"
    journal = work / "upload-audit.jsonl"
    audit = map(json.loads, journal.read_text().splitlines()) if journal.exists() else ()
    handled = {(row["group"], row["candidate"]) for row in audit}
    fit_root = work / "storage/nextgen_v2/h3_v2/fits" / stage
    while True:
        ended = all((work / f"worker-{index}.exit").is_file() for index in range(len(plan["groups"])))
        if not ended:
            units = [f"{unit_prefix}-cpu-{index}" for index in range(len(plan["groups"]))]
            ended = run(["systemctl", "--user", "is-active", "--quiet", *units], timeout=10).returncode == 3
        ready = []
        for index, group in enumerate(plan["groups"]):
            folder = fit_root / group["split"] / f"inner-{group['inner']}" / "gbdt"
            ready.extend((index, identity, folder / identity) for identity in group["candidates"]
                         if (index, identity) not in handled and (folder / identity / "receipt.json").is_file())
        for index, identity, local in ready:
            try:
                state = deliver(local, root, publisher, index, identity, batch=batch)
            except SubprocessError as exc:
                print(f"G1 import for {identity} will retry after {exc}", flush=True)
                continue
            proof = {"time_utc": datetime.now(timezone.utc).isoformat(), "candidate": identity,
                     "group": index, "status": state}
            with journal.open("a") as stream:
                print(json.dumps(proof), file=stream)
            handled.add((index, identity))
            print(state, index, identity, flush=True)
        if ended and all((index, identity) in handled for index, identity, _ in ready):
            print("Delivered or accounted for", len(handled), "completed CPU results", flush=True)
            return
        sleep(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("g1_root", type=Path)
    args = parser.parse_args()
    watch(args.work, args.g1_root)
