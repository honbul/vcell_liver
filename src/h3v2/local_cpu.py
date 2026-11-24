"""Run a registered CPU Huber batch using a staged copy of inner-CV inputs."""
import argparse
import json
import os
from pathlib import Path
from socket import gethostname


def run(storage, plan_file, group_index):
    from nextgen_h2 import assets
    from vc_build.io import sha256

    assets.STORAGE = storage.resolve()
    from .data import ROOT
    from .fit import fold_fit, options
    from .grid import identify

    plan = json.loads(plan_file.read_text())
    stage = plan["stage"]
    group = plan["groups"][group_index]
    affinity = sorted(os.sched_getaffinity(0))
    if (gethostname().split(".")[0] != plan["worker"] or affinity != group["affinity"]
            or len(affinity) != plan["threads"] or int(os.environ["OMP_NUM_THREADS"]) != plan["threads"]):
        raise ValueError("CPU worker or affinity differs from the registered run")
    if sha256(ROOT / "locks" / f"{stage}.json") != plan["plan_sha256"]:
        raise ValueError("The staged selection plan has changed")
    available = {identify(config): config for config in options(stage, group["split"], "gbdt", "gbdt")}
    if any(available[candidate]["objective"] != "huber" for candidate in group["candidates"]):
        raise ValueError("Local work must be a registered Huber candidate")
    print(f"CPU Huber worker {gethostname()} PID {os.getpid()} affinity {affinity} "
          f"for {group['split']}/{group['inner']} with {len(group['candidates'])} candidates", flush=True)
    fold_fit(stage, group["split"], "gbdt", group["inner"], "gbdt", candidate_ids=group["candidates"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("storage", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("group", type=int)
    args = parser.parse_args()
    run(args.storage, args.plan, args.group)
