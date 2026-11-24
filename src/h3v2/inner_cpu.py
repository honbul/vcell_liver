"""Dispatch smaller batches of registered inner-CV tree candidates."""
import argparse
import os
from pathlib import Path

from nextgen_h2.run import read_json
from vc_build.io import publish_json, sha256

from .data import ROOT
from .grid import identify

TREE_BATCH_SIZE = 3


def register_batches(stage, indexed, *, retry=False):
    source = ROOT / "locks" / f"{stage}.json"
    registered = read_json(source)["splits"]
    fit_root = ROOT / "fits" / stage
    expensive, short = [], []
    assigned = set()
    for index, receipts in sorted(indexed.items()):
        pending = [path for path in receipts if not path.is_file()]
        if not pending:
            continue
        split, role, family, _, _ = pending[0].relative_to(fit_root).parts
        config_by_id = {identify(config): config for config in registered[split][family]
                        if config["backend"] == "gbdt"}
        heavy, l2 = [], []
        for path in pending:
            group = path.relative_to(fit_root).parts
            if path in assigned or group[:3] != (split, role, family) or group[-1] != "receipt.json":
                raise ValueError(f"Tree work {index} repeats a candidate or spans multiple inner fits")
            assigned.add(path)
            identity = path.parent.name
            (l2 if config_by_id[identity]["objective"] == "l2" else heavy).append(identity)
        base = {"index": index, "split": split, "inner": int(role.removeprefix("inner-")), "family": family}
        expensive.extend(base | {"candidates": heavy[start:start + TREE_BATCH_SIZE]}
                         for start in range(0, len(heavy), TREE_BATCH_SIZE))
        if l2:
            short.append(base | {"candidates": l2})
    batches = expensive + short
    if not batches:
        return None, range(0)
    suffix = "-retry" if retry else ""
    path = ROOT / "locks" / f"{stage}-cpu-batches{suffix}.json"
    publish_json(path, {"stage": stage, "plan_sha256": sha256(source), "batches": batches})
    return path, range(len(batches))


def fit_batch(path, index):
    record = read_json(path)
    stage = record["stage"]
    if (ROOT / "score/consumed.json").exists() or (
        record["plan_sha256"] != sha256(ROOT / "locks" / f"{stage}.json")
    ):
        raise ValueError("The registered inner tree work is closed or differs from its selection plan")
    selection = record["batches"][index]
    from .fit import fold_fit

    fold_fit(stage, selection["split"], selection["family"], selection["inner"], "gbdt",
             candidate_ids=selection["candidates"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_plan", type=Path)
    args = parser.parse_args()
    fit_batch(args.batch_plan, int(os.environ["SLURM_ARRAY_TASK_ID"]))
