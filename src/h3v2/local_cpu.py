"""Run a registered CPU Huber batch using a staged copy of inner-CV inputs."""
import argparse
import json
import os
from pathlib import Path
from socket import gethostname


def plan_source(root, stage):
    if stage == "boundary-1":
        return root / "locks/boundary-1.json"
    if stage == "boundary-2":
        return root / "locks/boundary-2-early-primary-tree.json"
    raise ValueError(f"Unsupported local Huber stage {stage}")


def huber_options(stage, split):
    from .fit import options
    from .selection import read_primary_tree_expansion

    if stage == "boundary-2":
        return read_primary_tree_expansion(stage)["splits"][split]["gbdt"]
    return options(stage, split, "gbdt", "gbdt")


def run(storage, plan_file, group_index):
    import numpy as np

    from nextgen_h2 import assets
    from vc_build.io import sha256

    assets.STORAGE = storage.resolve()
    from .data import ROOT
    from .fit import fold_fit
    from .grid import identify

    plan = json.loads(plan_file.read_text())
    stage = plan["stage"]
    group = plan["groups"][group_index]
    affinity = sorted(os.sched_getaffinity(0))
    if (gethostname().split(".")[0] != plan["worker"] or affinity != group["affinity"]
            or len(affinity) != plan["threads"] or int(os.environ["OMP_NUM_THREADS"]) != plan["threads"]):
        raise ValueError("CPU worker or affinity differs from the registered run")
    if sha256(plan_source(ROOT, stage)) != plan["plan_sha256"]:
        raise ValueError("The staged selection plan has changed")
    available = {identify(config): config for config in huber_options(stage, group["split"])}
    if any(available[candidate]["objective"] != "huber" for candidate in group["candidates"]):
        raise ValueError("Local work must be a registered Huber candidate")
    split, inner = group["split"], group["inner"]
    mask_path = ROOT / group["role_mask"]
    proof = json.loads(mask_path.with_suffix(".json").read_text())
    atlas_dir = ROOT / "features" / split / f"inner-{inner}"
    atlas = json.loads((atlas_dir / "receipt.json").read_text())
    if ((proof["split"], proof["inner"], proof["outer_labels_used"]) != (split, inner, False)
            or sha256(mask_path) != proof["mask_sha256"]
            or sha256(ROOT / "splits" / split / "fit.parquet") != proof["source_sha256"]
            or sha256(atlas_dir / "schema.pkl") != atlas["schema_sha256"]
            or (proof["fit_id_sha256"], proof["held_id_sha256"]) !=
                (atlas["fit_id_sha256"], atlas["held_id_sha256"])):
        raise ValueError("The transferred inner roles or feature schema differ from the G1 registration")
    with np.load(mask_path, allow_pickle=False) as record:
        fit_mask = record["fit_mask"]
    print(f"CPU Huber worker {gethostname()} PID {os.getpid()} affinity {affinity} "
          f"for {group['split']}/{group['inner']} with {len(group['candidates'])} candidates", flush=True)
    fold_fit(stage, split, "gbdt", inner, "gbdt", candidate_ids=group["candidates"], fit_mask=fit_mask,
             early_primary_tree=stage == "boundary-2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("storage", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("group", type=int)
    args = parser.parse_args()
    run(args.storage, args.plan, args.group)
