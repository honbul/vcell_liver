"""Check every locked inner validation arm and K on the same A8-complement Q."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from vc_build.io import sha256

from .inputs import json_write
from .metrics import Retrieval
from .models import KS, chemistry
from .outer import derange
from .tune import get_inputs, inner_splits


def check_one(args):
    data, frozen, task = args
    folder = frozen / task
    seal = json.loads((folder / "freeze.json").read_text())
    study, block, _, meta = get_inputs(data, task)
    chem, similarity = chemistry(meta)
    rows = []
    for j, split in enumerate(inner_splits(block, study, chem, similarity)):
        q, good = split.with_policy(seal["policy"])
        a = split.fit.anchors
        x = split.held[good][:, a]
        indexes = np.flatnonzero(good)
        order_a = derange(len(x), 442181)
        order_rna = derange(len(x), 442183)
        score = Retrieval(split.held[good][:, q])
        if len(q) < 15 or set(q) & set(a) or len(set(a)) != 8:
            raise ValueError("Inner Q/A8 contract failed")
        for k in KS:
            obs = x[:, :k]
            recipes = seal["baselines"] | {"H1-Anchor-Best": seal["primary"], "N1": seal["primary"]}
            if seal["rna_control"] is not None:
                recipes["N2"] = seal["rna_control"]
            if seal["rna_secondary"] is not None:
                recipes["H1-RNA+Anchor-Best"] = seal["rna_secondary"]
            for label, settings in recipes.items():
                anchor_input = obs[order_a] if label == "N1" else obs
                rna_inputs = indexes[order_rna] if label == "N2" else indexes
                value = split.fit.predict(anchor_input, k, settings, rna_index=rna_inputs)
                ndcg, precision = np.nanmean(score(value[:, q]), axis=0)
                rows.append({"task": task, "inner_fold": j, "k": k, "method": label,
                             "ndcg_at_5": ndcg, "precision_at_5": precision, "n": int(good.sum()),
                             "q": len(q), "A8": "|".join(block["panel"][a])})
    return rows


def run(data, frozen, out):
    out.mkdir()
    register = json.loads((data / "receipt.json").read_text())
    tasks = register["tasks"]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))) as pool:
        rows = [r for group in pool.map(check_one, [(data, frozen, task) for task in tasks]) for r in group]
    path = out / "H1_INNER_LOCKED_CONTROLS.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, float_format="%.8g")
    json_write(out / "inner_control_receipt.json",
               {"outer_outcomes_opened": False, "check_rows": len(rows), "tasks": tasks,
                "data_receipt_sha256": sha256(data / "receipt.json"), "table_sha256": sha256(path),
                "freeze_sha256": {task: sha256(frozen / task / "freeze.json") for task in tasks}})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "frozen", "out"):
        cli.add_argument(key, type=Path)
    run(**vars(cli.parse_args()))
