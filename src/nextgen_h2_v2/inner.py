"""Outcome-partitioned H2-v2 search and immutable per-fold metric receipts."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nextgen_h2.assets import KEY
from nextgen_h2.metrics import Evaluator, stack
from nextgen_h2.run import META, read_json, response, subfold
from vc_build.external import native_sensitivity
from vc_build.io import publish_json, sha256

from .assets import V1_ASSETS
from .kernel import Engine, context, design


def save_evidence(work, stage, outer, inner, fit, held, pairs, engine, manifest):
    path = work / stage / f"{outer}-{inner}"
    path.mkdir(parents=True, exist_ok=False)
    drug, cell = stack(pairs)
    for name, frame in (("drug", drug), ("cell", cell)):
        frame.to_parquet(path / (name + ".parquet"), index=False)
    diagnostics = [(str(config), steps, residual) for config, (steps, residual) in engine.diagnostics.items()]
    publish_json(path / "receipt.json", {"stage": stage, "outer": outer, "inner": inner,
        "fit_rows": len(fit), "eval_rows": len(held), "fit_parents": fit[KEY].nunique(),
        "eval_parents": held[KEY].nunique(), "outer_query_labels_read": 0,
        "study_manifest_sha256": sha256(manifest), "solver": diagnostics,
        "files": {name + ".parquet": sha256(path / (name + ".parquet")) for name in ("drug", "cell")}})


def evaluate(work, root, manifest, stage, index):
    outer, inner = divmod(index, 3)
    job = read_json(manifest)
    device = torch.device("cuda")
    data, assets, registry, coverage, _ = context(root, device)
    fit, check = subfold(registry, outer, inner)
    training, labels = response(fit, labels=True), response(check, labels=True)
    query = labels[META]
    engine = Engine(data, assets, training, query, registry)
    assessor = Evaluator(query.assign(truth=native_sensitivity(labels)), registry.scaffold_block, inner,
                         interaction=job.get("interaction", False))
    observed = query[KEY].map(coverage.lincs_modz24).to_numpy(dtype=bool)
    splits = {"all": None, "lincs": observed, "unobserved": ~observed}
    pairs = []
    print(f"{stage} {outer}-{inner} {torch.cuda.get_device_name(0)} {len(job[str(outer)])} candidates",
          flush=True)

    def score(pred, name, supports):
        values = engine.extract_rows(pred)
        for label, subset in supports.items():
            pairs.extend(assessor.score(name, values, subset, support=label))

    score(engine._c1, "C1", splits)
    score(engine.v1(), "H2-v1", splits)
    additional = {}
    for item in job[str(outer)]:
        if "support" in item:
            kind, recipe, axis = item["support"]
            key = f"{kind}/{recipe}/{axis}"
            if key not in additional:
                additional[key] = engine.support(kind, recipe=recipe, cell=axis == "cell")
    if stage == "view_controls":
        source_tissue = pd.read_parquet(V1_ASSETS / "cells.parquet", columns=["tissue"]).tissue
        tissue = query.cell_id.map(source_tissue)
        drivers = {"EGFR", "ERBB2", "ALK", "MET", "KIT", "RET", "BRAF", "KRAS", "MAP2K1", "MTOR",
                   "PIK3CA", "CDK4", "CDK6", "FLT3", "VEGFA", "KDR", "PARP1", "BCL2", "MDM2",
                   "ABL1", "FGFR1", "FGFR2", "FGFR3", "ROS1", "NTRK1", "BTK", "JAK2"}
        mechanism = {key for key in query[KEY].unique() if drivers & set(data.targets.get(key, ()))}
        substrates = {"target_annotation": query[KEY].isin(mechanism).to_numpy()}
        substrates |= {f"tissue/{x}": tissue.eq(x).to_numpy() for x in tissue.value_counts().index}
    else:
        substrates = {}
    for i, item in enumerate(job[str(outer)]):
        spec = design(item["design"])
        seed = 145993 + 100 * outer + inner
        pred = engine.solve(spec, seed=seed)
        modes = {"all": None, "lincs": observed}
        if item.get("all_supports"):
            modes |= additional
        elif "support" in item:
            kind, recipe, axis = item["support"]
            key = f"{kind}/{recipe}/{axis}"
            modes[key] = additional[key]
        score(pred, item["id"], modes)
        if substrates and (item["axis"] == "real" or item["id"] == "R1"):
            diagnostics = {f"ablation/{key}": inspected for key, inspected in substrates.items()
                           if np.count_nonzero(inspected) >= 30}
            score(pred, item["id"], diagnostics)
        if i % 15 == 0:
            print(f"{stage} {outer}-{inner} {i+1}/{len(job[str(outer)])}", flush=True)
    save_evidence(work, stage, outer, inner, training, labels, pairs, engine, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("stage")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    torch.set_num_threads(6)
    if not torch.cuda.is_available():
        raise RuntimeError("H2-v2 fitting requires a Slurm GPU")
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) if args.index is None else args.index
    evaluate(args.work, args.root, args.manifest, args.stage, idx)
