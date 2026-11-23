"""Seal v1-matched forecasts and independently replay learned H2-v2 inference."""
import argparse
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nextgen_h2.assets import KEY, parquet_lock
from nextgen_h2.replay import network_deny
from nextgen_h2.run import META, read_json, response, subfold
from vc_build.io import publish_json, sha256

from .assets import V1_RUN
from .kernel import Engine, context, design
from .nonlinear import Candidate
from .portable import create_kit
from .residual_run import FittedResidual
from .virtual import Generator, VirtualLibrary

ARMS = ("C1", "H2-v1", "H2-R1", "H2-R2a", "H2-R2b", "H2-R3-GBDT", "H2-R3-Neural",
        "H2-v2", "Structure-C1", "H2-StructOnly")
V1_SHA = (
    "a31f0e51952af872e58396c951bc8829b6580c5bc8c3b833c0bb8b183b47ac41",
    "182c3a5e175e1157f95e416160e2bf58c23a4f4f225ee8fd63e446fa8398960a",
    "af060ebef01c9dd34c04f128b989b2f45d3193dcdd76c050dcd61e32fefe9565",
    "70db278d12558129e604086e994a39cf685fe99bd7e082b1d0cbe182f14a7bca",
)


class Bundle:
    def __init__(self, engine, exact):
        self.query = engine.query.reset_index(drop=True)
        self.selector = engine.selector
        self.c1 = engine._c1.cpu()
        self.v1 = exact
        self.checkpoints = {}
        self.residuals = {}
        self.virtual = None
        self.final = "H2-R2b"

    def predict_kernel(self, name, device, cache):
        if name not in cache:
            p, fallback = self.checkpoints[name]
            active = p["active"].to(device)
            recorded = {key: p[key].to(device) for key in ("anchor", "cell_cross", "coef", "dcross")}
            result = recorded["anchor"] + (recorded["cell_cross"] @ recorded["coef"]) @ recorded["dcross"].T
            if p["apply_c1"]:
                result = torch.where(active[None], result, self.c1.to(device))
            for key in ("extra_drug", "cell_coverage"):
                if p[key] is not None:
                    observed = p[key].to(device)
                    ok = observed[None] if key == "extra_drug" else observed[:, None]
                    alternate = self.predict_kernel(fallback, device, cache) if fallback else self.c1.to(device)
                    result = torch.where(ok, result, alternate)
            cache[name] = result
        return cache[name]

    def apply(self, device):
        raw, cache = {}, {}
        raw["C1"] = self.c1.numpy()[self.selector]
        raw["H2-v1"] = self.v1
        for name in ("H2-R1", "H2-R2a", "H2-R2b", "H2-StructOnly"):
            raw[name] = self.predict_kernel(name, device, cache).cpu().numpy()[self.selector]
        reference = raw["H2-R2b"]
        for family, record in self.residuals.items():
            model, features, weight = record
            if family == "gbdt":
                ft = features.tabular(self.query, reference, False)
                _, drug = features.coordinates(self.query, False)
                offset = np.where(features.drug_seen[drug], model.predict(ft), 0.)
            else:
                offset = model.predict(features, self.query, reference, False, device)
            name = {"gbdt": "H2-R3-GBDT", "neural": "H2-R3-Neural"}[family]
            raw[name] = reference + weight * offset
        raw["H2-v2"] = raw[self.final]
        raw["Structure-C1"] = raw["C1"].copy()
        return raw


def freeze(work, root, decision, index):
    policy = read_json(decision / "champion.json")[str(index)]
    lock = sha256(decision / "champion.json")
    if read_json(decision / "receipt.json")["selection_sha256"] != lock:
        raise ValueError("Unregistered H2-v2 champion")
    device = torch.device("cuda")
    data, assets, reg, coverage, external = context(root, device, external=index == 3)
    if index == 3:
        fit = response(reg.index, True)
        query = external[META].sort_values("row_id", ignore_index=True)
    else:
        a, b = subfold(reg, index)
        fit, query = response(a, True), response(b, False)
    frozen = parquet_lock(V1_RUN / "forecasts" / str(index) / "predictions.parquet", V1_SHA[index])
    if not query[META].equals(frozen[META]):
        raise ValueError("H2-v2 outer evaluation keys differ from v1")
    engine = Engine(data, assets, fit, query, reg)
    pinned = torch.as_tensor(frozen.C1.to_numpy(np.float32), device=device)
    engine._c1[engine.selector] = pinned
    model = Bundle(engine, frozen.champion.to_numpy(np.float32))
    seed = 75201 + index
    for name, upstream in (("R1", None), ("R2a", "H2-R1"), ("R2b", "H2-R2a")):
        params = design(policy[name]["design"])
        _, state = engine.solve(params, seed, retain=True)
        model.checkpoints["H2-" + name] = (state, upstream)
    g = VirtualLibrary(engine, seed)
    v = policy["virtual"]["spec"]
    _, state = g.arm(Generator(**v["generator"]), v["weight"], "real", retain=True)
    model.checkpoints["H2-StructOnly"] = (state, None)
    kernel = design(policy["R2b"]["design"])
    common = FittedResidual(engine, kernel, seed)
    for family in ("gbdt", "neural"):
        spec = policy["R3"][family]
        fitted, _ = common.train(Candidate(**spec["spec"]))
        if family == "neural":
            fitted.model.cpu()
        model.residuals[family] = (fitted, common.feature, spec["scale"])
    model.final = {"kernel": "H2-R2b", "gbdt": "H2-R3-GBDT", "neural": "H2-R3-Neural"}[policy["final_family"]]
    seen = query[KEY].map(coverage.lincs_modz24).to_numpy(dtype=bool)
    panel = query.assign(lincs_observed=seen)
    assembled = model.apply(device)
    if not np.array_equal(assembled["C1"], frozen.C1.to_numpy()):
        raise ValueError("Frozen C1 changed")
    dest = work / "forecasts" / str(index)
    dest.mkdir(exist_ok=False, parents=True)
    for family, rec in model.residuals.items():
        if family == "neural":
            rec[0].model.cpu()
    with (dest / "bundle.pkl").open("xb") as stream:
        pickle.dump(model, stream, protocol=5)
    if index == 3:
        kit = create_kit(engine, design(policy["R1"]["design"]), seed)
        with (dest / "portable.pkl").open("xb") as handle:
            pickle.dump(kit, handle, protocol=5)
    panel.assign(**assembled).to_parquet(dest / "forecast.parquet", index=False)
    publish_json(dest / "receipt.json", {"part": index, "query_outcomes_loaded": 0, "rows": len(panel),
                "source_rows": len(fit), "v1_sha256": V1_SHA[index], "selection_sha256": lock,
                "family": policy["final_family"], "all_rows_structure_policy": "no measured perturbation",
                "files": {f.name: sha256(f) for f in sorted(dest.iterdir())}})


def replay(work, index, device):
    sys.addaudithook(network_deny)
    dest = work / "forecasts" / str(index)
    receipt = read_json(dest / "receipt.json")
    file = dest / "bundle.pkl"
    if sha256(file) != receipt["files"][file.name]:
        raise ValueError("The offline inference bundle changed")
    expected = parquet_lock(dest / "forecast.parquet", receipt["files"]["forecast.parquet"])
    with file.open("rb") as handle:
        model = pickle.load(handle)
    actual = model.apply(device)
    delta = max(float(np.abs(expected[name].to_numpy() - actual[name]).max()) for name in ARMS)
    c1 = actual["C1"]
    structure = float(np.abs(actual["Structure-C1"] - c1).max())
    omics = model.checkpoints["H2-R2b"][0]["cell_coverage"]
    missing = np.ones(len(expected), dtype=bool) if omics is None else ~omics.numpy()[model.selector[0]]
    fallback = float(np.max(np.abs(actual["H2-R2b"][missing] - actual["H2-R2a"][missing]), initial=0))
    if not delta < 1e-6 or structure != 0. or fallback != 0.:
        raise ValueError(f"Offline replay or exact fallback failed {delta:g}, {structure:g}, {fallback:g}")
    publish_json(dest / "replay.json", {"part": index, "network_allowed": False, "max_abs_diff": delta,
                                        "structure_fallback_max_diff": structure,
                                        "RNA_fallback_rows": int(missing.sum()), "RNA_fallback_max_diff": fallback,
                                        "forecast_sha256": receipt["files"]["forecast.parquet"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "replay"))
    parser.add_argument("work", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--decision", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(6)
    part = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.stage == "predict":
        freeze(args.work, args.root, args.decision, part)
    else:
        replay(args.work, part, torch.device("cuda"))
