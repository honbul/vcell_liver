"""Test both nonlinear residual families in the fixed nested scaffold roles."""
import argparse
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nextgen_h2.assets import KEY
from nextgen_h2.metrics import Evaluator
from nextgen_h2.run import META, read_json, response, subfold
from vc_build.external import native_sensitivity
from vc_build.io import publish_json, sha256

from .inner import save_evidence
from .kernel import Engine, context, design
from .nonlinear import Candidate, fit_boost, fit_neural, search_space, stop_holdout
from .residual_data import ResidualFeatures, kernel_oof
from .search import common_score, ranker, score_index, stage_input, write_manifest

SCALES = (0., .00390625, .015625, .03125, .0625, .125, .25, .5, 1., 2., 4., 8., 16.)
STAGE = "resid_strict"


def manifest(kernel, target):
    selections = read_json(kernel)
    rows = []
    for f in range(3):
        spec = selections[str(f)]["R2b"]["design"]
        configurations = [{"id": f"{x.family}/{x.objective}/{x.width}/{x.depth}/{x.dropout:g}",
                          "parameters": asdict(x), "kernel": spec} for x in search_space()]
        rows.append(configurations)
    write_manifest(target, rows, interaction=True)


class FittedResidual:
    def __init__(self, engine, params, seed):
        self.engine = engine
        self.params = params
        self.seed = seed
        self.keep = stop_holdout(engine.fit, engine.reg.scaffold_block)
        self.training = kernel_oof(engine, params, seed, self.keep)
        self.feature = ResidualFeatures(engine, params, engine.fit[~self.keep])
        self.query = engine.query
        self.reference = engine.extract_rows(engine.solve(params, seed))
        self.train_matrix, self.query_matrix = None, None

    def train(self, spec):
        if spec.family == "gbdt":
            if self.train_matrix is None:
                self.train_matrix = self.feature.tabular(self.training, self.training.oof.to_numpy(), True)
            return fit_boost(self.training, self.train_matrix, self.keep, spec)
        return fit_neural(self.training, self.feature, self.keep, spec, self.engine.device, self.seed)

    def predict(self, model, family):
        if family == "gbdt":
            if self.query_matrix is None:
                self.query_matrix = self.feature.tabular(self.query, self.reference, False)
            _, drug = self.feature.coordinates(self.query, False)
            keep = self.feature.drug_seen[drug]
            return np.where(keep, model.predict(self.query_matrix), 0.)
        return model.predict(self.feature, self.query, self.reference, False, self.engine.device)


def measure(work, root, manifest_path, index):
    f, i = divmod(index, 3)
    seed = 17049 + f * 100 + i
    schedule = read_json(manifest_path)[str(f)]
    device = torch.device("cuda")
    data, assets, registry, coverage, _ = context(root, device)
    fitting, held = subfold(registry, f, i)
    fit, test = response(fitting, True), response(held, True)
    query = test[META]
    engine = Engine(data, assets, fit, query, registry)
    apparatus = FittedResidual(engine, design(schedule[0]["kernel"]), seed)
    assessor = Evaluator(query.assign(truth=native_sensitivity(test)), registry.scaffold_block, i,
                         interaction=True)
    known = query[KEY].map(coverage.lincs_modz24).to_numpy(dtype=bool)
    pairs, curves = [], []

    def score(name, p):
        for mask, h in (("all", None), ("lincs", known)):
            pairs.extend(assessor.score(name, p, h, support=mask))

    score("H2-Kernel-Max", apparatus.reference)
    for entry in schedule:
        spec = Candidate(**entry["parameters"])
        print(f"{f}-{i}: {entry['id']}", flush=True)
        model, curve = apparatus.train(spec)
        delta = apparatus.predict(model, spec.family)
        for weight in SCALES:
            score(entry["id"] + f"/{weight:g}", apparatus.reference + weight * delta)
        record = {"name": entry["id"], "family": spec.family, "data": curve}
        if spec.family == "gbdt":
            record["backend"] = model.backend
        curves.append(record)
    save_evidence(work, STAGE, f, i, fit, test, pairs, engine, manifest_path)
    path = work / STAGE / f"{f}-{i}"
    publish_json(path / "learning.json", {"fit_pool_oof_rows": len(apparatus.training),
                 "early_stop_rows": int(apparatus.keep.sum()),
                 "early_stop_parent_count": apparatus.training.loc[apparatus.keep, KEY].nunique(),
                 "early_stop_labels_available_to_crossfit_base": 0,
                 "early_stop_used_to_fit_residual_preprocessors": False,
                 "inner_query_labels_available_to_estimator": 0,
                 "curve": curves})


def decide(work, contracts, kernel, virtual, out):
    canonical = read_json(kernel)
    synthetic = read_json(virtual)
    chosen, stages, score_rows = {}, [], []
    flags = {"gbdt": [], "neural": []}
    for outer in range(3):
        specs, unit, summary = stage_input(work, contracts, STAGE, outer)
        scores = score_index(summary)
        ranking = {}
        for family in ("gbdt", "neural"):
            candidates = [(entry, weight) for entry in specs if entry["parameters"]["family"] == family
                          for weight in SCALES if weight > 0]
            entry, weight = max(candidates, key=lambda x: ranker(scores, x[0]["id"] + f"/{x[1]:g}", "all"))
            name = entry["id"] + f"/{weight:g}"
            effect = common_score(unit, "H2-Kernel-Max", name, "all")
            guard = common_score(unit, "H2-Kernel-Max", name, "all", "ndcg_at_5")
            win = effect["lo"] > 0 and guard["lo"] >= -.005
            flags[family].append(bool(win))
            ranking[family] = {"spec": entry["parameters"], "scale": weight, "rho": effect,
                               "ndcg": guard, "pass": bool(win),
                               "zero_residual_point_rho": ranker(scores, "H2-Kernel-Max", "all")[0]}
        stages.append(ranking)
        score_rows.append(summary)
        chosen[str(outer)] = canonical[str(outer)] | {"R3": ranking, "virtual": synthetic[str(outer)]}
    eligible = [family for family, passes in flags.items() if all(passes)]
    family = max(eligible, key=lambda f: np.mean([x[f]["rho"]["delta"] for x in stages]), default="kernel")
    for k in range(3):
        chosen[str(k)]["final_family"] = family
    # Independent v1-style majority across the inner-fit policies, with lowest-fold stable tie break.
    labels = [str(chosen[str(k)]["R2b"]["design"]) for k in range(3)]
    ext = max(range(3), key=lambda k: (labels.count(labels[k]), -k))
    chosen["3"] = chosen[str(ext)] | {"external_policy_from_outer_fit": ext}
    out.mkdir(exist_ok=False)
    pd.concat(score_rows).to_csv(out / "points.tsv", sep="\t", index=False, na_rep="NA")
    publish_json(out / "champion.json", chosen)
    publish_json(out / "receipt.json", {"selection_sha256": sha256(out / "champion.json"),
                  "final_family": family, "promotion_by_outer_fit": flags, "query_outcomes_opened": False,
                  "strict_stage": STAGE, "stage_manifest_sha256": sha256(contracts / (STAGE + ".json")),
                  "kernel_recipe_sha256": sha256(kernel), "virtual_recipe_sha256": sha256(virtual)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("manifest", "measure", "decide"))
    parser.add_argument("target", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--contracts", type=Path)
    parser.add_argument("--kernel", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--virtual", type=Path)
    a = parser.parse_args()
    if a.stage == "manifest":
        manifest(a.kernel, a.target)
    elif a.stage == "measure":
        torch.set_num_threads(6)
        measure(a.work, a.root, a.manifest, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        decide(a.work, a.contracts, a.kernel, a.virtual, a.target)
