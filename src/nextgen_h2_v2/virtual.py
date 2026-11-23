"""Generate scaffold-cross-fitted virtual L1000 signatures from molecule inputs."""
import argparse
import itertools
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from nextgen_h2.assets import KEY, fold_ids
from nextgen_h2.metrics import Evaluator
from nextgen_h2.run import META, response, subfold
from vc_build.external import native_sensitivity
from vc_build.io import publish_json

from .inner import save_evidence
from .kernel import Design, Engine, View, context
from .search import common_score, expand_axis, ranker, score_index, stage_input, write_manifest


@dataclass(frozen=True)
class Generator:
    method: str
    regularization: float
    width: int = 256


def train_generator(x, y, spec, seed):
    center = y.mean(dim=0)
    if spec.method == "ridge":
        gram = x @ x.T
        regularization = spec.regularization
        lhs = gram + torch.eye(len(x), device=x.device) * regularization
        coefficients = x.T @ torch.linalg.solve(lhs, y - center)
        return lambda q: q @ coefficients + center
    torch.manual_seed(seed)
    scale = y.std(dim=0, correction=0).clamp_min(.2)
    model = nn.Sequential(nn.Linear(x.shape[1], spec.width), nn.LayerNorm(spec.width), nn.GELU(),
                          nn.Dropout(spec.regularization), nn.Linear(spec.width, spec.width),
                          nn.GELU(), nn.Linear(spec.width, y.shape[1])).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 70)
    target = (y - center) / scale
    gen = torch.Generator(device=x.device).manual_seed(seed)
    for _ in range(70):
        model.train()
        for chunk in torch.randperm(len(x), generator=gen, device=x.device).split(128):
            loss = nn.functional.smooth_l1_loss(model(x[chunk]), target[chunk])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()

    @torch.no_grad()
    def predict(q):
        return model(q) * scale + center

    return predict


class VirtualLibrary:
    def __init__(self, engine, seed):
        self.engine, self.seed = engine, seed
        columns = engine.data.profiles["modz24"].reindex(engine.drugs)
        self.observed = torch.as_tensor(columns.notna().all(axis=1).to_numpy(),
                                        device=engine.device)
        self.target = torch.as_tensor(columns.fillna(0.).to_numpy(dtype=np.float32), device=engine.device)
        self.source = engine.drug.explicit
        self.assign = fold_ids(engine.reg.loc[engine.fit_drugs, "scaffold_block"])
        self.cache = {}

    def matrix(self, params):
        if params in self.cache:
            return self.cache[params]
        source, y, engine = self.source, self.target, self.engine
        n = engine.stop
        matrix = torch.empty_like(y)
        for t in range(3):
            left = torch.as_tensor(self.assign.eq(t).to_numpy(), device=engine.device)
            known = ~left & self.observed[:n]
            model = train_generator(source[:n][known], y[:n][known], params, self.seed + t)
            matrix[:n][left] = model(source[:n][left])
        known = self.observed[:n]
        model = train_generator(source[:n][known], y[:n][known], params, self.seed + 75)
        matrix[n:] = model(source[n:])
        if not torch.isfinite(matrix).all():
            raise ValueError("Nonfinite virtual L1000 estimate")
        self.cache[params] = matrix
        return matrix

    def arm(self, spec, weight, control, *, retain=False):
        matrix = self.matrix(spec)
        x = matrix[:self.engine.stop]
        value = (matrix - x.mean(dim=0)) / x.std(dim=0, correction=0).clamp_min(.01)
        value = nn.functional.normalize(value, dim=1)
        view = View("virtual", f"{spec.method}-{spec.regularization:g}-{spec.width}", weight=weight, control=control)
        projected = value, torch.ones(len(value), device=self.engine.device, dtype=torch.bool)
        self.engine._features[("drug", view.name, view.recipe, view.transform, view.dimension)] = projected
        baseline = Design(lincs=View("lincs", weight=0., mask=0.))
        param = replace(baseline, drug_views=(view,))
        return self.engine.solve(param, seed=self.seed, retain=retain)


def search_space():
    choice = [Generator("ridge", alpha) for alpha in (.15, .6, 2.4, 9.6)]
    choice += [Generator("mlp", dropout, width) for dropout, width in itertools.product((.1, .3), (256, 512))]
    return [row(spec, weight) for spec, weight in itertools.product(choice, (.25, 1., 4., 16.))]


def row(spec, weight, control="real", name=None):
    return {"id": name or f"virtual/{spec.method}/{spec.regularization:g}/{spec.width}/{weight:g}",
            "generator": spec.__dict__, "weight": weight, "control": control}


def measure(work, root, contract, stage, index):
    outer, inner = divmod(index, 3)
    from nextgen_h2.run import read_json
    schedule = read_json(contract)[str(outer)]
    data, assets, registry, coverage, _ = context(root, torch.device("cuda"))
    a, b = subfold(registry, outer, inner)
    train, evaluated = response(a, True), response(b, True)
    query = evaluated[META]
    engine = Engine(data, assets, train, query, registry)
    e = Evaluator(query.assign(truth=native_sensitivity(evaluated)), registry.scaffold_block, inner)
    lib = VirtualLibrary(engine, 49128 + 100 * outer + inner)
    seen = query[KEY].map(coverage.lincs_modz24).to_numpy(dtype=bool)
    pairs = []
    for item in [{"id": "C1"}, *schedule]:
        predicted = (engine._c1 if item["id"] == "C1" else lib.arm(
            Generator(**item["generator"]), item["weight"], item["control"]))
        for name, subset in (("all", None), ("unobserved", ~seen)):
            pairs.extend(e.score(item["id"], engine.extract_rows(predicted), subset, support=name))
    save_evidence(work, stage, outer, inner, train, evaluated, pairs, engine, contract)


def vary(work, contracts, out, stages):
    from nextgen_h2.run import read_json
    reports, next_plan, controls, selections = [], [], [], {}
    for k in range(3):
        candidates, tables = [], []
        for stage in stages:
            ss, _, table = stage_input(work, contracts, stage, k)
            candidates.extend(x for x in ss if x["id"] != "incumbent")
            tables.append(table)
        rows = pd.concat(tables, ignore_index=True)
        scores = score_index(rows.drop_duplicates(["support", "axis", "arm"], keep="last"))
        winner = max(candidates, key=lambda x: ranker(scores, x["id"], "unobserved"))
        g = Generator(**winner["generator"])
        had = [r["weight"] for r in candidates if r["generator"] == winner["generator"]]
        new = [row(g, w) for w in expand_axis(had, winner["weight"])]
        if g.method == "ridge":
            grid = [x["generator"]["regularization"] for x in candidates if x["generator"]["method"] == "ridge"
                    and x["weight"] == winner["weight"]]
            new.extend(row(replace(g, regularization=alpha), winner["weight"])
                       for alpha in expand_axis(grid, g.regularization))
        next_plan.append(new)
        controls.append([row(g, winner["weight"], control, "V/" + control) for control in
                         ("real", "mask", "shuffle")])
        reports.append(rows)
        selections[str(k)] = winner
    out.mkdir(exist_ok=False)
    pd.concat(reports).to_csv(out / "points.tsv", sep="\t", index=False, na_rep="NA")
    publish_json(out / "winner.json", selections | {"settled": not any(next_plan)})
    if any(next_plan):
        for k, items in enumerate(next_plan):
            if not items:
                items.append(selections[str(k)] | {"id": "incumbent"})
        write_manifest(out / "next.json", next_plan)
    else:
        write_manifest(out / "virtual_controls.json", controls)


def gates(work, contracts, out):
    final = {}
    for k in range(3):
        specimens, unit, _ = stage_input(work, contracts, "virtual_controls", k)
        proof = {x: common_score(unit, x, "V/real", "unobserved") for x in ("C1", "V/mask", "V/shuffle")}
        clears = proof["C1"]["delta"] >= .01 and all(x["lo"] > 0 for x in proof.values())
        final[str(k)] = {"spec": specimens[0], "pass": bool(clears), "paired": proof}
    publish_json(out, final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "measure", "vary", "gates"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--contracts", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stages", nargs="+")
    a = parser.parse_args()
    if a.action == "register":
        schedule = search_space()
        write_manifest(a.out, [schedule] * 3)
    elif a.action == "measure":
        torch.set_num_threads(6)
        [stage] = a.stages
        measure(a.work, a.root, a.manifest, stage, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif a.action == "vary":
        vary(a.work, a.contracts, a.out, a.stages)
    else:
        gates(a.work, a.contracts, a.out)
