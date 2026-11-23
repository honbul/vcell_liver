"""Inner model-family search, boundary expansion and diagnostics."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from nextgen_h1_anchor.inputs import json_write
from nextgen_h1_anchor.metrics import Retrieval
from vc_build.io import sha256

from .boost import fit_boost, region
from .data import BUDGETS, TASKS, within_drug
from .kernels import KernelModel


def key(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def template(family):
    if family == "A":
        return dict(family="A", method="rbf", ridge=.3, width=1., rna_weight=.25)
    if family in {"B", "D"}:
        cfg = dict(family=family, objective="dual", lr=.05, leaves=15, depth=5, children=12,
                   subsample=.8, cols=.8, l2=1., rounds=80, blend=.35)
        if family == "D":
            return cfg | dict(objective="rank", stage=dict(ridge=.3, alpha=.25, prior="pan", clip=3.),
                              top=12, blend=.30)
        return cfg
    return dict(family="C", hidden=48, heads=4, dropout=.25, decay=.001, lr=.001,
                rank_weight=.35, rho_weight=.2, rank_objective="listnet", epochs=100)


def sweep(base, dimensions):
    return [base, *(base | {parameter: value}
                    for parameter, values in dimensions.items() for value in values if value != base[parameter])]


def seed_grid(family, dual):
    b = template(family)
    if family == "A":
        eb = dict(family="A", method="eb", prior="pan", ridge=.3, alpha=.25)
        rows = sweep(eb, {"ridge": (.003, .03, 3., 100.), "alpha": (0., .5, .75, .95), "prior": ("chem",)})
        axes = {"ridge": (.003, .03, 3., 100.), "width": (.05, .25, 4., 16.)}
        if dual:
            axes["rna_weight"] = (0., .5, .75, .95)
        rows += sweep(b, axes)
        p = b | {"method": "product"}
        rows += sweep(p, {"ridge": (.003, 3., 100.), "width": (.05, 16.),
                          **({"rna_weight": (0., .95)} if dual else {})})
        return rows
    if family in {"B", "D"}:
        rows = sweep(b, {"lr": (.02, .12), "leaves": (7, 31), "depth": (3, 8), "children": (5, 25),
                         "subsample": (.55, 1.), "cols": (.55, 1.), "l2": (.05, 8.), "rounds": (35, 150)})
        if family == "B":
            rows += [b | {"objective": target, "blend": weight}
                     for target, weight in (("l2", 0.), ("huber", 0.), ("rank", 1.), ("dual", .15), ("dual", .65))]
        else:
            rows += sweep(b, {"top": (6, 20), "blend": (.1, .55)})[1:]
            rows.extend(b | {"stage": b["stage"] | setting} for setting in (
                {"ridge": .03}, {"ridge": 3.}, {"prior": "chem"}, {"alpha": .65}))
        return rows
    return sweep(b, {"hidden": (24, 96), "heads": (2, 8),
                     "dropout": (.1, .4), "decay": (1e-5, .01), "lr": (.0003, .003),
                     "rank_weight": (0., .9), "rho_weight": (0., .5), "epochs": (50, 150),
                     "rank_objective": ("pairwise",)}) + [
        b | {"hidden": 24, "dropout": .4, "decay": .01},
        b | {"hidden": 96, "dropout": .1, "decay": 1e-5},
    ]


DOMAINS = {
    "ridge": (.00003, 1000.), "width": (.003, 256.), "alpha": (0., 1.), "rna_weight": (0., 1.),
    "lr": (.00004, .5), "leaves": (2, 127), "depth": (1, 32), "children": (2, 128),
    "subsample": (.12, 1.), "cols": (.12, 1.), "l2": (.0001, 100.), "rounds": (12, 360),
    "blend": (0., 1.), "top": (5, 60), "hidden": (8, 256), "heads": (1, 16), "dropout": (0., .8),
    "decay": (1e-7, .25), "rank_weight": (0., 4.), "rho_weight": (0., 2.), "epochs": (25, 400),
}


def extrapolate(c, tested, dual):
    """Check both exposed edges around this specific method and objective."""
    subset = [cfg for cfg in tested if all(cfg.get(f) == c.get(f) for f in ("family", "method", "objective"))]
    if c["family"] == "B" and len(subset) < 3:
        # Give one-objective arms the architecture bounds screened in the
        # shared B design as well as their objective-specific local search.
        subset += [cfg | {"objective": c["objective"], "blend": c["blend"]}
                   for cfg in tested if cfg.get("objective") == "dual"]
    rows, edges = [], []
    for name, (mn, mx) in DOMAINS.items():
        if name not in c or (name == "rna_weight" and not dual) or (
                name == "blend" and c.get("objective") in {"l2", "huber", "rank"} and c["family"] == "B"):
            continue
        values = [cfg[name] for cfg in subset if name in cfg]
        value = c[name]
        for target, is_edge in ((max(mn, value / 2), value <= min(values)),
                                (min(mx, value * 2) if value else max(mn, .05), value >= max(values))):
            if not is_edge:
                continue
            if isinstance(value, int):
                target = int(target)
            if name == "hidden":
                target = max(c.get("heads", 1), round(target / c.get("heads", 1)) * c.get("heads", 1))
            if name == "heads" and c["hidden"] % target:
                pool = [h for h in range(1, 17) if c["hidden"] % h == 0]
                target = min(pool, key=lambda h: abs(target - h))
            entry = {"parameter": name, "winner": value, "low": min(values), "high": max(values), "expanded_to": target}
            if target == value:
                entry["status"] = "domain-limit"
            else:
                entry["status"] = "expanded"
                rows.append(c | {name: target})
            edges.append(entry)
    if c["family"] == "D":
        stage = c["stage"]
        for param, bounds in {"ridge": (.003, 100.), "alpha": (0., 1.)}.items():
            values = [cfg["stage"][param] for cfg in subset]
            val = stage[param]
            if val == min(values) and val > bounds[0]:
                rows.append(c | {"stage": stage | {param: max(bounds[0], val / 2)}})
            if val == max(values) and val < bounds[1]:
                rows.append(c | {"stage": stage | {param: min(bounds[1], val * 2)}})
    return rows, edges


@dataclass
class Search:
    states: list
    mode: str
    policy: str
    device: str

    def __post_init__(self):
        self.dual = self.mode == "dual"
        self.cache = [{} for _ in self.states]
        self.evaluations = {}
        self.predictions = {}
        self.curves = {}

    def score(self, cfg, phase):
        tag = key(cfg)
        if tag in self.evaluations:
            return self.evaluations[tag]
        folds, curves, predictions = [], [], []
        for j, (ctx, sub, orders, _, _) in enumerate(self.states):
            order = orders[self.policy]
            q = region(order, 5, len(ctx.mean))
            model = fit_model(ctx, sub, order, self.dual, cfg, self.device, self.cache[j])
            scores = model.predict(ctx.valid[:, order[:5]], 5)
            good = eligible(ctx, order, q, 5)
            y, p = ctx.valid[good][:, q], scores[good][:, q]
            retrieval = np.nanmean(Retrieval(y)(p), axis=0)
            stats = dict(inner_fold=j, n=int(good.sum()), q=len(q), ndcg_at_5=float(retrieval[0]),
                         precision_at_5=float(retrieval[1]), rho=within_drug(y, p),
                         mae=float(np.nanmean(np.abs(y - p))))
            stats["objective"] = stats["ndcg_at_5"] + (.12 if ctx.study == "LICOB" else .06) * stats["rho"]
            if cfg["family"] == "C":
                stats["best_epoch"] = model.best_epoch
            elif cfg["family"] in {"B", "D"}:
                stats["best_epoch"] = max(h.best_iteration for h in model.heads.values())
            folds.append(stats)
            predictions.append((q, good, p))
            curves.append(getattr(model, "curves", {}))
        entry = dict(config=cfg, phase=phase, folds=folds, objective=float(np.mean([x["objective"] for x in folds])),
                     ndcg_at_5=float(np.mean([x["ndcg_at_5"] for x in folds])),
                     rho=float(np.mean([x["rho"] for x in folds])))
        self.evaluations[tag] = entry
        self.predictions[tag] = predictions
        if cfg["family"] != "A":
            self.curves[tag] = curves
        return entry

    def winner(self):
        return max(self.evaluations.values(), key=lambda e: (e["objective"], e["ndcg_at_5"]))


def eligible(ctx, order, q, k):
    require = order[:12 if k == 12 else 8]
    return np.isfinite(ctx.valid[:, require]).all(1) & (np.isfinite(ctx.valid[:, q]).sum(1) >= 15)


def fit_model(ctx, sub, order, dual, cfg, device, cache):
    family = cfg["family"]
    if family == "A":
        return KernelModel(ctx, order, cfg, dual)
    if family in {"B", "D"}:
        return fit_boost(ctx, sub, order, dual, cfg, cache)
    from .neural import fit_neural
    return fit_neural(ctx, order, dual, cfg, device, epochs=cfg.get("refit_epochs") if ctx.valid is None else None)


def cache_read(directory):
    stamp = json.loads((directory / "round1.json").read_text())
    artifact = directory / "study.pkl"
    if sha256(artifact) != stamp["cache_sha256"] or stamp["query_labels_opened"]:
        raise ValueError("Round 1 cache SHA or outcome custody failed")
    with artifact.open("rb") as stream:
        return pickle.load(stream)


def diag(ctx, predictions, q, good, moa, names, modality):
    actual = ctx.valid[good][:, q]
    error = np.abs(predictions - actual)
    result = []
    tissue = ctx.tissue[good]
    for kind, groups in (("tissue", tissue), ("mechanism", np.array(moa)[q]), ("drug", np.array(names)[q])):
        for name in sorted(set(groups)):
            if kind == "drug" and name.lower() not in ("sorafenib", "lenvatinib", "regorafenib", "cabozantinib"):
                continue
            select = groups == name
            selected = error[select] if kind == "tissue" else error[:, select]
            if np.isfinite(selected).any():
                result.append(dict(group_type=kind, group=name, n=int(np.isfinite(selected).sum()),
                                   mae=float(np.nanmean(selected)), modality=modality))
    return result


def budget_diagnostics(search, config, moa, names):
    curves, errors = [], []
    for fold, (ctx, sub, orders, _, _) in enumerate(search.states):
        order = orders[search.policy]
        fitted = fit_model(ctx, sub, order, search.dual, config, search.device, search.cache[fold])
        for k in BUDGETS:
            q = region(order, k, len(ctx.mean))
            good = eligible(ctx, order, q, k)
            if good.sum() < 2:
                continue
            packet = ctx.valid[:, order[:k]]
            p = fitted.predict(packet, k)[good][:, q]
            actual = ctx.valid[good][:, q]
            judge = Retrieval(actual)
            n, precision = np.nanmean(judge(p), axis=0)
            curves.append(dict(fold=fold, k=k, n=int(good.sum()), q=len(q), ndcg=float(n),
                               rho=within_drug(actual, p), precision=float(precision), noise_sd=0.))
            if k == 5:
                errors.extend(dict(fold=fold, **row) for row in diag(ctx, p, q, good, moa, names, search.mode))
            if ctx.study == "LICOB" and k in {3, 5, 8}:
                for spread in (.1, .25, .5):
                    rng = np.random.default_rng(74031 + fold)
                    measured = []
                    for _ in range(25):
                        jitter = rng.normal(0, spread, size=packet.shape) * ctx.sd[order[:k]]
                        noisy = fitted.predict(packet + jitter, k)[good][:, q]
                        measured.append((np.nanmean(judge(noisy)[:, 0]), within_drug(actual, noisy)))
                    values = np.asarray(measured)
                    curves.append(dict(fold=fold, k=k, n=int(good.sum()), q=len(q), noise_sd=spread,
                                       ndcg=float(values[:, 0].mean()), rho=float(values[:, 1].mean()),
                                       ndcg_p10=float(np.quantile(values[:, 0], .10))))
    return curves, errors


def train(task, family, r1, out, device):
    states, policies, _, names, moa = cache_read(r1 / task)
    out.mkdir(parents=True)
    answer = {}
    for mode, policy in policies.items():
        search = Search(states, mode, policy, device)
        for cfg in seed_grid(family, search.dual):
            search.score(cfg, "round2")
        start = search.winner()
        print(task, family, mode, "round2", round(start["objective"], 6), flush=True)
        boundaries = []
        for iteration in range(3):
            c = search.winner()["config"]
            candidates, bound = extrapolate(c, [r["config"] for r in search.evaluations.values()], search.dual)
            boundaries.append(dict(iteration=iteration, config=c, edges=bound))
            unique = {key(v): v for v in candidates if key(v) not in search.evaluations}
            if not unique:
                break
            for cfg in unique.values():
                search.score(cfg, f"round3-expansion-{iteration+1}")
            if key(search.winner()["config"]) == key(c):
                break
        # A single targeted refinement, selected from the observed train/validation gap.
        winner = search.winner()
        cfg = winner["config"]
        contrast = search.curves.get(key(cfg), [])
        if family == "C":
            gap = float(np.mean([next(x["train_ndcg"] - x["val_ndcg"] for x in curve
                                      if x["epoch"] == f["best_epoch"]) for curve, f in zip(contrast, winner["folds"])]))
            extra = cfg | ({"dropout": min(.75, cfg["dropout"] + .1), "decay": cfg["decay"] * 2, "noise": .10}
                           if gap > .02 else {"hidden": min(256, cfg["hidden"] * 2), "noise": .05})
        elif family in {"B", "D"}:
            gaps = []
            for block in contrast:
                head = next(iter(block.values()))
                metric = next(iter(head["inner_val"]))
                if metric == "ndcg@5":
                    gaps.append(head["train"][metric][-1] - head["inner_val"][metric][-1])
                else:
                    gaps.append((head["inner_val"][metric][-1] - head["train"][metric][-1])
                                / max(1e-8, head["inner_val"][metric][0]))
            gap = float(np.mean(gaps))
            extra = cfg | ({"children": min(128, cfg["children"] * 2), "l2": cfg["l2"] * 2}
                           if gap > .04 else {"lr": cfg["lr"] * .5, "rounds": min(360, cfg["rounds"] * 2)})
        else:
            gap = None
            extra = cfg | {"clip": 2.5}
        search.score(extra, "round3-error-refinement")
        selected = search.winner()
        budgets, errors = budget_diagnostics(search, selected["config"], moa, names)
        answer[mode] = dict(policy=policy, round2=start, selected=selected, grid=list(search.evaluations.values()),
                            saturation=boundaries, learning_curves=search.curves, budget=budgets, errors=errors,
                            train_validation_gap=gap, refinement=extra)
        print(task, family, mode, "round3", round(selected["objective"], 6), selected["config"], flush=True)
    json_write(out / "result.json", dict(task=task, family=family, device=device, modes=answer,
                                        round1_sha256=sha256(r1 / task / "round1.json"),
                                        outer_labels_opened=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("round1", type=Path)
    cli.add_argument("out", type=Path)
    cli.add_argument("selectors")
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--task")
    args = cli.parse_args()
    from os import environ
    if args.task:
        task, family = args.task, args.selectors
    else:
        index = int(environ["SLURM_ARRAY_TASK_ID"])
        task, family = TASKS[index % len(TASKS)], args.selectors[index // len(TASKS)]
    train(task, family, args.round1, args.out / task / family, args.device)
