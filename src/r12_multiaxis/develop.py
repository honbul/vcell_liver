"""Evaluate registered choices only in an outer fold's development partitions."""

import argparse
from dataclasses import asdict, replace
from functools import cached_property
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP
from src.r11_chemical_probe.metrics import Contrast

from .calibration import Calibration
from .candidates import (
    BASELINES, CONTEXTS, LOSSES, SHIFT_BASES, STRUCTURES, Candidate, Evaluator, expansion, registry, shift_base,
)
from .data import Fit, Source
from .split import balanced_partition


def candidate_dict(candidate):
    return asdict(candidate)


def definition(row):
    return Candidate(**row["candidate"])


def key(candidate):
    return tuple(asdict(candidate).values())


def settings(candidate):
    return (
        candidate.arm != "B0", candidate.network, candidate.head != "concat",
        candidate.context not in {"raw", "pca", "none"}, len(candidate.context),
        0 if candidate.arm == "C3_rank0" else candidate.vic, candidate.lr, -candidate.value,
        candidate.horizon, candidate.response_rank, candidate.context, candidate.arm,
    )


def best_rows(records, predicate, metric, tie):
    by_candidate = {}
    for row in records:
        if predicate(row):
            by_candidate.setdefault(key(definition(row)), []).append(row)
    choices = []
    for variants in by_candidate.values():
        use = [r for r in variants if r["metric"] == metric]
        if len(use) != 2 or {r["inner"] for r in use} != {0, 1}:
            continue
        score = float(np.average([r["score"] for r in use], weights=[r["groups"] for r in use]))
        candidate = definition(use[0])
        fixed = replace(candidate, epoch=int(np.median([r["best_epoch"] for r in use])))
        choices.append((score, fixed))
    if not choices:
        return None
    minimum = min(score for score, _ in choices)
    return min((x for x in choices if x[0] <= minimum + tie), key=lambda p: settings(p[1]))


def pick_shift(records, predicate, tie, metric="abs"):
    labels = sorted({(r["base"]["arm"], r["family"], str(r["penalty"])) for r in records if predicate(r)})
    scored = []
    for label in labels:
        use = [r for r in records if (r["base"]["arm"], r["family"], str(r["penalty"])) == label]
        if len(use) == 2:
            point = float(np.average([r[metric] for r in use], weights=[r["groups"] for r in use]))
            scored.append((point, use[0]))
    if not scored:
        return None
    minimum = min(s for s, _ in scored)
    return min([x for x in scored if x[0] <= minimum + tie],
               key=lambda x: (x[1]["base"]["arm"] not in {"S_base_B2", "S_base_B3", "S_base_B5"},
                              x[1]["family"], x[1]["penalty"] != "identity",
                              -(x[1]["penalty"] if x[1]["penalty"] != "identity" else 0),
                              x[1]["base"]["arm"]))


class CalibratorFits:
    def __init__(self, source, parent_fit, parent_rows, device, seed):
        self.parent = parent_fit
        self.source = source
        self.rows = parent_rows
        self.device = device
        self.seed = seed
        self.cache = {}

    @cached_property
    def children(self):
        labels = balanced_partition(self.rows, 2, self.seed)
        folded = self.rows[GROUP].map(labels).to_numpy()
        children = []
        for k in range(2):
            mask = folded == k
            child = Fit(self.source, self.rows.loc[~mask], self.rows.loc[mask], panel=self.parent.panel)
            children.append((Evaluator(child, self.device), mask))
        return children

    def oof(self, candidate, seed=42):
        label = (candidate, seed if candidate.network else 42)
        if label not in self.cache:
            pred = np.empty_like(self.parent.residual)
            anchor = np.empty_like(pred)
            runs = []
            for evaluate, mask in self.children:
                best, record, _, _ = evaluate.evaluate(candidate, seed=seed)
                pred[mask] = best["abs"][2]
                anchor[mask] = evaluate.fit.offset
                runs.append(record)
            self.cache[label] = (pred, anchor, runs)
        return self.cache[label]

    def fit_one(self, candidate, family, penalty):
        pred, anchors, _ = self.oof(candidate)
        fit = self.parent
        return Calibration(fit.train.rows, pred, anchors, fit.train.delta, fit.scale, penalty, family)


def dual_points(metric, pred, delta):
    contrast, absolute = metric.points(pred, delta)[:2]
    return {"abs": absolute, "contrast": contrast}


def add_grid(evaluator, candidates, fold, records, ledger, known):
    fit = evaluator.fit
    metric = Contrast(fit.query.rows, fit.source.cfg["minimum_groups"])
    for candidate in candidates:
        lookup = (fold, key(candidate))
        if lookup in known:
            continue
        pred, runtime, curve, _ = evaluator.evaluate(candidate, retain=False)
        if candidate.network and candidate.epoch == 0:
            values = {m: pred[m][0] for m in ("abs", "contrast")}
        else:
            values = dual_points(metric, pred["abs"][2], fit.query.delta)
        tagged = candidate_dict(candidate)
        for readout in ("abs", "contrast"):
            records.append(dict(candidate=tagged, inner=fold, metric=readout, score=values[readout],
                                best_epoch=pred[readout][1], groups=fit.query.rows[GROUP].nunique()))
        ledger.append(dict(phase="inner_grid", inner=fold, candidate=tagged, runtime=runtime, curve=curve))
        known.add(lookup)
        print(f"inner={fold} {candidate.arm} lr={candidate.lr:g} "
              f"contrast={values['contrast']:.6f} abs={values['abs']:.6f}", flush=True)


def add_shift(evaluator, cross, fold, records, ledger):
    fit = evaluator.fit
    metric = Contrast(fit.query.rows, fit.source.cfg["minimum_groups"])
    cfg = fit.cfg
    for arm in SHIFT_BASES:
        base = shift_base(arm, cfg)
        base_best, runtime, curve, _ = evaluator.evaluate(base)
        prediction = base_best["abs"][2]
        point = dual_points(metric, prediction, fit.query.delta)
        oof, anchor, cv_runs = cross.oof(base)
        ledger.append(dict(phase="inner_shift", inner=fold, candidate=candidate_dict(base), runtime=runtime,
                           curve=curve, calibration_runs=cv_runs, crossfit_prediction_rms=rms(oof),
                           refit_prediction_rms=rms(prediction)))
        families = ("S1",) if arm == "B2" else ("S1", "S2", "S3")
        for family in families:
            for penalty in cfg["calibration_penalties"]:
                model = Calibration(fit.train.rows, oof, anchor, fit.train.delta, fit.scale, penalty, family)
                adjusted = model.predict(fit.query.rows, prediction, fit.offset)
                scores = dual_points(metric, adjusted, fit.query.delta)
                records.append(dict(base=candidate_dict(base), family=family, penalty=penalty, inner=fold,
                                    groups=fit.query.rows[GROUP].nunique(), abs_before=point["abs"],
                                    contrast_before=point["contrast"], **scores))
        print(f"inner={fold} shift baseline {arm} complete", flush=True)


def rms(a):
    return float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))


def single_choices(records, shift, cfg):
    tie = cfg["tie"]
    results = {}
    for arm in sorted({r["candidate"]["arm"] for r in records}):
        selected = best_rows(records, lambda r: r["candidate"]["arm"] == arm, "contrast", tie)
        if selected is not None:
            results[arm] = selected
    winners = {}
    for name, arms, metric in [
        ("B_abs", BASELINES, "abs"), ("B_contrast", BASELINES, "contrast"),
        ("Context", CONTEXTS, "contrast"), ("Structure", STRUCTURES, "contrast"),
        ("Loss", LOSSES, "contrast"),
    ]:
        winners[name] = best_rows(records, lambda r: r["candidate"]["arm"] in arms, metric, tie)
        if winners[name] is None:
            raise ValueError(f"The required selector {name} has no complete inner comparisons")
    diag = {}
    for family in ("S1", "S2", "S3"):
        for arm in SHIFT_BASES:
            candidate = pick_shift(shift, lambda r: r["family"] == family and r["base"]["arm"] == f"S_base_{arm}"
                                   and r["penalty"] != "identity", tie)
            if candidate is not None:
                diag[f"{family}_{arm}"] = candidate
    official = pick_shift(shift, lambda r: r["penalty"] != "identity", tie)
    simple = diag["S1_B2"]
    neural = pick_shift(shift, lambda r: r["base"]["arm"] in {"S_base_B4", "S_base_A4"}
                        and r["penalty"] != "identity", tie)
    return results, winners, diag, official, simple, neural


def registered(fit, phase):
    choices = registry(fit)
    if phase == "scaffold":
        return [a for a in choices if not a.arm.startswith(("D1", "E"))]
    if phase == "core":
        mandatory = {*BASELINES, "C1", "C2", "C3", "C3_rank0", "A1", "A2", "A3", "A4", "A5", "A6",
                     "L1", "L2", "L3", "L5", "L6", "J1_FiLM_L2"}
        return [a for a in choices if a.arm in mandatory]
    return choices


def run_development(cfg, source, args, prior):
    start = time.monotonic()
    records = [] if prior is None else prior["records"]
    shift = [] if prior is None else prior["shift"]
    ledger = [] if prior is None else prior["ledger"]
    known = {(r["inner"], key(definition(r))) for r in records}
    fold_runs = range(2)
    for inner in fold_runs:
        fit_rows, query_rows = source.inner(inner)
        fit = Fit(source, fit_rows, query_rows, observe_query=True)
        eval_fit = Evaluator(fit, args.device)
        arms = registered(fit, args.phase)
        add_grid(eval_fit, arms, inner, records, ledger, known)
        if not any(r["inner"] == inner for r in shift):
            cross = CalibratorFits(source, fit, fit_rows, args.device, cfg["split_seed"] + inner + 100)
            add_shift(eval_fit, cross, inner, shift, ledger)
        del fit, eval_fit

    if args.phase != "core":
        tuning = cfg["training"]
        for inner in fold_runs:
            fit_rows, query_rows = source.inner(inner)
            fit = Fit(source, fit_rows, query_rows, observe_query=True)
            eval_fit = Evaluator(fit, args.device)
            arms = registered(fit, args.phase)
            add_grid(eval_fit, expansion(arms, tuning), inner, records, ledger, known)
            del fit, eval_fit
        boundary = {
            definition(r).arm for r in records if r["best_epoch"] >=
            math.ceil(tuning["max_epochs"] * tuning["proximate_epoch_fraction"])
        }
        for family in [("A1", "A2", "A3"), ("A4", "A5", "A6", "L1", "L2", "L3", "L4", "L5", "L6", "J1_FiLM_L2"),
                       ("C1", "C2", "C3", "C3_rank0", "A1"), ("C4", "C4_mean"), ("C7", "C7_mean"),
                       ("C8", "C8_mean"), ("C9", "C9_id", "A4"), ("D1", "D1_match"), ("E4_32", "E4_64", "A4")]:
            if boundary.intersection(family):
                boundary.update(family)
        for inner in fold_runs:
            fit_rows, query_rows = source.inner(inner)
            fit = Fit(source, fit_rows, query_rows, observe_query=True)
            evaluator = Evaluator(fit, args.device)
            specs = registered(fit, args.phase)
            extend = [replace(a, horizon=2 * tuning["max_epochs"])
                      for a in specs + expansion(specs, tuning) if a.network and a.arm in boundary]
            add_grid(evaluator, extend, inner, records, ledger, known)
            del fit, evaluator
    return dict(records=records, shift=shift, ledger=ledger,
                stage=args.phase, job=os.environ.get("SLURM_JOB_ID"), host=platform.node(),
                role=source.role, outer=source.outer, seconds=time.monotonic() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--outer", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--roles", nargs="+", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--phase", choices=["core", "expanded", "scaffold"], required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--split-dir", default="partition")
    parser.add_argument("--stage-dir", default="staged")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    for role in args.roles:
        prior = None if args.prior is None else read_json(args.prior / f"outer{args.outer}-{role}.json")
        source = Source(cfg, args.assets, args.outer, role, args.split_dir, args.stage_dir)
        results = run_development(cfg, source, args, prior)
        results.update(config_sha256=sha256(args.config), partition_sha256=source.split["manifest_sha256"],
                       device=args.device,
                       code={str(p): sha256(p) for p in Path("src/r12_multiaxis").glob("*.py")})
        write_json(args.out / f"outer{args.outer}-{role}.json", results)
        print(f"outer {args.outer} role {role}: {len(results['records']) // 2} completed grid fits", flush=True)


if __name__ == "__main__":
    main()
