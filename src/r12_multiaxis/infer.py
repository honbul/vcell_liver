"""Select inside the held-in partitions and freeze each role's held-out predictions."""

import argparse
from dataclasses import asdict, replace
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
from .candidates import Candidate, Evaluator, alternatives
from .data import Fit, Source
from .develop import CalibratorFits, dual_points, pick_shift, rms, single_choices


MATCHED = {
    "C2": "A1", "C3": "C3_rank0", "C4": "C4_mean", "C7": "C7_mean", "C8": "C8_mean", "C9": "A4",
    "A2": "A1", "A3": "A1", "A5": "A4", "A6": "A4",
    "L1": "A4", "L2": "A4", "L3": "A4", "L4": "A4", "L6": "L5",
}


def argmin(options, tie):
    limit = min(v[0] for v in options.values()) + tie
    return next(k for k, value in options.items() if value[0] <= limit)


def integrated_options(cfg, source, development, device):
    records, shifts = development["records"], development["shift"]
    arms, winners, calibrated, shift, shift_pool, shift_nn = single_choices(records, shifts, cfg["training"])
    if {"D1", "D1_match"} <= set(arms):
        comparison = arms["D1_match"][1]
        dose = arms["D1"][1]
        arms["D1"] = (None, replace(dose, epoch=comparison.epoch, lr=comparison.lr,
                                    horizon=comparison.horizon))
    best_context = winners["Context"][1]
    best_structure = winners["Structure"][1]
    best_loss = winners["Loss"][1]
    combo = replace(best_loss, arm="J2_raw", context=best_context.context, head=best_structure.head)
    scores = []
    ledger = []
    family = shift[1]["family"]
    regs = cfg["training"]["calibration_penalties"]
    for inner in range(2):
        train, val = source.inner(inner)
        fit = Fit(source, train, val, observe_query=True)
        metric = Contrast(fit.query.rows, cfg["minimum_groups"])
        evaluator = Evaluator(fit, device)
        cross = CalibratorFits(source, fit, train, device, cfg["split_seed"] + inner + 100)
        best, runtime, curve, _ = evaluator.evaluate(combo)
        prediction = best["contrast"][2]
        raw = dual_points(metric, prediction, fit.query.delta)
        ledger.append(dict(phase="inner_joint", inner=inner, candidate=asdict(combo), runtime=runtime, curve=curve))
        oof, anchor, cv_runs = cross.oof(combo)
        ledger.append(dict(phase="inner_joint_calibration", inner=inner, candidate=asdict(combo),
                           calibration_runs=cv_runs, crossfit_prediction_rms=rms(oof),
                           refit_prediction_rms=rms(prediction)))
        for penalty in regs:
            model = Calibration(fit.train.rows, oof, anchor, fit.train.delta, fit.scale, penalty, family)
            p = model.predict(fit.query.rows, prediction, fit.offset)
            point = dual_points(metric, p, fit.query.delta)
            scores.append(dict(base=asdict(combo), family=family, penalty=penalty, inner=inner,
                               groups=fit.query.rows[GROUP].nunique(), abs_before=raw["abs"],
                               contrast_before=raw["contrast"], **point))
    unshifted = pick_shift(scores, lambda r: r["penalty"] == "identity", cfg["training"]["tie"], "contrast")
    calib_combo = pick_shift(scores, lambda r: r["penalty"] != "identity", cfg["training"]["tie"], "contrast")
    single_candidates = {"Context": winners["Context"], "Structure": winners["Structure"], "Loss": winners["Loss"]}
    shift_contrast = pick_shift(shifts, lambda r: r["base"] == shift[1]["base"] and
                                r["penalty"] == shift[1]["penalty"] and r["family"] == shift[1]["family"],
                                cfg["training"]["tie"], "contrast")
    single_candidates["Shift"] = (shift_contrast[0], Candidate("Shift", "calibrated"))
    best_single = argmin(single_candidates, cfg["training"]["tie"])
    selection = {
        "B_contrast": winners["B_contrast"],
        "single": single_candidates[best_single],
        "J2_raw": (unshifted[0], combo),
        "J2_calibrated": (calib_combo[0], combo),
    }
    best = argmin(selection, cfg["training"]["tie"])
    chosen = dict(
        arm_fits={arm: dict(score=s, candidate=asdict(c)) for arm, (s, c) in arms.items()},
        winners={arm: dict(score=s, candidate=asdict(c)) for arm, (s, c) in winners.items()},
        shift={name: {"inner_abs": res[0], **res[1]} for name, res in
               {**calibrated, "Shift": shift, "Shift_pool": shift_pool, "Shift_neural": shift_nn}.items()},
        joint=dict(raw=asdict(combo), calibrated=calib_combo[1], grid=scores,
                   selected=best if best != "single" else best_single, best_single=best_single,
                   scores={k: v[0] for k, v in selection.items()},
                   single_scores={k: v[0] for k, v in single_candidates.items()}),
        attribution={k: MATCHED[v[1].arm] for k, v in winners.items() if v[1].arm in MATCHED},
        source_contract="All decisions derived from this role and outer-train inner predictions only",
    )
    return chosen, ledger


def _network_identity(candidate):
    return replace(candidate, arm="", horizon=0 if candidate.epoch else candidate.horizon)


class FrozenPredictions:
    def __init__(self, out, fit, evaluator, cfg, cross, selection):
        self.out = out
        self.fit = fit
        self.evaluator = evaluator
        self.cfg = cfg
        self.cross = cross
        self.selection = selection
        self.entries = {}
        self.aliases = {}
        self.checkpoints = []
        self.ledger = []
        self.cached = {}
        self.diagnostic_by_seed = {}
        self.shape = fit.offset.shape

    def save(self, arm, seed, prediction):
        label = f"{arm}__{seed}"
        relative = f"predictions/{label}.npy"
        if prediction.shape != self.shape or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid frozen prediction for {arm}/{seed}")
        np.save(self.out / relative, prediction)
        self.entries.setdefault(arm, {})[str(seed)] = relative

    def train(self, arm, spec, confirm, perturb=False):
        seeds = self.cfg["confirmation_seeds"] if spec.network and confirm else (
            [self.cfg["seed"]] if spec.network else ["deterministic"])
        design = alternatives(self.fit, [42, 43, 44, 45, 46]) if perturb else None
        for seed in seeds:
            identity = (_network_identity(spec), seed)
            if identity in self.cached and not perturb:
                self.entries.setdefault(arm, {})[str(seed)] = self.cached[identity]
                continue
            result, runtime, curve, diagnostic = self.evaluator.evaluate(
                spec, seed=seed if spec.network else 42, retain=True, replacement=design)
            pred, checkpoint = result["contrast"][2:]
            name = f"{arm}__{seed}"
            self.save(arm, seed, pred)
            self.cached[identity] = self.entries[arm][str(seed)]
            for label in diagnostic.keys() - {"representation"}:
                self.save(label, seed, diagnostic[label])
            if "representation" in diagnostic:
                self.diagnostic_by_seed[str(seed)] = diagnostic["representation"]
            if checkpoint:
                entry = f"models/{name}"
                if spec.network:
                    entry += ".pt"
                    torch.save(checkpoint, self.out / entry)
                else:
                    entry += ".npz"
                    np.savez(self.out / entry, **checkpoint)
                self.checkpoints.append(entry)
            self.ledger.append(dict(phase="outer_fit", arm=arm, candidate=asdict(spec),
                                    runtime=runtime, curve=curve))
            print(f"frozen {arm} seed={seed}", flush=True)

    def apply(self, arm, reference, details):
        candidate = Candidate(**details["base"])
        family, penalty = details["family"], details["penalty"]
        for seed, file in self.entries[reference].items():
            base = np.load(self.out / file)
            oof, oof_anchor, cv_runs = self.cross.oof(candidate, seed=int(seed) if seed.isdecimal() else 42)
            model = Calibration(self.fit.train.rows, oof, oof_anchor, self.fit.train.delta, self.fit.scale,
                                penalty, family)
            self.save(arm, seed, model.predict(self.fit.query.rows, base, self.fit.offset))
            relative = f"models/{arm}__{seed}_coefficients.npz"
            np.savez(self.out / relative, **model.vectors())
            self.checkpoints.append(relative)
            self.ledger.append(dict(phase="outer_calibration", arm=arm, seed=seed, calibration=model.record(),
                                    calibration_runs=cv_runs, crossfit_prediction_rms=rms(oof),
                                    refit_prediction_rms=rms(base),
                                    rms_difference=rms(base) - rms(oof), coefficients=relative))

    def alias(self, arm, source):
        self.entries[arm] = self.entries[source]
        self.aliases[arm] = source


def create_predictions(cfg, source, out, selection, device):
    fit = Fit(source, source.fit_rows, source.query_rows)
    cross = CalibratorFits(source, fit, source.fit_rows, device, cfg["split_seed"] + source.outer + 200)
    final = FrozenPredictions(out, fit, Evaluator(fit, device), cfg["training"], cross, selection)
    selected_arms = {r["candidate"]["arm"] for r in selection["winners"].values()}
    confirmed = selected_arms | set(selection["attribution"].values()) | {
        "A1", "A2", "A3", "A4", "A5", "A6", "C1", "C2", "C3_rank0", "L1", "L2", "L3", "L5", "L6", "J1_FiLM_L2",
    }
    for arm, row in selection["arm_fits"].items():
        final.train(arm, Candidate(**row["candidate"]), arm in confirmed, perturb=arm == "A1")
    for label, record in selection["winners"].items():
        chosen = Candidate(**record["candidate"])
        if chosen == Candidate(**selection["arm_fits"][chosen.arm]["candidate"]):
            final.alias(label, chosen.arm)
        else:
            final.train(label, chosen, confirm=True)
    for arm, details in selection["shift"].items():
        base = Candidate(**details["base"])
        if base.arm not in final.entries:
            final.train(base.arm, base, confirm=True)
        final.apply(arm, base.arm, details)
    combo = Candidate(**selection["joint"]["raw"])
    final.train("J2_raw", combo, confirm=True)
    if combo.context != "pca":
        final.train("J2_pca_context", replace(combo, context="pca"), confirm=True)
    if combo.loss != "L0":
        final.train("J2_mse", replace(combo, loss="L0", vic=0.0), confirm=True)
    final.apply("J2_calibrated", "J2_raw", selection["joint"]["calibrated"])
    final.alias("single", selection["joint"]["best_single"])
    final.alias("Integrated", selection["joint"]["selected"])
    for label in ("Context", "Structure", "Loss"):
        final.alias(f"{label}_matched", selection["attribution"][label])
    np.save(out / "panel.npy", fit.panel)
    np.save(out / "anchor.npy", fit.offset)
    for arm, value in selection["arm_fits"].items():
        rank = value["candidate"]["response_rank"]
        if rank:
            vectors, center = fit.inputs("pca", rank=rank).response_head
            original = fit.residual / fit.scale
            lost = original - ((original - center) @ vectors.T @ vectors + center)
            rmse = np.sqrt(np.square(lost * fit.scale).mean(axis=1))
            np.savez(out / f"models/{arm}_projection.npz", vectors=vectors, center=center, scale=fit.scale,
                     train_reconstruction_rmse=np.average(rmse, weights=fit.weight))
            final.checkpoints.append(f"models/{arm}_projection.npz")
    fit.query.rows.to_csv(out / "rows.tsv", sep="\t", index=False)
    write_json(out / "prediction_index.json", final.entries)
    write_json(out / "selection.json", selection)
    write_json(out / "runtime.json", {"ledger": final.ledger, "aliases": final.aliases,
                                     "vicreg_representation": final.diagnostic_by_seed,
                                     "pool_offset_fallback": {k: fit.offset_level.count(k) for k in sorted(set(fit.offset_level))}})
    files = {"panel.npy", "anchor.npy", "rows.tsv", "selection.json", "prediction_index.json", "runtime.json", "joint_dev.json",
             *final.checkpoints, *(file for specs in final.entries.values() for file in specs.values())}
    return {f: sha256(out / f) for f in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--roles", nargs="+", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--split-dir", default="partition")
    parser.add_argument("--stage-dir", default="staged")
    args = parser.parse_args()
    cfg = config(args.config)
    torch.set_num_threads(4)
    for role in args.roles:
        start = time.monotonic()
        development_path = args.dev / f"outer{args.outer}-{role}.json"
        development = read_json(development_path)
        source = Source(cfg, args.assets, args.outer, role, args.split_dir, args.stage_dir)
        if development["config_sha256"] != sha256(args.config) or development["partition_sha256"] != source.split["manifest_sha256"]:
            raise ValueError("Inference config or manifest does not match development")
        selection, ledger = integrated_options(cfg, source, development, args.device)
        out = args.out / f"outer{args.outer}" / role
        out.mkdir(parents=True, exist_ok=False)
        (out / "predictions").mkdir()
        (out / "models").mkdir()
        write_json(out / "joint_dev.json", ledger)
        files = create_predictions(cfg, source, out, selection, args.device)
        write_json(out / "receipt.json", dict(
            role=role, outer=args.outer, role_selection_source=sha256(development_path),
            files=files, partition_sha256=source.split["manifest_sha256"],
            config_sha256=sha256(args.config), seconds=time.monotonic() - start,
            code={str(p): sha256(p) for p in Path("src/r12_multiaxis").glob("*.py")},
            host=platform.node(), job=os.environ.get("SLURM_JOB_ID"), device=args.device,
        ))


if __name__ == "__main__":
    main()
