"""Verify frozen arrays and estimate paired compound-group effects with pool re-centering."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, GROUP, KEY, LINE, load_data, table
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.report import align, align_pool
from src.r12_multiaxis.score import measure
from src.r12_multiaxis.stage import load_staged
from src.r6_followup_v1.metrics import grams

from .objectives import GainScores
from .predict import competing_families


def stage_roots(assets, protocol):
    keys = ("partition", "staged") if protocol == "compound" else ("scaffold", "staged-scaffold")
    return (assets / keys[0], assets / keys[1])


def check_task(path, args, split, fold, role):
    rec = read_json(path / "receipt.json")
    if (rec["protocol"], rec["fold"], rec["role"]) != (args.protocol, fold, role):
        raise ValueError("Prediction receipt belongs to a different slice")
    if rec["partition_sha256"] != read_json(split / "split.json")["manifest_sha256"]:
        raise ValueError("Prediction receipt belongs to a different partition")
    if rec["study_sha256"] != sha256(args.study) or rec["config_sha256"] != sha256(args.config):
        raise ValueError("Prediction receipt does not implement this contract")
    for name, expected in rec["files"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"Frozen artifact changed {path / name}")
    for name, expected in rec["implementation"].items():
        if sha256(name) != expected:
            raise ValueError(f"Implementation changed {name}")
    families = competing_families(config(args.study)["selection"])
    if set(rec["predictions"]) != {"B0", "B_abs", "anchor", "selector", "ensemble", *families}:
        raise ValueError("Frozen prediction arms do not match the study contract")


def freeze(args):
    split, stage = stage_roots(args.assets, args.protocol)
    receipts = {}
    for fold in range(3):
        for ri, role in enumerate("AB"):
            task = f"task{2 * fold + ri}"
            check_task(args.predictions / task, args, split, fold, role)
            receipts[task] = sha256(args.predictions / task / "receipt.json")
        a, b = [args.predictions / f"task{2 * fold + ri}" for ri in range(2)]
        rows = [table(p / "rows.tsv") for p in (a, b)]
        if not rows[0][KEY + [GROUP]].equals(rows[1][KEY + [GROUP]]) or rows[0][CONTROL].eq(rows[1][CONTROL]).any():
            raise ValueError("Control roles or physical well spaces are not paired")
        if not np.array_equal(np.load(a / "panel.npy"), np.load(b / "panel.npy")):
            raise ValueError("Role output panels differ")
    write_json(args.freeze, dict(
        time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        protocol=args.protocol, predictions=str(args.predictions.resolve()),
        exposure="R12-exposed data, prospectively frozen R13 cross-validation",
        study_sha256=sha256(args.study), config_sha256=sha256(args.config),
        stage_receipt_sha256=sha256(stage / "receipt.json"), task_receipts=receipts,
        scorer_sha256=sha256(__file__),
    ))


def draw_contrast(metric, pred, observed, draw, group_index):
    """Compute one weighted numerator per bootstrap draw using pool Gram matrices."""
    kernel = grams(pred.astype(np.float64) - observed, metric.pools)
    result = np.zeros(len(draw))
    for grain, rows, center in zip(kernel, metric.pools, metric.centers):
        positive = draw[:, group_index[rows]]
        normalized = positive * center
        normalized /= normalized.sum(axis=1, keepdims=True)
        m = normalized @ grain
        variance = np.diag(grain) - 2 * m + (m * normalized).sum(axis=1, keepdims=True)
        result += np.einsum("bi,bi,i->b", np.sqrt(np.maximum(variance, 0)), positive,
                            metric.primary_weight[rows])
    return result


def controls(r11, rows, role, panel):
    other = "B" if role == "A" else "A"
    x, xc = load_data(r11 / "source", role, panel)
    y, yc = load_data(r11 / "source", other, panel)
    xi = align_pool(rows, x, ["plate_id", LINE])
    yi = align_pool(rows, y, ["plate_id", LINE])
    if not np.array_equal(x.iloc[xi][KEY[0]].values, rows[CONTROL].values):
        raise ValueError("Own-control lookup differs from frozen predictor")
    if np.any(x.iloc[xi][KEY[0]].values == y.iloc[yi][KEY[0]].values):
        raise ValueError("Score uses predictor's own physical control well")
    return xc[xi].toarray(), yc[yi].toarray()


def score(args):
    frozen = read_json(args.freeze)
    if (frozen["study_sha256"], frozen["config_sha256"], frozen["scorer_sha256"]) != (
            sha256(args.study), sha256(args.config), sha256(__file__)):
        raise ValueError("Evaluation code or contract changed after freeze")
    if frozen["predictions"] != str(args.predictions.resolve()) or frozen["protocol"] != args.protocol:
        raise ValueError("Evaluation paths differ from frozen manifest")
    _, stage = stage_roots(args.assets, args.protocol)
    if sha256(stage / "receipt.json") != frozen["stage_receipt_sha256"]:
        raise ValueError("Staged input changed after freeze")
    for task, expected in frozen["task_receipts"].items():
        rec_path = args.predictions / task / "receipt.json"
        if sha256(rec_path) != expected:
            raise ValueError("Frozen receipt was replaced")
        for name, checksum in read_json(rec_path)["files"].items():
            if sha256(rec_path.parent / name) != checksum:
                raise ValueError(f"Frozen prediction bytes changed {rec_path.parent / name}")
    staged = read_json(stage / "receipt.json")
    for name, expected in staged["files"].items():
        if "/score-" in name and sha256(stage / name) != expected:
            raise ValueError("Held-out outcome staging differs from its receipt")

    settings = config(args.study)["evaluation"]
    cfg = config(args.config)
    all_rows = {int(task[4:]): table(args.predictions / task / "rows.tsv") for task in frozen["task_receipts"]}
    groups = sorted(set().union(*(set(r[GROUP]) for r in all_rows.values())))
    draw = np.random.default_rng(settings["seed"]).exponential(size=(settings["draws"], len(groups)))
    draw /= draw.mean(axis=1, keepdims=True)
    denominator = dict(contrast=np.zeros(len(draw)), absolute=np.zeros(len(draw)))
    runs = []
    boot = defaultdict(lambda: {"contrast": np.zeros(len(draw)), "absolute": np.zeros(len(draw))})
    for fold in range(3):
        true_rows, counts = load_staged(stage / f"outer{fold}", "score", staged["genes"])
        for ri, role in enumerate("AB"):
            task = fold * 2 + ri
            rows = all_rows[task]
            path = args.predictions / f"task{task}"
            rec = read_json(path / "receipt.json")
            panel = np.load(path / "panel.npy", allow_pickle=False)
            absolute = counts[align(rows, true_rows)][:, panel].toarray()
            own, opposite = controls(Path(cfg["r11_assets"]), rows, role, panel)
            observed = absolute - opposite
            ci = Contrast(rows, cfg["minimum_groups"])
            gidx = pd.Index(groups).get_indexer(rows[GROUP])
            scopes = {"contrast": ci.primary_weight, "absolute": ci.all_weight}
            for label, weight in scopes.items():
                norm = np.bincount(gidx, weights=weight, minlength=len(groups))
                denominator[label] += draw @ norm
            gains = GainScores(rows, observed, np.zeros_like(observed), cfg["minimum_groups"])
            for arm, files in rec["predictions"].items():
                for name in files:
                    prediction = np.load(path / name, allow_pickle=False)
                    measured = measure(rows, ci, prediction, observed, own, absolute)
                    coverage, alignment = gains.diagnostics(prediction)
                    point = {}
                    for key, values in measured.items():
                        weight = scopes["contrast" if key in ("contrast", "pds", "top1") else "absolute"]
                        active = np.isfinite(values) & (weight > 0)
                        point[key] = float(np.average(values[active], weights=weight[active]))
                    contrast_num = draw_contrast(ci, prediction, observed, draw, gidx)
                    abs_num = draw @ np.bincount(gidx, weights=measured["absolute"] * scopes["absolute"],
                                                 minlength=len(groups))
                    boot[arm]["contrast"] += contrast_num / len(files)
                    boot[arm]["absolute"] += abs_num / len(files)
                    runs.append(dict(arm=arm, fold=fold, role=role, prediction=name, groups=rows[GROUP].nunique(),
                                     contrast_weight=scopes["contrast"].sum(), absolute_weight=scopes["absolute"].sum(),
                                     alignment_coverage=coverage, centered_cosine=alignment, **point))
                print(f"{args.protocol} {fold}/{role} scored {arm}", flush=True)
    df = pd.DataFrame(runs)
    meta = ["groups", "contrast_weight", "absolute_weight", "arm", "fold", "role", "prediction"]
    metrics = [c for c in df if c not in meta]
    per_task = df.groupby(["arm", "fold", "role"], as_index=False).mean(numeric_only=True)
    points = {}
    for arm, rows in per_task.groupby("arm"):
        points[arm] = {
            m: float(np.average(rows.loc[rows[m].notna(), m],
                                weights=rows.loc[rows[m].notna(), "absolute_weight" if m == "absolute" else "contrast_weight"]))
            if rows[m].notna().any() else None for m in metrics
        }
    effects = []
    for arm in points:
        for metric, baseline in [("contrast", settings["contrast_baseline"]),
                                 ("absolute", settings["absolute_baseline"])]:
            if arm == baseline:
                continue
            draws = (boot[arm][metric] - boot[baseline][metric]) / denominator[metric]
            # Two pre-registered primary endpoint comparisons for selector only.
            confidence = settings["confidence"] if arm == settings["primary"] else 0.95
            tail = (1 - confidence) / 2
            low, high = np.quantile(draws, [tail, 1 - tail])
            effect = points[arm][metric] - points[baseline][metric]
            effects.append(dict(arm=arm, metric=metric, baseline=baseline, confidence=confidence,
                                effect=effect, lower=low, upper=high, n_groups=len(groups),
                                decision="improves" if high < 0 else "worse" if low > 0 else "inconclusive"))
    args.out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame.from_dict(points, orient="index").rename_axis("arm").to_csv(args.out / "arm_metrics.tsv", sep="\t")
    df.to_csv(args.out / "by_fold_role_seed.tsv", sep="\t", index=False)
    pd.DataFrame(effects).to_csv(args.out / "paired.tsv", sep="\t", index=False)
    np.savez(args.out / "bootstrap.npz", draw=draw, groups=groups, **denominator)
    write_json(args.out / "receipt.json", {
        "freeze_sha256": sha256(args.freeze), "n_groups": len(groups), "draws": len(draw),
        "files": {p.name: sha256(p) for p in args.out.iterdir()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "score"])
    for k in ("config", "study", "assets", "predictions", "freeze"):
        parser.add_argument(f"--{k}", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--protocol", choices=["compound", "scaffold"], default="compound")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args)
    elif args.out is None:
        parser.error("Score requires --out")
    else:
        score(args)


if __name__ == "__main__":
    main()
