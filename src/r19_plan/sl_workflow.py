"""Freeze S/L A forecasts and score held drug deltas on exact physical pools."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.metrics import Contrast, correlations
from src.r19_plan.i_evaluate import JoinedPredictions, open_forecasts
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.i_stage import load_score_delta
from src.r19_plan.i_workflow import registered_axes
from src.r19_plan.r18_predict import model_rows
from src.r19_plan.sl_components import exact_pools
from src.r19_plan.sl_predict import FULL_ARMS, SETTINGS, dose_map, predictor_code

FILES = {"rows.tsv", "landmarks.tsv", "full.npz", "neural_only.npz", "selection.json"}
BASES = {"B_R17", "S0", "L0"}
PRIMARY = {"S2", "L1"}
SWAPS = 10_000


def check_forecast(args, key, rows, genes):
    path = args.forecasts / key
    digest = sha256(path / "receipt.json")
    receipt = read_json(path / "receipt.json")
    if (receipt["slice"] != key or receipt["protocol_sha256"] != sha256(args.protocol)
            or receipt["dose_reference_sha256"] != sha256(args.dose_reference)
            or receipt["inputs"]["access"] != sha256(args.split_reference)
            or receipt["code_sha256"] != predictor_code()
            or set(receipt["files"]) != FILES):
        raise ValueError(f"Unregistered forecast metadata: {key}")
    for name, expected in receipt["files"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"S/L forecast changed: {key}/{name}")
    if ((path / "rows.tsv").read_bytes() != rows.to_csv(sep="\t", index=False).encode()
            or (path / "landmarks.tsv").read_bytes() != genes.to_csv(sep="\t", index=False).encode()):
        raise ValueError(f"S/L well or gene axis changed: {key}")
    for name, arms in (("full.npz", FULL_ARMS), ("neural_only.npz", set(SETTINGS))):
        with np.load(path / name, allow_pickle=False) as archive:
            if set(archive.files) != arms:
                raise ValueError(f"S/L model inventory changed: {key}")
            for arm in arms:
                values = archive[arm]
                if values.shape != (len(rows), len(genes)) or values.dtype.kind != "f" or not np.isfinite(values).all():
                    raise ValueError(f"S/L received misaligned or nonfinite predictions: {key}/{arm}")
            if name == "full.npz" and any(not np.array_equal(archive["B_R17"], archive[arm]) for arm in ("S0", "L0")):
                raise ValueError(f"S/L baseline aliases disagree: {key}")
    return path, digest


def parent_swap(rmse, sse, energy):
    """Use common randomized parent-level swaps for the two one-sided endpoints."""
    if len(energy) < 2 or not (energy.sum() > 0):
        raise ValueError("S/L randomization requires evaluable independent parent groups")
    benefits = np.stack((rmse[:, 1] - rmse[:, 0], sse[:, 1] - sse[:, 0]), axis=1)
    observed = benefits.sum(axis=0)
    extreme = np.zeros(2, dtype=np.int64)
    rng = np.random.default_rng(150104)
    for start in range(0, SWAPS, 128):
        size = min(SWAPS - start, 128)
        signs = 2 * rng.integers(0, 2, (size, len(benefits)), dtype=np.int8) - 1
        extreme += (signs @ benefits >= observed - 1e-10).sum(axis=0)
    values = (extreme + 1) / (SWAPS + 1)
    return values, values.max()


def contrast_direction(contrast, prediction, truth):
    by_pool = []
    for ix, weight in zip(contrast.pools, contrast.centers, strict=True):
        p, y = prediction[ix].astype(np.float64), truth[ix].astype(np.float64)
        p -= np.average(p, axis=0, weights=weight)
        y -= np.average(y, axis=0, weights=weight)
        cosine, _ = correlations(p, y)
        parents, to_parent = np.unique(contrast.rows.iloc[ix].compound_group_id, return_inverse=True)
        p = np.stack([p[to_parent == i].mean(axis=0) for i in range(len(parents))])
        y = np.stack([y[to_parent == i].mean(axis=0) for i in range(len(parents))])
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        similarity = p @ y.T
        correct = similarity.diagonal()[:, None]
        tied = np.isclose(similarity, correct, atol=1e-10, rtol=1e-8)
        better = ((similarity > correct) & ~tied).sum(axis=1)
        tied = tied.sum(axis=1)
        by_pool.append(pd.DataFrame(dict(parent=parents[to_parent], cosine=cosine,
            chance_top1=1. / len(parents), chance_top5=min(5, len(parents)) / len(parents),
            top1=(np.clip((1 - better) / tied, 0, 1))[to_parent],
            top5=(np.clip((5 - better) / tied, 0, 1))[to_parent])))
    pooled = pd.concat(by_pool, ignore_index=True)
    macro = pooled.groupby("parent").mean().mean()
    return dict(contrast_cosine=macro.cosine, contrast_top1=macro.top1, contrast_top5=macro.top5,
                chance_top1=macro.chance_top1, chance_top5=macro.chance_top5)


def compare(rows, truth, forecasts, contrast, track):
    reference = "B_R17" if track == "full" else "S0"
    baseline = forecasts[reference]
    energy = squared_rows(truth)
    baseline_error = squared_rows(truth, baseline)
    baseline_contrast = contrast.point(baseline, truth)
    baseline_direction = {f"reference_{key}": value for key, value in contrast_direction(contrast, baseline, truth).items()}
    paired, diagnostics, strata = [], [], []
    for arm in forecasts:
        if (track == "full" and arm in BASES) or (track != "full" and arm == reference):
            continue
        values = forecasts[arm]
        error = squared_rows(truth, values)
        summary = parent_summaries(rows, truth.shape[1], energy, error, baseline_error)
        points, confidence, draws = paired_effects(*summary)
        primary = track == "full" and arm in PRIMARY
        p, joint = parent_swap(*summary) if primary else ([np.nan, np.nan], np.nan)
        rel_low, rel_high, _ = relative_interval(summary[0]) if track == "full" else (np.nan, np.nan, 0)
        for metric, (pred, ref), (low, high), n, single in zip(METRICS, points, confidence, draws, p, strict=True):
            paired.append(dict(track=track, arm=arm, reference=reference, metric=metric, arm_value=pred,
                reference_value=ref, difference=pred - ref, low99=low, high99=high, valid_bootstrap_draws=int(n),
                relative_gain=(1. - pred / ref) if metric == "rmse" and ref > 0 else np.nan,
                relative_low99=rel_low if metric == "rmse" else np.nan,
                relative_high99=rel_high if metric == "rmse" else np.nan,
                n_parents=rows.parent.nunique(), primary=primary, one_sided_p=single, joint_p_unadjusted=joint))
        if track == "full":
            masks = [("fold", int(fold), rows.fold.eq(fold)) for fold in sorted(rows.fold.unique())]
            if primary:
                masks.extend(("plate_removed", plate, rows.rna_plate.ne(plate))
                             for plate in sorted(rows.rna_plate.unique()))
            for scope, value, keep in masks:
                if not keep.any():
                    continue
                rmse, sse, total = parent_summaries(rows.loc[keep], truth.shape[1], energy[keep], error[keep],
                                                     baseline_error[keep])
                strata.append(dict(arm=arm, reference=reference, scope=scope, value=value, parents=len(rmse),
                    delta_rmse=float((rmse[:, 0] - rmse[:, 1]).mean()),
                    delta_r2=float((sse[:, 1] - sse[:, 0]).sum() / total.sum()) if total.sum() else np.nan))
        cosine, _ = correlations(values, truth)
        rms_ratio = np.sqrt(squared_rows(values).sum() / energy.sum()) if energy.sum() else np.nan
        diagnostics.append(dict(track=track, arm=arm, reference=reference,
            contrast_rmse=contrast.point(values, truth), reference_contrast_rmse=baseline_contrast,
            contrast_wells=len(contrast.active), contrast_parents=rows.iloc[contrast.active].parent.nunique(),
            mean_cosine=np.nanmean(cosine) if np.isfinite(cosine).any() else np.nan,
            cosine_valid_wells=int(np.isfinite(cosine).sum()), prediction_truth_rms_ratio=rms_ratio,
            **contrast_direction(contrast, values, truth), **baseline_direction))
    return paired, diagnostics, strata


def run(args):
    if args.action == "score" and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("S/L scoring requires the independent freeze digest and a fresh result path")
    raw, queries, genes, fit = registered_axes(args.plan, args.split_reference, args.cache)
    paths, receipts = [], {}
    for (fold, role), rows in queries.items():
        key = f"f{fold}-{role}"
        path, receipts[key] = check_forecast(args, key, rows, genes)
        paths.append(path)
    module = Path(__file__)
    stamp = dict(protocol_sha256=sha256(args.protocol), split_sha256=sha256(args.split_reference),
                 dose_reference_sha256=sha256(args.dose_reference), folds=receipts,
                 code_sha256={name: sha256(module.with_name(name)) for name in
                              (module.name, "sl_predict.py", "sl_components.py", "i_score.py")})
    if args.action == "freeze":
        write_json(args.freeze, stamp)
        return sha256(args.freeze)
    if read_json(args.freeze) != stamp:
        raise ValueError("S/L predictions or code changed since the handoff to the score process")
    rows = checked_rows(pd.concat(queries.values(), ignore_index=True), fit)
    measured = dose_map(args.instances, args.dose_reference, rows.inst_id)
    models = exact_pools(model_rows(rows), measured)
    # DMSO rotations and outer folds must never share a centering pool.
    models["plate_id"] = models.plate_id + ":" + models.fold.astype(str) + ":" + models.role
    contrast = Contrast(models, minimum=2)
    truth = np.concatenate([load_score_delta(args.cache, args.vehicles, args.plan, args.split_reference, key, query,
                                             genes) for key, query in raw.items()])
    paired, diagnostics, strata = [], [], []
    with ExitStack() as stack:
        for track, filename, arms in (("full", "full.npz", FULL_ARMS), ("neural_only", "neural_only.npz", set(SETTINGS))):
            archives = open_forecasts(stack, paths, filename, arms)
            results, other, sensitivity = compare(
                rows, truth, JoinedPredictions(archives, list(map(len, queries.values())), genes), contrast, track)
            paired.extend(results)
            diagnostics.extend(other)
            strata.extend(sensitivity)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".sl-score-") as temp:
        stage = Path(temp)
        for name, table in (("paired.tsv", paired), ("diagnostics.tsv", diagnostics), ("sensitivity.tsv", strata)):
            pd.DataFrame(table).to_csv(stage / name, sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(stamp, freeze_sha256=args.expected_sha256,
                   files={p.name: sha256(p) for p in sorted(stage.iterdir())}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    for name in ("plan", "split-reference", "cache", "dose-reference", "protocol", "forecasts", "freeze"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("vehicles", "instances", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.action == "score" and any(value is None for value in
                                     (args.out, args.vehicles, args.instances, args.expected_sha256)):
        parser.error("score requires --out, --vehicles, --instances, and --expected-sha256")
    print(run(args))
