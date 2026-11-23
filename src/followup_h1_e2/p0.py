"""Reproduce the archived HCC recommendation evidence and change only the resampling unit."""
import argparse
from ast import literal_eval
from functools import lru_cache
from hashlib import sha256 as digest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from confirm_hcc import fewshot, metrics
from confirm_hcc.contract import OBSERVATIONS, ORIGINAL
from nextgen_h1_anchor.inputs import json_write
from vc_build.evaluate import join_observations
from vc_build.io import sha256, verify_forecast

from .curves import CONTRACT, ROOT, checked, read_json

ARCHIVE = ROOT.parents[1] / "confirm_20261123/worker1_eff"
# Copied from the pinned Git tree, alongside its exact original implementation.
ORIGINAL_CODE = Path(__file__).resolve().parents[2] / "p0"
ORIGINAL_REPORTS = ORIGINAL_CODE / "reports/confirm_hcc_efficacy_v1"
ORIGINAL_CONTRACT = ORIGINAL_CODE / "experiments/confirm_hcc_efficacy_v1/contract.json"
RANK_METRICS = ("ndcg_at_5", "precision_at_k")
EFFECT = ("delta", "ci95_low", "ci95_high")


def read_tsv(path):
    return pd.read_csv(path, sep="\t")


def axes_sha(values):
    return digest(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def ranks(y, p):
    return np.stack(metrics.within_cell(y, p, 5, 5)[:2], axis=1)


def weights(count, draws, seed):
    sampled = np.random.default_rng(seed).integers(count, size=(draws, count))
    return np.array([np.bincount(row, minlength=count) for row in sampled])


def average_split(row_scores, w, minimum):
    w = w * np.isfinite(row_scores)[None]
    total = w.sum(axis=1)
    return np.divide(w @ np.nan_to_num(row_scores), total, where=total >= minimum,
                     out=np.full(len(w), np.nan))


def check_errors(observed, original, complete):
    if not len(original) or not (observed.shape == original.shape):
        raise ValueError("Published and reproduced evidence differ in size")
    if not np.array_equal(np.isnan(observed), np.isnan(original)):
        raise ValueError("Archived evidence has a different missingness pattern")
    error = float(np.nanmax(np.abs(observed - original)))
    if error >= 7e-10:
        raise ValueError(f"Original {complete} reproduction error {error:.14g}")
    return {"numeric_elements": int(np.isfinite(original).sum()), "max_abs_diff": error}


def signs(lo, hi):
    return "positive" if lo > 0 else "negative" if hi < 0 else "includes_zero"


def contrast_row(path, track, k, n, panel, crossed_n, original, fixed, donor_hash, draws):
    lo, hi = metrics.interval(fixed)
    return {"study": path, "track": track, "k": k, "independent_n": n, "fixed_drugs": len(panel),
            "panel_sha256": axes_sha(panel), "unit_roster_sha256": donor_hash,
            "model_id": original.model_id, "baseline": original.baseline,
            "metric": original.metric.replace("precision_at_k", "precision_at_5"),
            "delta": original.delta, "original_low": original.ci95_low, "original_high": original.ci95_high,
            "original_draws": crossed_n, "fixed_low": lo, "fixed_high": hi, "fixed_draws": draws,
            "fixed_draws_nonmissing": int(np.isfinite(fixed).sum()),
            "original_zero_test": signs(original.ci95_low, original.ci95_high),
            "fixed_zero_test": signs(lo, hi),
            "zero_test_changed": signs(lo, hi) != signs(original.ci95_low, original.ci95_high)}


@lru_cache
def report_source_hashes():
    files = [ORIGINAL_REPORTS / name for name in (
        "RESULTS.md", "FEWSHOT_3D_CONFIRM.md", "reference_decision.json", "fewshot_decision.json")]
    files.extend(Path(module.__file__) for module in (metrics, fewshot))
    files.append(ORIGINAL_CONTRACT)
    return {str(p.relative_to(ORIGINAL_CODE)): sha256(p) for p in files}


def limore(out, draws, seed):
    old = read_json(ORIGINAL_REPORTS / "score_receipt.json")
    if sha256(ORIGINAL_CONTRACT) != old["contract_sha256"]:
        raise ValueError("Archived LIMORE contract changed")
    fpath = ARCHIVE / "task1/limore.parquet"
    forecast, stamp = verify_forecast(fpath)
    label_path = checked(ORIGINAL / "test_labels/outcomes.parquet", old["labels_sha256"])
    if (stamp["sha256"] != old["prediction_sha256"] or old["score_passes"] != 1 or
            sha256(ARCHIVE / "task1/score/score_receipt.json") != sha256(ORIGINAL_REPORTS / "score_receipt.json") or
            sha256(fpath.with_suffix(".parquet.freeze.json")) != sha256(ORIGINAL_REPORTS / "limore_forecast_receipt.json") or
            old["labels_sha256"] != OBSERVATIONS):
        raise ValueError("Archived LIMORE prediction or receipt changed")
    joined = join_observations(forecast, pd.read_parquet(label_path))
    truth = metrics.pivot(joined[joined.method.eq("drug_mean")], "endpoint_value")
    y = truth.to_numpy()
    models = {name: metrics.pivot(block, "predicted_value").reindex_like(truth).to_numpy()
              for name, block in joined.groupby("method")}
    if y.shape != (50, 68) or not np.isfinite(y).all() or set(models) != set(old["models"]) or not all(
            np.isfinite(v).all() for v in models.values()):
        raise ValueError("LIMORE changed from the complete original panel")
    original_points = read_tsv(checked(ORIGINAL_REPORTS / "UNIFIED_2D_LEDGER.tsv",
                                      old["files"]["UNIFIED_2D_LEDGER.tsv"])).set_index("model_id")
    original_points = original_points.rename(columns={"precision_at_5": "precision_at_k"})
    original_ci = read_tsv(checked(ORIGINAL_REPORTS / "LIMORE_PAIRED_95CI.tsv",
                                   old["files"]["LIMORE_PAIRED_95CI.tsv"]))
    point = pd.DataFrame({m: metrics.metrics(y, p) for m, p in models.items()}).T
    ans = check_errors(point.loc[original_points.index, list(metrics.AXES)].to_numpy(),
                       original_points.loc[:, list(metrics.AXES)].to_numpy(), "LIMORE point")
    legacy_n = old["crossed_bootstrap_draws"]
    legacy = metrics.bootstrap(y, models, repeats=legacy_n)
    reproduced = np.array([(point.loc[r.model_id, r.metric] - point.loc[r.baseline, r.metric],
                            *metrics.interval(legacy[r.model_id][r.metric] - legacy[r.baseline][r.metric]))
                           for r in original_ci.itertuples()])
    crossed = check_errors(reproduced, original_ci[list(EFFECT)].to_numpy(), "LIMORE crossed intervals")
    seal = {"study": "LIMORE", "k": 0, "forecast_sha256": stamp["sha256"], "labels_sha256": sha256(label_path),
            "published_score_receipt_sha256": sha256(ORIGINAL_REPORTS / "score_receipt.json"),
            "original_point_replay": ans, "original_crossed_replay": crossed, "original_crossed_draws": legacy_n}

    # Confirm the originally published table before starting the fixed-panel resampling.
    labels = ("약물 평균", "약물 + 세포 가산", "약물별 Ridge", "CellHit 재적합", "정정 구조 커널",
              "Ridge + 구조 커널 50:50", "V0+ v2 Pairwise Rank", "V0+ v2 Pairwise Top-5", "ECFP4 + 경로 ElasticNet")
    source_md = (ORIGINAL_REPORTS / "RESULTS.md").read_text()
    for m, tag in zip(old["models"], labels):
        row = next(line for line in source_md.splitlines() if line.startswith(f"| {tag} |"))
        parts = [float(token.strip().split("†")[0]) for token in row.strip("|").split("|")[1:]]
        if not np.allclose(parts, point.loc[m, list(metrics.AXES)], atol=5e-5 + 1e-12, rtol=0):
            raise ValueError(f"LIMORE markdown does not reproduce {tag}")
    json_write(out / "original_validation.json", seal | {"published_markdown_point_rows_reproduced": len(labels)})

    w = weights(len(y), draws, seed)
    fixed = {m: {metric: average_split(s, w, 1) for metric, s in zip(RANK_METRICS, ranks(y, p).T)}
             for m, p in models.items()}
    group_hash = axes_sha(truth.index)
    table = pd.DataFrame([contrast_row("LIMORE", "LIMORE_adult_HCC_cell_cold", 0, len(y), truth.columns, legacy_n,
        r, fixed[r.model_id][r.metric] - fixed[r.baseline][r.metric], group_hash, draws)
        for r in original_ci[original_ci.metric.isin(RANK_METRICS)].itertuples()])
    oof_name = "SOURCE_OOF_MACRO.tsv"
    oof = read_tsv(checked(ORIGINAL_REPORTS / oof_name, old["files"][oof_name])).set_index("model_id")
    oof_axes = ["ndcg_at_5", "within_drug_spearman"]
    original_gate = read_json(ORIGINAL_REPORTS / "reference_decision.json")
    gates = []
    for m in original_gate["model_specific_evidence"]:
        model = m["model_id"]
        comp = table[(table.model_id == model) & table.baseline.isin(["additive", "drug_mean"]) &
                     table.metric.eq("ndcg_at_5")]
        oof_pass = all((oof.loc[model, oof_axes] - oof.loc[b, oof_axes]).gt(0).all()
                       for b in ("additive", "drug_mean"))
        unchanged = (m["top5_sets"] > 1 and m["residual_lower_95ci"] > 0 and oof_pass and
                     all((v["within_drug_spearman"] or 0) > 0 for v in m["limore_lower_bounds"].values()))
        legacy_gate = bool(len(comp) == 2 and unchanged and comp.original_low.gt(0).all())
        new_gate = bool(len(comp) == 2 and unchanged and comp.fixed_low.gt(0).all())
        if legacy_gate != m["qualified_joint"]:
            raise ValueError(f"Original LIMORE gate mismatch for {model}")
        gates.append({"model_id": model, "track": "LIMORE_adult_HCC_cell_cold",
                      "original_joint_gate": legacy_gate, "fixed_counterfactual_joint_gate": new_gate})
    return table, gates, len(y), axes_sha(w.ravel())


def licob(k, out, draws, seed):
    root = ARCHIVE / "task3"
    history = read_json(ORIGINAL_REPORTS / "fewshot_collection_receipt.json")
    registration = root / "registration"
    registered = checked(registration / "registration.json", history["registration_sha256"])
    key = f"LICOB/K{k}"
    record = history["keyed_evaluation"][key]
    prior_path = root / "source.parquet"
    forecast_dir = root / f"LICOB/k{k}/predict"
    score_dir = forecast_dir.parent / "score"
    score_path = checked(score_dir / "score_receipt.json", record["score_receipt_sha256"])
    old = read_json(score_path)
    fpath = forecast_dir / "forecasts.parquet"
    forecast, stamp = verify_forecast(fpath)
    provenance = stamp["provenance"]
    values = (provenance["source_sha256"], provenance["registration_sha256"], provenance["study"],
              provenance["k"], provenance["repeats"], stamp["sha256"])
    if values != (history["source_predicted_sha256"], sha256(registered), "LICOB", k, 100,
                  record["frozen_predictions_sha256"]) or old["predictions_sha256"] != stamp["sha256"]:
        raise ValueError("Archived LICOB prediction changed")
    checked(forecast_dir / "calibration_parameters.tsv", provenance["parameters_sha256"])
    checked(prior_path, history["source_predicted_sha256"])
    args = SimpleNamespace(registration=registration, prior=prior_path,
                           contract=ORIGINAL_CONTRACT, study="LICOB", k=k)
    _, label_rows, layouts, _ = fewshot.inputs(args)
    registered_files = read_json(registered)["artifacts"]
    ppoint_file, pci_file = "FEWSHOT_3D_METRICS.tsv", "FEWSHOT_3D_PAIRED_95CI.tsv"
    pp, pc = (read_tsv(checked(ORIGINAL_REPORTS / name, history["files"][name])) for name in (ppoint_file, pci_file))
    points, replay_stats, inputs = [], [], []
    for track, frame in forecast.groupby("track", sort=True):
        panel = pd.Index(sorted(frame.compound_id.unique()))
        truth = metrics.pivot(label_rows, "endpoint_value").loc[:, panel]
        y = truth.to_numpy()
        if y.shape != (29, 53) or not np.isfinite(y).all():
            raise ValueError("Original LICOB exact-parent panel changed")
        models = sorted(frame.method.unique())
        if set(models) != set(fewshot.MODELS) or len(layouts) != 100:
            raise ValueError("Incomplete LICOB model or repeat coverage")
        arrays, splits, rank_rows, episode_points = [], [], [], []
        for event in layouts:
            held = event["held"]
            block = frame[frame.fold.eq(event["repeat"])]
            if set(held) & set(event["fit"]) or set(held) | set(event["fit"]) != set(truth.index) or (
                set(block.lineage_group) != set(held) or len(block) != (29-k)*53*len(models) or
                block.duplicated(["lineage_group", "compound_id", "method"]).any()):
                raise ValueError("Original calibration/test split mismatch")
            current = {m: metrics.pivot(p, "predicted_value").loc[held, panel].to_numpy()
                       for m, p in block.groupby("method")}
            idx = truth.index.get_indexer(held)
            observed = y[idx]
            arrays.append(current)
            splits.append(idx)
            rstats = {}
            for m, p in current.items():
                info = metrics.metrics(observed, p, min_lineages=2)
                precision1 = float(metrics.mean(metrics.within_cell(observed, p, 5, 1)[1]))
                episode_points.append({"model_id": m, **info, "precision_at_1": precision1})
                rstats[m] = ranks(observed, p)
            rank_rows.append(rstats)

        reference = pp[pp.track.eq(track)].set_index("model_id")
        est = pd.DataFrame(episode_points).groupby("model_id").mean()
        point_columns = [*metrics.AXES, *metrics.RESIDUAL, "precision_at_1"]
        replay_point = check_errors(est.loc[reference.index, point_columns].to_numpy(),
                                    reference.loc[:, point_columns].to_numpy(), "LICOB point")
        jj = pc[pc.track.eq(track)]
        compare = read_tsv(checked(score_dir / "paired.tsv", old["files"]["paired.tsv"]))
        compare = compare[compare.track.eq(track)]
        if not jj.reset_index(drop=True).equals(compare.reset_index(drop=True)):
            raise ValueError("Published LICOB contrasts differ from the score receipt")
        legacy = fewshot.cross_bootstrap(y, arrays, splits, 5)
        reproduced = np.array([(est.loc[r.model_id, r.metric] - est.loc[r.baseline, r.metric],
            *metrics.interval(legacy[r.model_id][r.metric] - legacy[r.baseline][r.metric])) for r in jj.itertuples()])
        replay_ci = check_errors(reproduced, jj[list(EFFECT)].to_numpy(), "LICOB crossed intervals")
        inputs.append((track, panel, truth, jj, rank_rows, splits))
        replay_stats.append({"track": track, "original_point_replay": replay_point,
                             "original_crossed_replay": replay_ci, "splits": len(splits)})
        points.append(est)

    # The 3D markdown table presents D minus C for both models and metrics of interest.
    source_md = (ORIGINAL_REPORTS / "FEWSHOT_3D_CONFIRM.md").read_text()
    row = next(line for line in source_md.splitlines() if line.startswith(f"| LICOB / K{k} |")
               and len(line.strip("|").split("|")) == 7)
    if len(inputs) != 1 or not row:
        raise ValueError("Unexpected LICOB panel/report support")
    _, _, _, jj, _, _ = inputs[0]
    terms = row.strip("|").split("|")
    original_points = points[0]
    for i, model in enumerate(fewshot.D, start=3):
        r = jj[(jj.model_id == model) & (jj.baseline == fewshot.C) & jj.metric.eq("ndcg_at_5")].iloc[0]
        delta, ci = terms[i].strip().split(" ", 1)
        if not np.allclose([float(delta), *literal_eval(ci)], [r.delta, r.ci95_low, r.ci95_high],
                           atol=5e-5 + 1e-12, rtol=0):
            raise ValueError("Original LICOB NDCG markdown does not reproduce")
    if not np.allclose(list(map(float, terms[1:3])), original_points.loc[
        [fewshot.B, fewshot.C], "ndcg_at_5"], atol=5e-5 + 1e-12, rtol=0):
        raise ValueError("Original LICOB baseline markdown does not reproduce")
    json_write(out / "original_validation.json", {"study": "LICOB", "k": k,
        "original_report_score_receipt_sha256": sha256(score_path), "forecast_sha256": stamp["sha256"],
        "labels_sha256": registered_files["LICOB/test-outcomes.parquet"],
        "layout_sha256": registered_files[f"LICOB/k{k}-layout.json"], "registration_sha256": sha256(registered),
        "original_crossed_draws": old["paired_bootstrap_draws"], "panels": replay_stats,
        "published_markdown_original_ndcg_rows_reproduced": 4})

    old_gate = read_json(ORIGINAL_REPORTS / "fewshot_decision.json")
    records, gates = [], []
    w = weights(29, draws, seed)
    for track, panel, truth, jj, rank_rows, splits in inputs:
        fixed = {m: {} for m in (fewshot.B, fewshot.C, *fewshot.D)}
        for m in fixed:
            for ix, metric in enumerate(RANK_METRICS):
                episode_draws = np.stack([average_split(r[m][:, ix], w[:, idx], 2)
                                          for r, idx in zip(rank_rows, splits)], axis=1)
                fixed[m][metric] = metrics.mean(episode_draws)
        group_hash = axes_sha(truth.index)
        records.extend(contrast_row("LICOB", track, k, len(truth), panel, old["paired_bootstrap_draws"], r,
            fixed[r.model_id][r.metric] - fixed[r.baseline][r.metric], group_hash, draws)
            for r in jj[jj.metric.isin(RANK_METRICS)].itertuples())
        for model in fewshot.D:
            part = pd.DataFrame(records)
            ndcg = part[part.track.eq(track) & part.model_id.eq(model) & part.metric.eq("ndcg_at_5")]
            residual = jj[jj.model_id.eq(model) & jj.metric.eq("residual_spearman")]
            if len(ndcg) != 2 or len(residual) != 2:
                raise ValueError("Original gate is missing a required comparator")
            original_gate = bool(ndcg.original_low.gt(0).all() and residual.ci95_low.gt(0).all())
            fixed_gate = bool(ndcg.fixed_low.gt(0).all() and residual.ci95_low.gt(0).all())
            expected = next(r["joint_gate_pass"] for r in old_gate["stratum_gate"]
                            if r["track"] == track and r["model_id"] == model)
            if expected != original_gate:
                raise ValueError("Original few-shot gate mismatch")
            gates.append({"track": track, "model_id": model, "original_joint_gate": original_gate,
                          "fixed_counterfactual_joint_gate": fixed_gate})
    return pd.DataFrame(records), gates, 29, axes_sha(w.ravel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("LIMORE", "K2", "K4", "K8"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    plan = read_json(CONTRACT)
    args.out.mkdir()
    common = (args.out, plan["bootstrap_draws"], plan["p0_seed"])
    table, gates, count, weights_hash = (limore(*common) if args.task == "LIMORE"
                                         else licob(int(args.task[1:]), *common))
    output = args.out / "comparisons.tsv"
    table.to_csv(output, sep="\t", index=False, float_format="%.12g")
    json_write(args.out / "receipt.json", {"task": args.task, "analysis": "retrospective_fixed_panel_sensitivity",
        "independent_units": count, "weight_sha256": weights_hash, "fixed_panel_draws": plan["bootstrap_draws"],
        "fit_count": 0, "predictions_regenerated": False, "source_code_and_report_hashes": report_source_hashes(),
        "p0_git_commit": plan["p0_commit"], "positional_weights_reused_across_repeats": args.task != "LIMORE",
        "original_validation_sha256": sha256(args.out / "original_validation.json"),
        "fixed_table_sha256": sha256(output), "gates": gates})
    print(args.task, "original evidence verified and fixed-panel resampled", len(table), flush=True)


if __name__ == "__main__":
    main()
