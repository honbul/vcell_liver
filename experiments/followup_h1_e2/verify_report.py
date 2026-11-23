"""Verify published reports, inputs, numerical gates, and both resampling designs."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import re

from render_reports import (ALT, COMP, CURVES, INTERACTION, NDCG, PRECISION, REQUIRED, RHO,
                            SCREENS, UTILITY, load, tsv)

METRICS = (NDCG, PRECISION, RHO, "missed_true_top5_fraction", "selected_top5_actual_mean", "oracle_top5_regret")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def equal(a, b):
    return isclose(float(a), float(b), abs_tol=3e-10, rel_tol=3e-10)


def report_files(out):
    for name in REQUIRED:
        text = (out / name).read_text()
        starts = ("질문", "실제 바뀐 것", "주 비교 결과·구간", "판정", "다음 결정")
        require(all(line.startswith(f"**{tag}** — ") for line, tag in zip(text.splitlines()[:5], starts, strict=True)),
                f"Missing five-line summary in {name}")
        for path in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in path:
                require((out / path).is_file(), f"Broken {name} local source link {path}")


def exposures(out, stamp):
    contract = out.parents[1] / "experiments/followup_h1_e2/contract.json"
    agree = stamp["contract_sha256"]
    require(agree == digest(contract) == stamp["e2"]["contract_sha256"], "Lock hash mismatch")
    require(stamp["base_h1_v2_commit"] == "fa64b590a40b4d4c34a9b7b234a3a76c363ae322",
            "H1-v2 base changed")
    for name, digest_ in stamp["evidence_files"].items():
        require(digest(out / name) == digest_, f"Aggregate changed after shipping from G1: {name}")
    original = out.parent / "nextgen_h1_v2"
    require(digest(original / "score_receipt.json") == stamp["original_h1_v2_score_receipt_sha256"] and
            load(original / "score_receipt.json")["score_passes"] == stamp["original_h1_v2_score_passes"] == 1,
            "Original H1-v2 single pass changed")
    require(stamp["e2"]["score_passes"] == 1 and stamp["e2"]["bootstrap_draws"] == 2000 and
            not stamp["e2"]["query_values_in_forecast_stage"] and stamp["e2"]["h1_model_fit_count"] == 0,
            "New E2 run must have a label-blind forecast and one score")
    seal_path = out / "e2_forecast_seal_receipt.json"
    seal = load(seal_path)
    require(not seal["outer_q_outcomes_opened"] and digest(seal_path) == stamp["e2"]["sealed_forecast_gate_sha256"],
            "Pre-score gate mismatch")
    support = load(out / "curve_support_receipt.json")
    require(not support["licob_3d_per_dose_source_registered"] and support["query_support_only_before_scoring"],
            "2D assay provenance boundary missing")
    expected_folds = {f"{s}-{i}" for s in SCREENS for i in range(3)}
    require(set(support["folds"]) == set(seal["folds"]) == expected_folds, "Incomplete frozen study folds")
    historic_freeze = load(original / "frozen_receipt.json")
    historic_prediction = load(original / "prediction_receipt.json")
    for task, item in support["folds"].items():
        scope = item["scopes"]["primary"]
        scores = stamp["e2"]["q8_label_masks"][task]["scopes"]["primary"]
        require(not item["outer_q_outcomes_opened"] and item["same_original_q8"] and item["anchor_budget"] == 5 and
                not item["new_h1_model_fits"] and item["q8_parent_count"] == scores["q"] == 60 and
                item["common_curve_held_count"] == scores["n"] == scope["groups_n"] and
                item["held_exclusion_count"] == item["archived_original_held_count"]-scores["n"] and
                len(item["a8_curve_support_audit"]) == 8 and not item["original_reference_atlas"]["atlas_reused_marginal_wells"],
                f"Original atlas, Q8 or complete curve support changed: {task}")
        require(item["archive_freeze_receipt_sha256"] == historic_freeze["packages"][task] and
                item["raw_source_receipt_sha256"] == seal["sources"][item["study"]] and
                max(item["archived_native_forecast_replay_max_abs"].values()) < 1e-6,
                f"Original signed H1-v2 model was not replayed in {task}")
        require(task in historic_prediction["prediction_receipts"], "Original forecast provenance missing")
        for arm, parents in item["k5_exact_parents"].items():
            require(len(set(parents)) == 5 and set(parents) <= set(item["a8"]), f"Frozen K5 changed: {task} {arm}")
            rows = [item["a8_curve_support_audit"][d] for d in parents]
            for meta in rows:
                schedules = meta["dose_schedules_uM"]
                full = schedules["Full-curve"]
                for condition, points in schedules.items():
                    if points is None:
                        require(condition == "Reduced-8pt" and len(full) < 8, f"Unexpected missing grid {task}")
                        continue
                    require(set(points) <= set(full) and points == sorted(set(points)) and
                            (len(points) == 1 or points[0] == full[0] and points[-1] == full[-1]),
                            f"Reduced points changed physical interval {task}")
                require(meta["native_calibration_fit_only"]["n"] >= 20 and
                        meta["single_to_full_fit_only"]["n"] >= 20, f"Inadequate inner fit {task}")
            for name, cost in scope["well_cost"][arm].items():
                if name == "Full-curve-1rep":
                    expected = sum(len(x["dose_schedules_uM"]["Full-curve"]) for x in rows) + 12
                else:
                    points = [x["dose_schedules_uM"][name] for x in rows]
                    expected = sum(map(len, points))*item["primary_repeat_count"] + 12 if all(points) else None
                require(cost == expected, f"Incorrect physical well accounting in {task} {arm} {name}")
    return support


def e2_numbers(out, stamp):
    tab = tsv(out / "E2_METRICS.tsv")
    contrasts = tsv(out / "E2_PAIRED_95CI.tsv")
    expected_metrics = {metric for metric in METRICS}
    identify = ("screen", "scope", "level", "arm", "condition")
    cost = tsv(out / "E2_COST_AND_GATES.tsv")
    actual = {tuple(row[q] for q in identify) + (row["metric"],): row for row in tab}
    diffs = {(r["screen"], r["scope"], r["level"], r["model_condition"], r["reference"], r["metric"]): r
             for r in contrasts}
    require(len(actual) == len(tab) == len(cost)*len(expected_metrics) and
            len(diffs) == len(contrasts) and {r["metric"] for r in tab} == expected_metrics,
            "Metric or contrast grid is incomplete")
    primary = []
    for row in cost:
        key = tuple(row[q] for q in identify)
        screen, scope, level, arm, condition = key
        if scope == "primary" and level == "study":
            primary.append(row)
        for metric in expected_metrics:
            item = actual[key + (metric,)]
            require(all(item[k] == row[k] for k in ("lineages", "status", "observed_query_pairs", "q8_candidates_each_fold")),
                    "Metric conditions changed denominators")
        if row["status"] != "ESTIMATED" or not int(row["lineages"]):
            continue
        n = int(row["lineages"])
        ref = actual[(screen, scope, level, "No-anchors", "B0", NDCG)]
        require(int(ref["lineages"]) == n >= 20 and
                int(ref["observed_query_pairs"]) == int(row["observed_query_pairs"]), "Comparison denominators changed")
        lookup = lambda name: actual[key + (name,)]["value"]
        require(equal(float(lookup(PRECISION)) + float(lookup("missed_true_top5_fraction")), 1) and
                float(lookup("oracle_top5_regret")) >= -1e-12, "Top-5 metrics are inconsistent")
        if arm == "No-anchors" or condition == "Published-anchor":
            continue

        def r(base, metric):
            c = diffs[screen, scope, level, f"{arm}/{condition}", base, metric]
            require(int(c["bootstrap_nonmissing"]) == stamp["e2"]["bootstrap_draws"] and int(c["lineages"]) == n,
                    "Paired draw or denominator coverage changed")
            return c

        ordinary = []
        for baseline in ("B0", "B1"):
            ndcg, rho = (r(baseline, m) for m in (NDCG, RHO))
            threshold = float(ndcg["lo95"])
            utility = UTILITY if threshold > 0 else ALT if threshold >= -.01 and float(rho["lo95"]) > 0 else "UTILITY-NOT-PASSED"
            ordinary.append(utility)
            require(row[f"utility_vs_{baseline.lower()}"] == utility, "Wrong independent utility status")
        combined = ordinary[0] if ordinary[0] == ordinary[1] else (
            "UTILITY-NOT-PASSED" if "UTILITY-NOT-PASSED" in ordinary else ALT)
        require(row["utility_vs_both"] == combined, "Wrong combined utility")
        if condition in CURVES[1:]:
            full = f"{arm}/Full-curve"
            compression = (float(row["well_reduction_min_fraction"]) >= .2 and
                           float(r(full, NDCG)["lo95"]) >= -.01 and float(r(full, PRECISION)["lo95"]) >= -.02)
            require((row["compression"] == COMP) == compression, "Wrong compression threshold")
    successful = {(r["screen"], r["arm"], r["condition"]) for r in primary
                  if r["compression"] == COMP and r["utility_vs_both"] == UTILITY}
    require(successful == {("CTRPv2", "Dual", c) for c in ("Reduced-8pt", "Reduced-6pt", "Single-dose", "Full-curve-1rep")} |
                          {("GDSC2", "Dual", c) for c in ("Reduced-3pt", "Single-dose")}, "Unexpected study-level gate results")
    require(len([r for r in cost if r["scope"] == "two_rep_subset" and r["level"] == "study" and
                 r["condition"] == "Full-curve-1rep" and r["status"] == "INFEASIBLE"]) == 2,
            "GDSC2 two-replicate diagnostic should retain both unsupported rows")
    return len(tab), len(contrasts)


def p0_numbers(out, stamp):
    replay = load(out / "p0_replay_receipt.json")
    pairs = tsv(out / "FIXED_PANEL_CI_SENSITIVITY.tsv")
    require(replay["new_fits"] == stamp["p0"]["model_fits"] == 0 and replay["original_contract_preserved"] and
            stamp["p0"]["frozen_forecasts_reused"] and stamp["p0"]["original_joint_decisions_preserved"],
            "P0 must reuse every original forecast")
    counts = Counter((r["study"], r["k"]) for r in pairs)
    require(counts == {("LIMORE", "0"): 80, ("LICOB", "2"): 8, ("LICOB", "4"): 8, ("LICOB", "8"): 8},
            "Fixed-panel comparison coverage changed")
    studies = replay["analyses"]
    require(len({studies[f"K{k}"]["resample"]["weight_sha256"] for k in (2, 4, 8)}) == 1 and
            all(studies[f"K{k}"]["resample"]["positional_weights_reused_across_repeats"] for k in (2, 4, 8)),
            "Cohort-level LICOB weights were not shared")
    for job, trial in studies.items():
        src = trial["original_validation"]
        check = [src] if job == "LIMORE" else src["panels"]
        require(all(max(t[k]["max_abs_diff"] for k in ("original_point_replay", "original_crossed_replay")) < 5e-11
                    for t in check), "Original published point and crossed interval replay mismatch")
        changed = {g["model_id"] for g in trial["resample"]["gates"]
                   if g["original_joint_gate"] != g["fixed_counterfactual_joint_gate"]}
        require(changed == ({"structured", "ridge_structured_50_50"} if job == "LIMORE" else set()),
                "Counterfactual joint gate changed unexpectedly")
    for r in pairs:
        want = ("50", "68", "2400") if r["study"] == "LIMORE" else ("29", "53", "500")
        require((r["independent_n"], r["fixed_drugs"], r["original_draws"]) == want and
                r["fixed_draws"] == r["fixed_draws_nonmissing"] == "2000", "Fixed panel or draw support changed")
        for prefix in ("original", "fixed"):
            lo, hi = map(float, (r[f"{prefix}_low"], r[f"{prefix}_high"]))
            sign = "positive" if lo > 0 else "negative" if hi < 0 else "includes_zero"
            require(r[f"{prefix}_zero_test"] == sign, "Wrong P0 interval sign")
    return len(pairs)


def gates(out):
    ledger = tsv(out / "H1_GATE_EVIDENCE.tsv")
    origin = {}
    for v, p in (("v1", "nextgen_h1_anchor/H1_PAIRED_95CI.tsv"), ("v2", "nextgen_h1_v2/H1V2_PAIRED_95CI.tsv")):
        for r in tsv(out.parent / p):
            if r.get("panel", "Q8") == "Q8":
                origin[v, r["study"], r["candidate"], r["comparator"], r["metric"]] = r
    for row in ledger:
        source = origin[tuple(row[k] for k in ("version", "study", "candidate", "original_comparator", "metric"))]
        require(all(row[k] == source[k] for k in ("delta", "lo95", "hi95")), "Prior H1 gate figure changed")
    require({r["metric"] for r in ledger} == {NDCG, PRECISION, RHO, INTERACTION}, "H1 gate axes incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    out = parser.parse_args().reports.resolve()
    report_files(out)
    stamp = load(out / "score_receipt.json")
    exposures(out, stamp)
    metrics, pairs = e2_numbers(out, stamp)
    p0 = p0_numbers(out, stamp)
    gates(out)
    jobs = tsv(out / "SLURM_JOBS.tsv")
    require(len(jobs) == 25 and all(r["partition"] == "cpu" and r["gpu_count"] == "0" and
                r["name"].startswith("fu-h1-") and r["node"] in {"c3", "c4", "c5"} for r in jobs) and
            len([r for r in jobs if r["state"] == "COMPLETED" and r["scope"] == "published evidence"]) == 14,
            "Scientific jobs must stay in the isolated CPU scope")
    print(f"Verified 5 reports, 6 E2 folds, {metrics} metrics, {pairs} E2 CIs, {p0} P0 CIs, and {len(jobs)} CPU jobs")


if __name__ == "__main__":
    main()
