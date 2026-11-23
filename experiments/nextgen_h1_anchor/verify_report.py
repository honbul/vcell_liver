"""Verify released H1 aggregates and freeze/score/CLI contracts without outcome access."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from report import MAIN, NDCG, P, STUDIES, rows

REQUIRED = """
PLAN_LOCK.md DATA_SUPPORT_AND_EXPOSURE.md ROUND1_TO_3_INNER_TUNING.md RESULTS.md
H1_METRICS.tsv H1_PAIRED_95CI.tsv H1_PANCANCER_BY_TISSUE.tsv H1_ANCHOR_POLICIES.tsv
model_card.md cli_replay_receipt.json score_receipt.json SLURM_JOBS.tsv
""".split()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify(root):
    for name in REQUIRED:
        need((root / name).stat().st_size > 0, f"{name} is empty")
    score = json.loads((root / "score_receipt.json").read_text())
    replay = json.loads((root / "cli_replay_receipt.json").read_text())
    controls = json.loads((root / "inner_control_receipt.json").read_text())
    inner = json.loads((root / "inner_freeze_summary.json").read_text())
    support = json.loads((root / "data_support_receipt.json").read_text())
    fit = json.loads((root / "fold_integrity_receipt.json").read_text())
    need(not fit["held_labels_read"] and fit["worst_tier1_group_fold_count"] == 1
         and fit["data_receipt_sha256"] == score["input_sha256"], "Independent outer fold audit")
    need(all(t["fit_test_intersection"] == 0 and ("_to_" not in task or t["rna_features"] == 0)
             for task, t in fit["tasks"].items()), "Fit/test overlap or transfer RNA")

    for file, expected in score["outputs"].items():
        need(digest(root / file) == expected, f"Score output hash mismatch: {file}")
    need(score["score_passes"] == 1 and score["paired_bootstrap_draws"] == 2000, "Single held-score contract")
    need(score["input_sha256"] == inner["label_receipt_sha256"] == controls["data_receipt_sha256"],
         "Training, audit and score input receipts disagree")
    need(score["forecast_receipt_sha256"] == replay["forecast_receipt_sha256"], "CLI used a different forecast")
    need(replay["replay_passed"] and replay["cpu_only"] and replay["max_abs_diff"] < 1e-6, "CPU replay gate")
    need(replay["tasks"]["LIMORE_to_Wong-0"]["rows_outside_cli_envelope"] == 10, "Wong envelope audit")
    need(all(t["tested_rows"] > 0 for name, t in replay["tasks"].items() if "Wong" not in name),
         "Supported packages need a quantitative CLI replay")

    need(sum(1 for s in score["masks"] if s.startswith("LIMORE_to_LICOB-")) == 1, "Tier 4 must be one source transfer")
    need(score["masks"]["LIMORE_to_LICOB-0"]["eligible"] == 38, "Tier 4 must cover all LICOB donors")
    need(score["masks"]["LIMORE_to_Wong-0"]["minimum_measured_q"] >= 15, "Wong eligibility")
    need(support["common_pan_n_by_density"]["0.8"] == 4
         and min(len(support["cohorts"][s]["panel"]) for s in STUDIES[1:4]) >= 40, "Pan-cancer panel audit")
    need(not controls["outer_outcomes_opened"] and not inner["used_held_labels"], "Inner audits must be outcome blind")
    need(digest(root / "H1_INNER_LOCKED_CONTROLS.tsv") == controls["table_sha256"], "Inner control table mismatch")
    need(set(inner["tasks"]) == set(score["masks"]) == set(replay["tasks"]) == set(controls["tasks"]),
         "Run task coverage mismatch")
    for task, stamp in inner["tasks"].items():
        need(stamp["freeze_sha256"] == score["masks"][task]["freeze"] == controls["freeze_sha256"][task],
             f"Changed freeze for {task}")
        need(score["masks"][task]["q_drugs"] >= 15, f"Insufficient Q panel for {task}")
        need(set(replay["tasks"][task]["invalid_input_exit_codes"].values()) == {2}, f"CLI refusal failure for {task}")
    inner_support = defaultdict(list)
    for line in rows(root, "H1_INNER_LOCKED_CONTROLS.tsv"):
        inner_support[line["task"], line["inner_fold"]].append(line)
    need(len(inner_support) == 3 * score["task_count"], "Inner three-fold coverage")
    for (task, _), block in inner_support.items():
        need(len({(x["A8"], x["q"], x["n"]) for x in block}) == 1, f"Inner support mismatch for {task}")
        for k in ("0", "3", "5", "8"):
            arms = {x["method"] for x in block if x["k"] == k}
            expected = {"B0", "B2", "B3", "B4", "N1", P}
            if "_to_" not in task:
                expected.update(("B1", "N2"))
            need(expected <= arms, f"Missing inner comparison K={k} in {task}")
    all_scores = rows(root, "H1_METRICS.tsv")
    for study in STUDIES:
        found = [r for r in all_scores if r["study"] == study and r["metric"] == NDCG]
        need(len({(r["observed_entries"], r["independent_groups"], r["q_min"], r["q_max"]) for r in found}) == 1,
             f"Outer support mismatch for {study}")
        core = {"B0", "B2", "B3", "B4", "Bsimple", "N1", P}
        if "_to_" not in study:
            core.update(("B1", "N2"))
        for k in ("0", "3", "5", "8"):
            need(core <= {r["method"] for r in found if r["k"] == k}, f"Missing outer comparison K={k} in {study}")
    for s in MAIN:
        subset = {(r["candidate"], r["comparator"], r["metric"]) for r in rows(root, "H1_PAIRED_95CI.tsv")
                  if r["study"] == s}
        for ref in ("Bsimple/5", "N1/5", f"{P}/0"):
            need((f"{P}/5", ref, NDCG) in subset, f"Missing predeclared primary contrast: {s}, {ref}")
    first = (root / "RESULTS.md").read_text().splitlines()[:5]
    need(all(s.startswith(f"**{key}**") for s, key in zip(first, ("가설", "달라진 점", "결과", "판정", "다음 결정"))),
         "RESULTS first five lines do not summarize the required decisions")
    jobs = rows(root, "SLURM_JOBS.tsv")
    need(sum(r["stage"] == "outer_score" for r in jobs) == 1, "Find exactly one outer score job in ledger")
    need(all(r["partition"] == "cpu" and r["job_name"].startswith("ng-h1-") and r["gpus"] == "0" for r in jobs),
         "Job resource contract")
    lock = json.loads((root / "outer_score_claim_receipt.json").read_text())
    need(lock["score_passes_started"] == 1
         and lock["legacy_score_claim_sha256"] == score["score_claim_sha256"]
         and lock["input_sha256"] == score["input_sha256"]
         and lock["forecast_sha256"] == score["forecast_receipt_sha256"]
         and lock["completed_score_receipt_sha256"] == digest(root / "score_receipt.json"),
         "Study-wide lock must bind the original score")
    affine_inner = json.loads((root / "b3_inner_correction_receipt.json").read_text())
    affine_forecasts = json.loads((root / "b3_forecast_rank_receipt.json").read_text())
    need(not affine_inner["outer_outcomes_opened"] and not affine_forecasts["outer_outcomes_opened"],
         "B3 implementation audit used a held outcome")
    need(digest(root / "H1_B3_INNER_CORRECTION.tsv") == affine_inner["table_sha256"]
         and affine_inner["data_receipt_sha256"] == score["input_sha256"]
         and affine_inner["freeze_sha256"] == controls["freeze_sha256"]
         and affine_inner["shift_penalty"] == affine_forecasts["shift_penalty"] == 5.0,
         "B3 inner correction provenance mismatch")
    need(affine_forecasts["forecast_receipt_sha256"] == score["forecast_receipt_sha256"]
         and affine_forecasts["anchor_receipt_sha256"] == score["anchor_receipt_sha256"]
         and set(affine_forecasts["tasks"]) == set(score["masks"]), "B3 forecast audit provenance mismatch")
    affine_table = rows(root, "H1_B3_INNER_CORRECTION.tsv")
    actual = {(r["task"], r["inner_fold"], r["k"]) for r in affine_table}
    expected = {(t, fold, k) for t, fold in inner_support for k in ("0", "3", "5", "8")}
    need(len(affine_table) == len(actual) == affine_inner["rows"] and actual == expected,
         "B3 correction must cover each original inner fold and budget")
    for row in affine_table:
        ref = inner_support[(row["task"], row["inner_fold"])][0]
        need((row["a8"], row["q_size"], row["samples"]) == (ref["A8"], ref["q"], ref["n"])
             and float(row["max_inner_rank_metric_change"]) <= 1e-12, "B3 inner rank audit or Q mismatch")
    for task, by_k in affine_forecasts["tasks"].items():
        target = score["masks"][task]
        need(set(by_k) == {"0", "3", "5", "8"}
             and all(item["query_drugs"] == target["q_drugs"]
                     and item["samples"] == target["eligible"]
                     and item["samples_with_any_rank_change"] == 0 for item in by_k.values()),
             f"B3 forecast rank equivalence or support failed for {task}")
    print(f"Verified {len(REQUIRED)} required deliverables, {score['task_count']} frozen tasks and one held score pass")
    print("Verified the separate B3 inner correction and frozen-forecast rank equivalence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    verify(parser.parse_args().root)
