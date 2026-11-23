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
    for file, expected in score["outputs"].items():
        need(digest(root / file) == expected, f"Score output hash mismatch: {file}")
    need(score["score_passes"] == 1 and score["paired_bootstrap_draws"] == 2000, "Single held-score contract")
    need(score["input_sha256"] == inner["label_receipt_sha256"] == controls["data_receipt_sha256"],
         "Training, audit and score input receipts disagree")
    need(score["forecast_receipt_sha256"] == replay["forecast_receipt_sha256"], "CLI used a different forecast")
    need(replay["replay_passed"] and replay["cpu_only"] and replay["max_abs_diff"] < 1e-6, "CPU replay gate")
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
    print(f"Verified {len(REQUIRED)} required deliverables, {score['task_count']} frozen tasks and one held score pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    verify(parser.parse_args().root)
