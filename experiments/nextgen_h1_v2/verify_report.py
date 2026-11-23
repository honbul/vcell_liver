"""Cross-check the H1-v2 release against held scoring, model closure and v1 support."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from render_reports import ANCHOR, DUAL, MAIN_STUDIES, NDCG, load, num

REQUIRED = """
PLAN_LOCK.md DATA_SUPPORT_AND_EXPOSURE.md ROUND1_TO_3_CEILING_TUNING.md
RESULTS.md H1V2_METRICS.tsv H1V2_PAIRED_95CI.tsv H1V2_FAMILY_BENCHMARK.tsv
H1V2_PANCANCER_BY_TISSUE.tsv H1V2_HCC_TKI.tsv model_card.md replay_receipt.json
score_receipt.json SLURM_JOBS.tsv
""".split()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify(root, v1):
    for file in REQUIRED:
        need((root / file).stat().st_size > 0, f"Missing or empty {file}")
    def read(file):
        return json.loads((root / file).read_text())

    score, inner = read("score_receipt.json"), read("inner_receipt.json")
    freeze, forecasts = read("frozen_receipt.json"), read("prediction_receipt.json")
    prelabel, replay = read("prelabel_receipt.json"), read("replay_receipt.json")
    claim = read("outer_score_claim.json")
    old = json.loads((v1 / "score_receipt.json").read_text())
    for rec in (score, inner):
        for name, value in rec["outputs"].items():
            need(digest(root / name) == value, f"Corrupt release output {name}")
    need(score["score_passes"] == claim["score_passes_started"] == 1 and score["bootstrap_draws"] == 2000,
         "Expected one held scorer and 2,000 group bootstrap draws")
    need(digest(root / "outer_score_claim.json") == score["claim_sha256"], "Changed score claim")
    need(score["freeze_sha256"] == claim["frozen_sha256"] == forecasts["frozen_sha256"]
         == digest(root / "frozen_receipt.json"), "Model freeze provenance mismatch")
    need(score["predictions_sha256"] == claim["predictions_sha256"] == digest(root / "prediction_receipt.json"),
         "Prediction seal mismatch")
    need(not prelabel["holdout_outcomes_opened"] and prelabel["sealed_models"] == score["freeze_sha256"]
         and prelabel["sealed_forecasts"] == score["predictions_sha256"], "Prelabel checks did not cover the score")
    need(score["prelabel_sha256"] == claim["prelabel_sha256"] == digest(root / "prelabel_receipt.json"),
         "Scoring claim did not verify the prelabel receipt")
    need(freeze["v1_study_sha256"] == claim["cohort_sha256"] == old["input_sha256"], "v1 lineage support changed")
    need(not any(rec["outer_labels_opened"] for rec in (freeze, forecasts, inner)),
         "One of the pre-score stages opened query outcomes")
    task_names = set(score["support"])
    need(len(task_names) == freeze["task_count"] == inner["tasks"] == 16
         and task_names == set(prelabel["tasks"]) == set(replay["tasks"]) == set(freeze["packages"]),
         "Frozen task coverage mismatch")
    for task, record in score["support"].items():
        base = old["masks"][task]
        need(record["primary_groups"] == base["eligible"] and record["primary_q"] == base["q_drugs"]
             and record["primary_pairs"] == base["eligible_query_entries"],
             f"The original Q8 metric support changed for {task}")
        need(record["auxiliary_q"] == record["primary_q"] - 4
             and record["auxiliary_groups"] <= record["primary_groups"], f"Invalid Q12 support: {task}")
    need(replay["device"] == "cpu" and replay["passed"] and replay["max_abs_diff"] < 1e-6
         and not replay["outer_query_labels_opened"], "CPU replay gate")
    for task, entries in replay["tasks"].items():
        modes = {ANCHOR} if "_to_" in task else {ANCHOR, DUAL}
        ks = {0, 1, 3, 5, 8, 12} if score["support"][task]["auxiliary_groups"] >= 2 else {0, 1, 3, 5, 8}
        need({(e["arm"], e["k"]) for e in entries} == {(m, k) for m in modes for k in ks},
             f"Incomplete replay support for {task}")
        need(all(e["missing_exit_code"] == e["outside_exit_code"] == 2 for e in entries if e["k"] == 5),
             f"Bad CLI invalid-input handling for {task}")
    families = load(root, "H1V2_INNER_WINNERS.tsv")
    expected = {(t, f, m) for t in task_names for f in "ABCD" for m in
                (("anchor",) if "_to_" in t else ("anchor", "dual"))}
    need(len(families) == len(expected) and {(r["task"], r["family"], r["modality"]) for r in families} == expected,
         "Inner family coverage mismatch")
    need(all(int(r["remaining"]) == 0 for r in families), "Unexamined winning numerical boundary")
    grid = defaultdict(set)
    for row in load(root, "H1V2_INNER_GRID.tsv"):
        data = json.loads(row["recipe"])
        if row["family"] in "BCD":
            label = data.get("objective") or data.get("rank_objective")
            grid[row["family"]].add(label)
    need({"l2", "huber", "rank", "dual"} <= grid["B"] and {"listnet", "pairwise"} <= grid["C"]
         and "rank" in grid["D"], "Required direct ranking losses were not evaluated")
    metrics = load(root, "H1V2_METRICS.tsv")
    old_metrics = {(row["study"], row["method"], row["k"], row["metric"]): row
                   for row in load(v1, "H1_METRICS.tsv")}
    ref = {"v1-Anchor-Best":"H1-Anchor-Best", "v1-RNA+Anchor":"H1-RNA+Anchor-Best", "v1-Bsimple":"Bsimple"}
    for row in metrics:
        if row["panel"] == "Q8" and row["arm"] in ref and row["k"] in {"0", "3", "5", "8"} and row["metric"] == NDCG:
            one = old_metrics[row["study"], ref[row["arm"]], row["k"], NDCG]
            need(abs(num(row["value"]) - num(one["value"])) < 1e-7, "v1 reference no longer matches original scores")
    for s in MAIN_STUDIES:
        expect = {ANCHOR, "B0", "B2", "B3", "B4", "Bsimple", "N1", "v1-Anchor-Best",
                  *(f"Family-{f}" for f in "ABCD")}
        if "_to_" not in s:
            expect.update((DUAL, "B1", "N2", "N1-Dual"))
        if s in {"Tier1-macro", "GDSC2", "CTRPv2", "PRISM20Q2"}:
            expect.add("v1-RNA+Anchor")
        for k in ("0", "1", "3", "5", "8", "12"):
            panel = "Q12" if k == "12" else "Q8"
            found = {r["arm"] for r in metrics if r["study"] == s and r["panel"] == panel
                     and r["k"] == k and r["metric"] == NDCG}
            need(expect <= found, f"Missing matched arms at {s}/{panel}/K{k}: {expect - found}")
    jobs = load(root, "SLURM_JOBS.tsv")
    need(sum(r["stage"] == "outer_score" and r["status"] == "COMPLETED" for r in jobs) == 1,
         "Expected one completed score in the Slurm ledger")
    need(all(r["job_name"].startswith("ng2-h1-") and int(r["gpu_each"]) <= 1
             and int(r["gpu_concurrent_cap"]) <= 2 for r in jobs), "Resource contract failed")
    five = (root / "RESULTS.md").read_text().splitlines()[:5]
    need(len(five) == 5 and all(line.startswith("**" + topic + "**") for line, topic in
         zip(five, ("가설", "달라진 점", "결과", "판정", "다음 결정"))), "Missing first five result lines")
    print(f"Verified {len(REQUIRED)} deliverables, 16 matched tasks, four model families and one held score pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("v1", type=Path)
    verify(**vars(parser.parse_args()))
