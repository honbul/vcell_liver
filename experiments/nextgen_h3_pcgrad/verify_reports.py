"""Check delivered H3 exhibits against the one-pass analysis receipts."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

REQUIRED = {
    "PLAN_LOCK.md", "DATA_SUPPORT_AND_EXPOSURE.md", "PRIOR_EXPERIMENT_REUSE.md", "GRADIENT_DIAGNOSTICS.md",
    "ROUND1_TO_3_INNER_TUNING.md", "RESULTS.md", "H3_METRICS.tsv", "H3_PAIRED_95CI.tsv",
    "H3_PANCANCER_BY_TISSUE.tsv", "model_card.md", "score_receipt.json", "SLURM_JOBS.tsv",
}
SCREENS = ("CTRPv2", "GDSC2", "PRISM20Q2")
CORE = ("ndcg_at_5", "within_drug_spearman", "precision_at_k")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tsv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def verify(root):
    for name in REQUIRED:
        if not (root / name).is_file() or not (root / name).stat().st_size:
            raise ValueError(f"Missing or empty deliverable: {name}")
    first = (root / "RESULTS.md").read_text().splitlines()[:5]
    for actual, label in zip(first, ("가설", "달라진 점", "결과", "판정", "다음 결정")):
        if not actual.startswith(f"- **{label}**"):
            raise ValueError(f"The first five lines omit {label}")
    for file in root.glob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", file.read_text()):
            if "://" not in link and not link.startswith("#") and not (file.parent / link.split("#")[0]).exists():
                raise ValueError(f"Broken link in {file.name}: {link}")
    receipt = json.loads((root / "score_receipt.json").read_text())
    audit = json.loads((root / "verification.json").read_text())
    if receipt["score_passes"] != 1 or receipt["screen_score_passes"] != dict.fromkeys(SCREENS, 1):
        raise ValueError("Unexpected number of held score passes")
    for name, sha in receipt["output_sha256"].items():
        if digest(root / name) != sha:
            raise ValueError(f"Detached table changed: {name}")
    if not audit["all_forecast_keys_verified"] or audit["verification_loaded_outer_outcomes"]:
        raise ValueError("Independent audit failed to prove frozen support")
    if audit["score_receipt_sha256"] != digest(root / "score_receipt.json"):
        raise ValueError("Independent audit validated a different held score")
    if receipt["final_lock_sha256"] != digest(root / "final_lock.json"):
        raise ValueError("Final operating points hash differs")
    if receipt["data_manifest_sha256"] != digest(root / "split_manifest.json"):
        raise ValueError("Corrected data split manifest hash differs")

    metrics = tsv(root / "H3_METRICS.tsv")
    table = {(r["screen"], r["model"], r["metric"], r["seed"]): r for r in metrics}
    if len(table) != len(metrics) or {r["screen"] for r in metrics} != {*SCREENS, "macro"}:
        raise ValueError("Duplicate scored keys or a missing held screen")
    for (screen, model, metric, seed), data in table.items():
        if seed != "mean":
            continue
        repeat = [float(table[screen, model, metric, str(s)]["value"]) for s in (11711, 11729, 11743)]
        if not math.isclose(mean(repeat), float(data["value"]), abs_tol=2e-15):
            raise ValueError("Seed average mismatch")
        if not math.isclose(stdev(repeat), float(data["seed_sd"]), abs_tol=2e-15):
            raise ValueError("Seed SD mismatch")

    intervals = tsv(root / "H3_PAIRED_95CI.tsv")
    contrasts = {(r["hypothesis"], r["candidate"], r["baseline"], r["screen"], r["metric"]): r for r in intervals}
    if len(contrasts) != len(intervals):
        raise ValueError("Duplicated analysis contrast")
    for gate in receipt["gates"]:
        key = (gate["hypothesis"], gate["candidate"], gate["baseline"])

        def value(screen, metric, field):
            return float(contrasts[(*key, screen, metric)][field])

        expected = {
            "ndcg_gain": value("macro", CORE[0], "delta") >= .01,
            "ndcg_ci_positive": value("macro", CORE[0], "lower") > 0,
            "drug_spearman_noninferior": value("macro", CORE[1], "lower") >= -.01,
            "precision_noninferior": value("macro", CORE[2], "lower") >= -.01,
            "screens_positive": sum(value(s, CORE[0], "delta") > 0 for s in SCREENS) >= 2,
            "no_severe_loss": min(value(s, m, "delta") for s in SCREENS for m in CORE) >= -.02,
        }
        if gate["criteria"] != expected or gate["pass"] != all(expected.values()):
            raise ValueError("A threshold was not enforced")
    if any(int(row["replicates"]) != 1200 or row["bootstrap_unit"] != "global_lineage" for row in intervals):
        raise ValueError("Paired intervals must use one shared global-lineage draw")
    for name, verdict in receipt["decisions"].items():
        registered = [gate["pass"] for gate in receipt["gates"] if gate["hypothesis"] == name and gate["required"]]
        if (verdict["verdict"] == "GO-DEV") != (bool(registered) and all(registered)):
            raise ValueError("Promotion decision differs from registered gates")
    if all(item["verdict"] != "GO-DEV" for item in receipt["decisions"].values()):
        tissue = tsv(root / "H3_PANCANCER_BY_TISSUE.tsv")
        if [r["status"] for r in tissue] != ["SKIPPED_NO_GO_DEV"]:
            raise ValueError("The expanded tissue route was not explicitly stopped")
    print(f"{len(REQUIRED)} required deliverables, {len(metrics)} scores, "
          f"{len(intervals)} paired intervals, and all score gates verified")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("reports", type=Path)
    verify(p.parse_args().reports)
