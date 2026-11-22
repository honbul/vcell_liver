"""Audit the registered R18 score artifacts without reopening expression or predictions."""

import argparse
import csv
import math
from pathlib import Path

from src.r11_chemical_probe.common import read_json, sha256, write_json


SCOPES = {
    "C1": {"A": (502, 2142), "B": (500, 2137)},
    "C3_6h_HEPG2_only": {"A": (68, 239), "B": (12, 59)},
    "C3_6h_HUH7_added": {"A": (68, 239), "B": (12, 59)},
    "C3_24h_HEPG2_only": {"A": (67, 218), "B": (14, 53)},
    "C3_24h_HUH7_added": {"A": (67, 218), "B": (14, 53)},
}
FILES = {
    "points.tsv", "paired.tsv", "repeat_points.tsv", "repeat_fraction.tsv", "contributions.tsv",
    "robustness.tsv", "vehicle_alignment.tsv", "selection.tsv", "history.tsv", "decision.txt",
}
REFERENCES = ("B0", "B_pos", "R15", "best-VC", "best-published", "best-project")
METRICS = ("rmse", "r2_pooled")


def tsv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def primary_decision(pairs):
    selected = [r for r in pairs if r["scope"] == "C1" and r["arm"] == "blend_FiLM"]
    keys = {(scenario, ref, metric) for scenario in "AB"
            for ref in (*REFERENCES, *(("B_cell_gain",) if scenario == "B" else ())) for metric in METRICS}
    observed = {(r["scenario"], r["reference"], r["metric"]) for r in selected}
    if observed != keys or len(selected) != len(keys):
        raise ValueError("Expected each of the registered C1 primary comparisons exactly once")
    totals = {s: {"passed": 0, "comparisons": 0, "failed": []} for s in "AB"}
    for row in selected:
        low, high = (float(row[k]) for k in ("low99", "high99"))
        valid = math.isfinite(low) and math.isfinite(high) and low <= high
        passing = valid and row["eligible"] == "True" and (high < 0 if row["metric"] == "rmse" else low > 0)
        if passing != (row["passes"] == "True"):
            raise ValueError("Recorded C1 comparison disagrees with the registered 99% rule")
        group = totals[row["scenario"]]
        group["comparisons"] += 1
        group["passed"] += int(passing)
        if not passing:
            group["failed"].append([row["reference"], row["metric"]])
    good = {s: c["passed"] == c["comparisons"] for s, c in totals.items()}
    label = "확인되지 않음"
    if good["B"]:
        label = "확인됨" if good["A"] else "전이 조건에서만 확인됨"
    return label, totals


def support(points, doses):
    expected = {(case, scenario): values for case, scenarios in SCOPES.items() for scenario, values in scenarios.items()}
    for row in doses:
        if row["dose_uM"] == "all":
            continue
        for scenario in "AB":
            parents, wells = (int(row[f"{scenario}_{key}"]) for key in ("parents", "physical_wells"))
            if parents:
                expected[f"C2:{row['dose_uM']}", scenario] = (parents, wells)
    observed = {}
    for row in points:
        if row["arm"] != "B0":
            continue
        key = row["scope"], row["scenario"]
        parents, n = int(row["parents"]), int(row["rows"])
        if key in observed or n % 3:
            raise ValueError("Duplicate support row or inconsistent DMSO role count")
        observed[key] = parents, n//3
    if observed != expected:
        changed = sorted(key for key in expected.keys() | observed.keys() if expected.get(key) != observed.get(key))
        raise ValueError(f"Scored parent/well support differs from the locked audit: {changed[:5]}")
    return observed


def audit(run, reports):
    score = run/"score"
    receipt = read_json(score/"receipt.json")
    freeze = read_json(run/"freeze.json")
    digest = sha256(run/"freeze.json")
    if (digest != (run/"freeze.sha256").read_text().strip() or digest != receipt["freeze_sha256"]
            or len(freeze["prediction_receipts"]) != 108 or receipt["slices"] != 108
            or set(receipt["index_sha256"]) != set(freeze["prediction_receipts"])):
        raise ValueError("Score receipt does not bind all 108 previously frozen slices")
    if (freeze["protocol_sha256"] != sha256(reports/"R18_CONFIRMATION_PROTOCOL.md")
            or freeze["access_sha256"] != sha256(reports/"data/preflight/r18/access/receipt.json")
            or freeze["score_code_sha256"] != receipt["evaluator_code_sha256"]
            or set(receipt["files"]) != FILES):
        raise ValueError("Scorer, protocol or artifact list differs from the registered contract")
    for name, expected in receipt["files"].items():
        if sha256(score/name) != expected:
            raise ValueError(f"Score artifact does not match its receipt: {name}")
    label, totals = primary_decision(tsv(score/"paired.tsv"))
    if label != (score/"decision.txt").read_text().strip():
        raise ValueError("Independent application of the gate disagrees with decision.txt")
    scopes = support(tsv(score/"points.tsv"), tsv(reports/"data/c2_dose_support.tsv"))
    return dict(label=label, c1=totals, freeze_sha256=digest, score_receipt_sha256=sha256(score/"receipt.json"),
                slices=108, score_artifacts=len(FILES), scope_scenarios=len(scopes), score_expression_reopened=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "reports", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, audit(args.run, args.reports))
