"""Recover receipt-verified existing R14 aggregate tables without rescoring."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover(root, ledger, out):
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for task, entry in ledger["r14"].items():
        if not isinstance(entry, dict):
            continue
        target = out / task
        target.mkdir()
        freeze = root / entry["freeze_path"]
        score = root / entry["score_receipt_path"]
        checks = [(freeze, entry["freeze_sha256"]), (score, entry["score_receipt_sha256"])]
        if score.is_file() and sha(score) == entry["score_receipt_sha256"]:
            receipt = json.loads(score.read_text())
            if receipt.get("freeze_sha256") != entry["freeze_sha256"]:
                raise ValueError(f"{task}: score receipt references a different freeze")
            checks.extend((score.parent / name, digest) for name, digest in receipt["files"].items())
        for path, expected in checks:
            actual = sha(path) if path.is_file() else ""
            state = "MATCH" if actual == expected else "MISMATCH" if actual else "MISSING"
            if state == "MATCH":
                shutil.copyfile(path, target / path.name)
            rows.append((task, str(path), expected, actual, state))

    with (out / "hash_reconciliation.tsv").open("w") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("task", "original_path", "expected_sha256", "actual_sha256", "status"))
        writer.writerows(rows)
    return all(row[-1] == "MATCH" for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if recover(args.root, json.loads(args.ledger.read_text()), args.out) else 1)
