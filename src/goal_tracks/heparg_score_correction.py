"""Repair two score reference self-comparisons without touching any frozen prediction."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def fix_table(source, target, fields_by_arm):
    count = 0
    reader = gzip.open if source.suffix == ".gz" else Path.open
    with reader(source, "rt", newline="") as inp, target.open("wb") as raw:
        container = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) if source.suffix == ".gz" else raw
        with io.TextIOWrapper(container, encoding="utf-8", newline="") as out:
            rows = csv.DictReader(inp, delimiter="\t")
            writer = csv.DictWriter(out, fieldnames=rows.fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                if row["arm"] in fields_by_arm:
                    for key, (value, col) in fields_by_arm[row["arm"]].items():
                        row[key] = row[col] if col else value
                    count += 1
                writer.writerow(row)
    return count


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    cli.add_argument("--correction-commit", required=True)
    args = cli.parse_args()
    source = args.work / "scored"
    out = args.work / "scored-corrected"
    rec = json.loads((source / "receipt.json").read_text())
    if digest(args.work / "freeze.json") != rec["frozen_sha256"]:
        raise ValueError("Original scoring receipt does not bind the prediction freeze")
    if {f: digest(source / f) for f in rec["files"]} != rec["files"]:
        raise ValueError("Original score tables changed")

    metrics_cols = {
        "dose_shared_ridge": {
            "rmse_single_study_ridge": (None, "rmse_model"),
            "reduction_vs_ridge_pct": ("0.0", None), "ridge_lower95": ("0.0", None),
            "ridge_upper95": ("0.0", None), "passes_single_study_ridge": ("False", None),
        },
        "B_context": {
            "rmse_context": (None, "rmse_model"),
            "reduction_vs_context_pct": ("0.0", None), "context_lower95": ("0.0", None),
            "context_upper95": ("0.0", None),
        },
    }
    row_cols = {"dose_shared_ridge": {"ridge": (None, "row_rmse")}, "B_context": {"context": (None, "row_rmse")}}
    out.mkdir(exist_ok=False)
    changes = {}
    for name in rec["files"]:
        if name == "metrics.tsv":
            changes[name] = fix_table(source / name, out / name, metrics_cols)
        elif name.endswith("-paired.tsv.gz"):
            changes[name] = fix_table(source / name, out / name, row_cols)
        else:
            (out / name).write_bytes((source / name).read_bytes())
    rec.update(
        previous_score_receipt_sha256=digest(source / "receipt.json"),
        correction_commit=args.correction_commit,
        correction="Exact self-reference equality for dose_shared_ridge and B_context; other arm records preserved",
        self_reference_rows=changes,
        files={f: digest(out / f) for f in rec["files"]},
    )
    (out / "receipt.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
