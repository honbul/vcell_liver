"""Aggregate verified panel receipts into a publishable Track 6 scorecard."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .artifacts import file_record, new_directory, verify_record, write_json


FAMILIES = {
    "limore-cell-cold": ("limore-cell-cold",),
    "limore-drug-cold": tuple(f"limore-drug-cold-{i}" for i in range(5)),
    "gdsc2-drug-cold": tuple(f"gdsc2-drug-cold-{i}" for i in range(5)),
    "dili-scaffold": tuple(f"dili-scaffold-{i}" for i in range(5)),
    "dili-release-increment": ("dili-release-increment",),
}


def gather(results, scores, predictions):
    tables = []
    for task, panels in FAMILIES.items():
        for panel in panels:
            prediction = predictions / panel / "receipt.json"
            score = scores / panel / "receipt.json"
            data = json.loads(score.read_text())
            verify_record(prediction, data["frozen_prediction_receipt"])
            verify_record(prediction.with_name("predictions.parquet"), data["frozen_predictions"])
            metric_path = score.with_name("metrics.json")
            verify_record(metric_path, data["metrics"])
            for name, metrics in json.loads(metric_path.read_text()).items():
                feature, family = name.split("/") if "/" in name else (name, "baseline")
                tables.append({"task": task, "panel": panel, "representation": feature,
                               "head": family, "scored_rows": data["scored_rows"], **metrics})
            results[panel] = data
    return pd.DataFrame(tables)


def add_public_comparators(results, score_evidence, table):
    rows = []
    for task in ("limore-cell-cold", "dili-scaffold", "dili-release-increment"):
        for panel in FAMILIES[task]:
            prediction = results / "public_predictions" / panel / "receipt.json"
            score = results / "public_scores" / panel / "receipt.json"
            sidecar = json.loads(score.read_text())
            data_path = prediction.with_name("predictions.parquet")
            metric = score.with_name("metrics.json")
            verify_record(prediction, sidecar["frozen_prediction_receipt"])
            verify_record(data_path, sidecar["frozen_predictions"])
            verify_record(metric, sidecar["metrics"])
            matched = results / "predictions" / panel / "predictions.parquet"
            same_labels = sidecar["labels"] == score_evidence[panel]["labels"]
            original = pd.read_parquet(matched, columns=["id"]).sort_values("id").id.to_list()
            ids = pd.read_parquet(data_path, columns=["id"]).sort_values("id").id.to_list()
            if not same_labels or ids != original:
                raise ValueError(f"Public model support differs from the matched panel {panel}")
            for name, metrics in json.loads(metric.read_text()).items():
                rows.append({"task": task, "panel": panel, "representation": name,
                             "head": "public family", "scored_rows": len(ids), **metrics})
            score_evidence[panel]["public_family"] = sidecar
    return pd.concat([table, pd.DataFrame(rows)], ignore_index=True)


def summary_table(long_form):
    metric_names = ("within_drug_spearman", "cell_macro_ndcg5", "auroc", "auprc")
    key = ["task", "representation", "head"]
    metrics = long_form.groupby(key)[list(metric_names)].mean()
    for metric in metric_names:
        counts = long_form.groupby(key)[metric].count()
        metrics[f"{metric}_scored_folds"] = counts
    return metrics.reset_index()


def markdown_table(data, columns):
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in data.loc[:, columns].itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("—")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def model_evidence(store):
    rows = []
    receipts = {}
    active = ("chemistry-v1", "bulk-v2", "limore-v2", "protein-v1")
    for path in sorted(p for scope in active for p in (store / scope).glob("*/receipt.json")):
        record = json.loads(path.read_text())
        name = str(path.parent.relative_to(store))
        features = path.with_name("vectors.npz")
        verify_record(features, record["output"])
        rows.append({
            "store": name, "requested": record.get("n_requested", record.get("rows", record.get("genes"))),
            "embedded": record.get("n_embedded", record.get("embedded", record.get("genes"))),
            "dimension": record["dimension"], "device": record["device"],
            "seconds": record["wall_seconds"],
            "peak VRAM MiB": record.get("vram_peak_allocated_bytes", 0) / (1024 * 1024),
        })
        receipts[name] = {"receipt": file_record(path), "vectors": record["output"]}
    if not rows:
        raise ValueError("No verified embeddings")
    return pd.DataFrame(rows), receipts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("results", "store", "output"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    args = p.parse_args()
    score_evidence = {}
    table = gather(score_evidence, args.results / "scores", args.results / "predictions")
    table = add_public_comparators(args.results, score_evidence, table)
    averaged = summary_table(table)
    models, feature_evidence = model_evidence(args.store)
    with new_directory(args.output) as out:
        table.to_csv(out / "fold_scores.tsv", sep="\t", index=False)
        averaged.to_csv(out / "matched_heads.tsv", sep="\t", index=False)
        models.to_csv(out / "extraction.tsv", sep="\t", index=False)
        chunks = [
            "# Track 6 — foundation representation benchmark",
            "Scores are from separately verified frozen predictions. All means weight panel folds equally. "
            "Counts of scorable folds and assay groups are recorded in the TSVs.",
            "## Feature extraction",
            markdown_table(models, ["store", "requested", "embedded", "dimension",
                                    "device", "seconds", "peak VRAM MiB"]),
        ]
        for task, panels in FAMILIES.items():
            subset = averaged.loc[averaged.task.eq(task)]
            records = table.loc[table.task.eq(task)].drop_duplicates("panel")
            metric = ["auroc", "auprc"] if task.startswith("dili") else [
                "within_drug_spearman", "cell_macro_ndcg5",
            ]
            chunks.extend([f"## {task}",
                           f"{len(panels)} panels, {records.scored_rows.sum():,} predictions scored.",
                           markdown_table(subset, ["representation", "head", *metric])])
        chunks.extend([
            "## Public model families",
            "CellHit-style per-drug XGBoost was refit on normalized source AUC and evaluated on the "
            "identical LIMORE external cells. The published model uses assay-specific ln(IC50) and "
            "log-fold change and is unavailable for genuinely unseen compounds. "
            "The Chemprop-style D-MPNN was independently refit on identical DILI scaffold and "
            "release-increment partitions. The two model families share the benchmark's test IDs and label hashes.",
            "## External 3D data",
            "The LICOB RNA workbook failed the repository checksum and workbook parser. "
            "The LIMORE cell-cold table is the designated 2D fallback and contributes no standalone 3D score.",
        ])
        (out / "RESULTS.md").write_text("\n\n".join(chunks) + "\n")
        write_json(out / "receipt.json", {
            "panels": score_evidence,
            "feature_store": feature_evidence,
            "outputs": {p.name: file_record(p) for p in sorted(out.iterdir())},
        })


if __name__ == "__main__":
    main()
