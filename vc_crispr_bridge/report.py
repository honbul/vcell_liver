"""Assemble the verified HepG2, Chronos and drug-cold scorecards."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .perturbseq import record
from .predict import checked, write_json


TASKS = {
    "hepg2": ("hepg2_metrics.tsv", ["panel", "model", "measured_targets", "centered_pearson",
                                  "centered_rmse", "top20_recall", "top20_signed_recall"]),
    "chronos": ("chronos_metrics.tsv", ["panel", "method", "models", "eligible_groups",
                                       "within_group_spearman", "rmse", "mae"]),
    "bridge": ("bridge_metrics.tsv", ["split", "support", "method", "folds", "within_group_spearman",
                                     "cell_ndcg5", "rmse"]),
}


def read_scores(root, task, filename):
    scores, predictions = (root / f"{task}-{name}" for name in ("scores", "predictions"))
    scored_path, frozen_path = scores / "receipt.json", predictions / "receipt.json"
    scoring, forecast = (json.loads(p.read_text()) for p in (scored_path, frozen_path))
    if scoring["predictions"] != record(frozen_path) or scoring["panel"] != forecast["panel"]:
        raise ValueError(f"Track 4 receipts disagree for {task}")
    for name in forecast["files"]:
        checked(predictions, name, forecast)
    return checked(scores, filename, scoring), {
        "scoring": record(scored_path), "predictions": record(frozen_path),
        "model_count": len(forecast["models"]),
    }


def markdown(data, columns):
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in data[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join("—" if pd.isna(x) else f"{x:.5f}" if isinstance(x, float)
                                    else str(x) for x in row) + " |")
    return "\n".join(rows)


def matching_bridge_choices(scores, outputs, panel_root, n, seed):
    manifest = json.loads((outputs / "receipt.json").read_text())
    if record(panel_root / "receipt.json") != manifest["panel"]:
        raise ValueError("Bridge label registry changed after freezing")
    features = {
        "chemistry": ("ecfp4/", "molformer/"),
        "genetic_only": ("genetic_observed/",),
        "joint": ("joint_observed/", "joint_predicted/"),
    }
    rng = np.random.default_rng(seed)
    variants, contrasts = [], []
    for fold, candidates in sorted(manifest["models"].items()):
        role = panel_root / fold
        keys = json.loads((role / "receipt.json").read_text())
        frozen = pd.read_parquet(checked(outputs, f"{fold}.parquet", manifest))
        truth = pd.read_parquet(checked(role, "labels.parquet", keys))
        joined = frozen.merge(truth, on="id", validate="one_to_one")
        methods = {
            name: max((k for k in candidates if k.startswith(prefix)),
                      key=lambda k: (candidates[k]["source_val_spearman"], -candidates[k]["source_val_rmse"]))
            for name, prefix in features.items()
        }
        for key, name in methods.items():
            variants.append(scores.loc[scores.panel.eq(fold) & scores.method.eq(name)].assign(family=key))
        subset = joined[["cell_group", "drug_id", "outcome", *methods.values()]]
        subset = subset.groupby(["cell_group", "drug_id"], as_index=False).mean()
        for drug, observations in subset.groupby("drug_id"):
            y = observations.outcome.to_numpy()
            if len(y) < 3 or not np.ptp(y):
                continue
            rho = {name: float(spearmanr(y, observations[m]).statistic) if observations[m].nunique() > 1
                   else 0. for name, m in methods.items()}
            contrasts.append({"split": fold.rsplit("-", 1)[0], "drug_id": drug,
                              "joint_gain": rho["joint"] - rho["chemistry"],
                              "genetic_only_gain": rho["genetic_only"] - rho["chemistry"]})
    selected = pd.concat(variants, ignore_index=True)
    paired = pd.DataFrame(contrasts)
    out = []
    for split, group in paired.groupby("split"):
        if group.drug_id.duplicated().any():
            raise ValueError(f"Duplicate test drug in {split}")
        for col in ("joint_gain", "genetic_only_gain"):
            vals = group[col].to_numpy()
            rep = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
            lo, hi = np.quantile(rep, [.025, .975])
            out.append({"split": split, "contrast": col, "test_drugs": len(vals),
                        "delta_spearman": vals.mean(), "ci_low": lo, "ci_high": hi})
    return selected, pd.DataFrame(out)


def bridge_summary(table):
    result = table.assign(split=table.panel.str.rsplit("-", n=1).str[0]).groupby(
        ["split", "support", "method"], as_index=False
    ).agg(folds=("panel", "nunique"), within_group_spearman=("within_group_spearman", "mean"),
          cell_ndcg5=("cell_ndcg5", "mean"), rmse=("rmse", "mean"))
    return result


def generate(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        stage = Path(d)
        source_records = {}
        text = ["# Track 4 — HepG2 Perturb-seq and genetic-to-chemical transfer",
                "Independent scorers verified frozen prediction manifests and held assay roles. "
                "Each panel is reported on the measured axis listed with its metrics."]
        for task, (file, columns) in TASKS.items():
            scored, info = read_scores(a.results, task, file)
            source_records[task] = info
            shutil.copyfile(scored, stage / file)
            table = pd.read_csv(scored, sep="\t")
            text.append("## " + {"hepg2": "HepG2 transcriptional effects", "chronos": "DepMap Chronos",
                                "bridge": "LIMORE drug and target holdout"}[task])
            if task == "bridge":
                selected, deltas = matching_bridge_choices(
                    table, a.results / "bridge-predictions", a.bridge_panels, a.bootstrap, a.seed,
                )
                selected.to_csv(stage / "bridge_validation_selected.tsv", sep="\t", index=False)
                deltas.to_csv(stage / "bridge_paired_deltas.tsv", sep="\t", index=False)
                text.extend([
                    "Each fold selects chemistry-only, genetic-only and joint predictors using only "
                    "its blocked validation. The contrasts below bootstrap paired test-drug Spearman deltas.",
                    markdown(deltas, ["split", "contrast", "test_drugs", "delta_spearman", "ci_low", "ci_high"]),
                    "The full fold-averaged comparison includes exact culture Chronos matches as a separate subset.",
                ])
                table = bridge_summary(table)
                table.to_csv(stage / "bridge_fold_average.tsv", sep="\t", index=False)
            text.append(markdown(table, columns))
        (stage / "RESULTS.md").write_text("\n\n".join(text) + "\n")
        write_json(stage / "receipt.json", {
            "inputs": source_records, "bridge_panel": record(a.bridge_panels / "receipt.json"),
            "bootstrap": a.bootstrap, "seed": a.seed, "files": {p.name: record(p) for p in sorted(stage.iterdir())},
        })
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "bridge-panels", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--seed", type=int, default=41123)
    a = p.parse_args()
    if a.bootstrap < 100:
        p.error("Provide at least 100 paired drug resamples")
    generate(a)


if __name__ == "__main__":
    main()
