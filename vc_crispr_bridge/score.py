"""Score a frozen HepG2 transfer bundle on the registered observations."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .metrics import perturbation_metrics, row_correlation
from .perturbseq import record
from .predict import PANELS, checked, write_json


def paired_intervals(prediction, reference, truth, indices):
    y = truth - truth.mean(axis=0)
    model = prediction - prediction.mean(axis=0)
    baseline = reference - reference.mean(axis=0)
    model_mse = np.mean(np.square(model - y), axis=1)
    baseline_mse = np.mean(np.square(baseline - y), axis=1)
    corr_delta = row_correlation(model, y) - row_correlation(baseline, y)
    differences = {
        "centered_rmse_delta": np.sqrt(model_mse[indices].mean(axis=1))
        - np.sqrt(baseline_mse[indices].mean(axis=1)),
        "centered_pearson_delta": corr_delta[indices].mean(axis=1),
    }
    return {f"{metric}_{suffix}": float(value)
            for metric, values in differences.items()
            for suffix, value in zip(("lo", "hi"), np.quantile(values, (0.025, 0.975)))}


def score(panel, predictions, output, bootstrap, seed):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_record = record(panel / "receipt.json")
    required = json.loads((panel / "receipt.json").read_text())
    receipt_path = predictions / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if panel_record != receipt["panel"]:
        raise ValueError("Prediction receipt does not match the registered panel")
    truth_axis = checked(panel, "genes.tsv", required)
    found_axis = checked(predictions, "genes.tsv", receipt)
    if record(truth_axis) != record(found_axis):
        raise ValueError("Unaligned transcript gene axis")
    center = np.load(checked(predictions, "source_mean.npy", receipt), allow_pickle=False)
    rng = np.random.default_rng(seed)
    scores = []
    for name in PANELS:
        label_path = checked(panel, f"labels_{name}/delta.npy", required)
        label_ids = checked(panel, f"labels_{name}/identities.parquet", required)
        labels = pd.read_parquet(label_ids)
        outcomes = np.load(label_path, allow_pickle=False)
        if not labels.id.is_unique or len(labels) != len(outcomes):
            raise ValueError(f"Invalid observed outcomes in {name}")
        with np.load(checked(predictions, f"{name}.npz", receipt), allow_pickle=False) as frozen:
            if set(frozen.files) != set(receipt["models"]) | {"id"}:
                raise ValueError(f"Unexpected model columns in {name}")
            order = pd.Index(labels.id).get_indexer(frozen["id"])
            if len(order) != len(labels) or (order < 0).any() or len(np.unique(order)) != len(order):
                raise ValueError(f"Prediction IDs are incomplete or repeated in {name}")
            y = outcomes[order]
            folds = rng.integers(len(y), size=(bootstrap, len(y)))
            comparator = frozen["ridge"]
            for method in sorted(receipt["models"]):
                values = frozen[method]
                scores.append({
                    "panel": name, "model": method, **perturbation_metrics(values, y, center),
                    **paired_intervals(values, comparator, y, folds),
                })
    table = pd.DataFrame(scores)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as directory:
        staged = Path(directory)
        score_path = staged / "hepg2_metrics.tsv"
        table.to_csv(score_path, sep="\t", index=False, float_format="%.8g")
        write_json(staged / "receipt.json", {
            "predictions": record(receipt_path), "panel": panel_record,
            "paired_reference": "ridge",
            "interval": "Percentile pair bootstrap across target genes on panel-centered residuals",
            "bootstrap": bootstrap, "seed": seed, "score_rows": len(table),
            "files": {score_path.name: record(score_path)},
        })
        staged.rename(output)


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("chronos", "bridge"):
        if sys.argv[1] == "chronos":
            from . import chronos as task
        else:
            from . import bridge as task

        del sys.argv[1]
        task.score(task.arguments(score_mode=True))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["hepg2"], metavar="{hepg2,chronos,bridge}")
    for name in ("panel", "predictions", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=41027)
    a = p.parse_args()
    if a.bootstrap < 1:
        p.error("--bootstrap must be positive")
    score(a.panel, a.predictions, a.output, a.bootstrap, a.seed)


if __name__ == "__main__":
    main()
