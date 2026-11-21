"""Measure directional accuracy and common rescaling on already scored R12 predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import table

from .objectives import GainScores


ARMS = ("A1", "A4", "L1", "C2", "C3", "J2_calibrated")
GAINS = np.array([0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4])


def verified(path, hashes):
    if sha256(path) != hashes[path.name]:
        raise ValueError(f"R12 artifact changed: {path}")
    return path


def diagnose(predictions, scored, out, minimum):
    score_receipt = read_json(scored / "receipt.json")
    records = []
    for fold in range(3):
        for role in "AB":
            folder = predictions / f"outer{fold}" / role
            inference_receipt = read_json(folder / "receipt.json")
            hashes = inference_receipt["files"]
            truth = verified(scored / f"outer{fold}-{role}-truth.npz", score_receipt["files"])
            with np.load(truth, allow_pickle=False) as data:
                measured = data["score_delta"]
                if not np.array_equal(np.load(verified(folder / "panel.npy", hashes)), data["panel"]):
                    raise ValueError(f"R12 panel mismatch in {folder}")
            rows = table(verified(folder / "rows.tsv", hashes))
            index = read_json(verified(folder / "prediction_index.json", hashes))
            scores = GainScores(rows, measured, np.zeros_like(measured), minimum)
            for arm in ARMS:
                for seed, relative in index[arm].items():
                    path = folder / relative
                    if sha256(path) != hashes[relative]:
                        raise ValueError(f"R12 prediction changed: {path}")
                    response = np.load(path, allow_pickle=False)
                    coverage, cosine = scores.diagnostics(response)
                    errors, _ = scores.points(response, GAINS)
                    records.extend(
                        dict(arm=arm, fold=fold, role=role, seed=seed, gain=gain, contrast=error,
                             group_weight=scores.metric.primary_weight.sum(), coverage=coverage, cosine=cosine)
                        for gain, error in zip(GAINS, errors)
                    )
    detail = pd.DataFrame(records)
    by_task = detail.groupby(["arm", "gain", "fold", "role"], as_index=False).mean(numeric_only=True)
    summary = []
    for (arm, gain), group in by_task.groupby(["arm", "gain"]):
        summary.append(dict(arm=arm, gain=gain, **{
            metric: np.average(group[metric], weights=group.group_weight)
            for metric in ("contrast", "coverage", "cosine")
        }))
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summary).to_csv(out / "r12_scale.tsv", index=False, sep="\t")
    detail.to_csv(out / "r12_scale_by_fit.tsv", index=False, sep="\t")
    write_json(out / "receipt.json", dict(
        analysis="Retrospective, R12 outcomes used for diagnosis", minimum_groups=minimum,
        source_score_sha256=sha256(scored / "receipt.json"),
        source_fit_sha256={f"outer{i}/{role}": sha256(predictions / f"outer{i}" / role / "receipt.json")
                           for i in range(3) for role in "AB"},
        files={p.name: sha256(p) for p in out.iterdir()},
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-groups", type=int, required=True)
    args = parser.parse_args()
    diagnose(args.predictions, args.scored, args.out, args.minimum_groups)


if __name__ == "__main__":
    main()
