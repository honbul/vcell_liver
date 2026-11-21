"""Recalculate official effects directly from frozen vectors and grouped weights."""

import argparse
from pathlib import Path

import numpy as np

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, POOL, group_weights, table, weights_for

from .score import FORMAL


def readout(rows, error, weights, draw, contrast):
    if not contrast:
        return np.dot(weights * draw, np.sqrt(np.mean(error**2, axis=1)))
    total = 0.0
    for idx in rows.groupby(POOL, sort=False).indices.values():
        part = rows.iloc[idx]
        center_weight = weights_for(part, (GROUP, "compound_entity_id", KEY[0])) * draw[idx]
        centered = error[idx] - np.average(error[idx], axis=0, weights=center_weight)
        total += np.dot(weights[idx] * draw[idx], np.sqrt(np.mean(centered**2, axis=1)))
    return total


def audit(cfg, scored, out):
    receipt = read_json(scored / "receipt.json")
    for name, expected in receipt["files"].items():
        if sha256(scored / name) != expected:
            raise ValueError(f"Scored artifact differs from receipt: {name}")
    freeze_path = scored.parent / ("scaffold-freeze.json" if scored.name.endswith("-scaffold") else "freeze.json")
    if sha256(freeze_path) != receipt["freeze_sha256"]:
        raise ValueError("Scorer used a different freeze")
    freeze = read_json(freeze_path)
    predictions = Path(freeze["predictions"])
    weights = np.load(scored / "bootstrap_weights.npz")
    draw_matrix = weights["draws"]
    if len(draw_matrix) != cfg["evaluation"]["replicates"] or (draw_matrix <= 0).any():
        raise ValueError("Bootstrap draw size/positivity differs from specification")
    assert np.max(np.abs(draw_matrix.mean(axis=1) - 1)) < 1e-12
    groups = dict(zip(weights["groups"], range(len(weights["groups"]))))
    effects = table(scored / "paired_effects.tsv")
    designated = {"point": np.ones(len(groups)), "first": draw_matrix[0], "last": draw_matrix[-1]}
    numerator = {tag: {name: 0.0 for name in FORMAL} for tag in designated}
    denominator = dict.fromkeys(designated, 0.0)
    actual_arms = []
    for k in range(3):
        for role in ("A", "B"):
            directory = predictions / f"outer{k}" / role
            ref = read_json(directory / "receipt.json")
            dirname = str(directory.relative_to(predictions))
            if sha256(directory / "receipt.json") != freeze["predictions_receipts"][dirname + "/receipt.json"]:
                raise ValueError("Inference does not match pre-score freeze")
            rows = table(directory / "rows.tsv")
            index = read_json(directory / "prediction_index.json")
            w = group_weights(rows)
            with np.load(scored / f"outer{k}-{role}-truth.npz") as z:
                targets = {"absolute": z["absolute"] - z["own_control"], "contrast": z["score_delta"]}
            lookup = rows[GROUP].map(groups).to_numpy()
            local_draws = {tag: values[lookup] for tag, values in designated.items()}
            required = {"A1", "A2", "A3", "A4", "A5", "A6", "L1", "L2", "L3", "L5", "L6",
                        "C1", "C2", "C3", "J1_FiLM_L2", "C5_seed42", "C5_seed46"}
            if not required <= set(index):
                raise ValueError(f"Missing mandatory matrix predictions {required - set(index)}")
            actual_arms.append(dict(role=role, outer=k, rows=len(rows), groups=rows[GROUP].nunique(),
                                    arms=len(index), seed_vector_references=sum(map(len, index.values()))))
            calc = {}
            for tag, draw in local_draws.items():
                denominator[tag] += w @ draw
                for label, (metric, terms) in FORMAL.items():
                    for arm, coef in terms.items():
                        pieces = []
                        for filename in index[arm].values():
                            key = (tag, metric, filename)
                            if key not in calc:
                                pred = np.load(directory / filename)
                                if sha256(directory / filename) != ref["files"][filename]:
                                    raise ValueError("Consumed frozen prediction hash differs")
                                err = pred.astype(np.float64) - targets[metric]
                                calc[key] = readout(rows, err, w, draw, metric == "contrast")
                            pieces.append(calc[key])
                        numerator[tag][label] += coef * np.mean(pieces)
    with np.load(scored / "bootstrap_effects.npz") as boot:
        discrepancies = {}
        for label in FORMAL:
            row = effects.loc[effects.contrast.eq(label) & effects.scope.eq("all")].iloc[0]
            draws = boot[f"{label}|all"]
            reference = dict(point=row.effect, first=draws[0], last=draws[-1])
            actual = {k: numerator[k][label] / weight for k, weight in denominator.items()}
            discrepancies[label] = float(max(abs(actual[k] - v) for k, v in reference.items()))
        interval_error = 0.0
        for row in effects.itertuples():
            z = boot[f"{row.contrast}|{row.scope}"]
            limits = np.percentile(z, [(1 - row.confidence) * 50, 100 - (1 - row.confidence) * 50])
            interval_error = max(interval_error, max(abs(limits - np.asarray([row.lower, row.upper]))))
    if max(discrepancies.values()) > 5e-8 or interval_error > 1e-12:
        raise ValueError(f"Independent effect/CI verification failed: {discrepancies}, {interval_error}")
    write_json(out, dict(
        score_receipt=sha256(scored / "receipt.json"), pairs=actual_arms,
        independently_recalculated_estimates=3 * len(FORMAL),
        tested_intervals=len(effects), effect_max_errors=discrepancies, percentile_max_error=interval_error,
        bootstrap_groups=len(groups), bootstrap_replicates=len(draw_matrix),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(config(args.config), args.scored, args.out)


if __name__ == "__main__":
    main()
