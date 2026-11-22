"""Run the frozen CLAMP DILI research screen on local SMILES inputs."""
import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from rdkit import rdBase

from vc_fm_bench.artifacts import verify_record
from .data import CLAMP, WORKER, read
from .models import FEATURES, raw_score
from .molecular import arrays, extract, molecule, nearest


def batch_input(path):
    with path.open(newline="") as handle:
        if path.suffix.lower() in (".csv", ".tsv"):
            reader = csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            if not reader.fieldnames or "smiles" not in reader.fieldnames:
                raise ValueError("CSV or TSV must have a smiles column")
            return [row["smiles"].strip() if row["smiles"] else "" for row in reader]
        return [line.strip() for line in handle if line.strip()]


def predict(smiles, model_directory, encoder_directory):
    if not smiles:
        raise ValueError("No SMILES inputs")
    standardized = []
    for line, item in enumerate(smiles, start=1):
        try:
            with rdBase.BlockLogs():
                normalized, key, *_ = molecule(item)
        except ValueError as exc:
            raise ValueError(f"Invalid SMILES at input {line}: {item!r}") from exc
        standardized.append((normalized, key))
    policy = read(model_directory / "freeze.json")
    checkpoint = model_directory / "checkpoint.joblib"
    verify_record(checkpoint, policy["checkpoint"])
    saved = joblib.load(checkpoint)
    model = policy["deployment"]
    if saved["model"] != model or saved["calibrator"].method != policy["methods"][model]["method"]:
        raise ValueError("The installed DILI head or calibrator does not match its release contract")
    parents, keys = zip(*standardized)
    # Always run actual CLAMP molecular extraction, including for structures in the reference roster.
    embeddings = extract(encoder_directory, parents)
    values = arrays(parents, embeddings)
    cal = saved["calibrator"]
    inputs = values[FEATURES[model]]
    raw = raw_score(model, saved["head"], inputs)
    p = cal.probability(raw)
    bounds = cal.interval(raw, saved["draws"])
    similarity = nearest(values["bits"], saved["references"])
    output = []
    for i, value in enumerate(p):
        abstain = bool(similarity[i] < saved["ood_cutoff"])
        output.append({
            "input_smiles": smiles[i],
            "standardized_smiles": parents[i],
            "inchikey": keys[i],
            "model": model,
            "calibrator": cal.method,
            "probability_of_dilirank_concern": float(value),
            "calibration_parameter_interval90": bounds[i].tolist(),
            "nearest_training_tanimoto": float(similarity[i]),
            "decision_threshold": cal.threshold,
            "ood_tanimoto_threshold": saved["ood_cutoff"],
            "abstain": abstain,
            "warning": "LOW_TRAINING_SIMILARITY" if abstain else None,
            "screen_positive": None if abstain else bool(value >= cal.threshold),
        })
    if not np.isfinite(p).all() or not np.isfinite(bounds).all():
        raise ValueError("DILI model produced a nonfinite uncertainty estimate")
    return output


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles")
    source.add_argument("--batch", type=Path)
    cli.add_argument("--model", type=Path, default=WORKER / "task1-01/historical")
    cli.add_argument("--encoder", type=Path, default=CLAMP)
    args = cli.parse_args()
    try:
        values = [args.smiles] if args.smiles is not None else batch_input(args.batch)
        results = predict(values, args.model, args.encoder)
    except (ValueError, OSError) as exc:
        print(f"confirm_dili: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(results[0] if args.smiles is not None else results, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
