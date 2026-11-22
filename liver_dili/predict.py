"""Offline prediction from a small-molecule SMILES and a frozen DILI research release."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from .chemistry import features, identity
from .embeddings import ChemBERTa
from .splits import digest, nearest


DEFAULT = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong/liver_dili_v1/release-v2")


def infer(smiles, model):
    canonical, key, connectivity, _ = identity(smiles)
    freeze = json.loads((model / "receipt.json").read_text())
    for name, expected in freeze["assets"].items():
        candidate = (model / name).resolve()
        if model.resolve() not in candidate.parents or digest(candidate) != expected:
            raise ValueError(f"Invalid release artifact {name}")
    contract = json.loads((model / "contract.json").read_text())
    package = joblib.load(model / "checkpoint.joblib")
    encoder = ChemBERTa(model / "chemberta")
    fp, structured = features(canonical)
    text, truncation = encoder.encode([canonical])
    raw = np.c_[structured[None, :], text]
    pathways = package["path_head"].predict(raw, pd.Index([key]))
    x = np.c_[raw, pathways] if package["mode"] == "predicted_pathways" else raw
    classifier = package["classifier"]
    if package["method"] == "logistic":
        scaler, fitted = classifier
        log_odds = fitted.decision_function(scaler.transform(x))
    else:
        log_odds = classifier.booster_.predict(x, raw_score=True)
    risk = float(package["calibration"].predict_proba(log_odds[:, None])[0, 1])
    parameters = package["coefficients"]
    lo, hi = np.quantile(expit(parameters[:, 0] * log_odds[0] + parameters[:, 1]), [.05, .95]).tolist()
    similarity = float(nearest(package["references"], [fp])[0])
    warns = []
    if similarity < package["ood_threshold"]:
        warns.append("low_nearest_training_tanimoto")
    if truncation[0]:
        warns.append("chemberta_sequence_truncated")
    return dict(canonical_isomeric_smiles=canonical, inchikey=key, connectivity=connectivity,
        predicted_pathway_scores={name: float(value) for name, value in pathways.iloc[0].items()},
        probability_dilirank_concern=risk, calibrated_probability_interval_90=[lo, hi],
        interval_method=contract["calibration"], evaluation_threshold=package["threshold"],
        ood=bool(warns), warnings=warns, nearest_training_tanimoto=similarity,
        ood_similarity_threshold=package["ood_threshold"], version=contract["version"],
        label=contract["label"], clinical_exposure_inputs=False,
        scope=["preclinical structure-only estimate", "input chemistry and 24-hour liver assay training support",
               "FDA concern label probability; clinical incidence and individual patient risk are different endpoints",
               "coverage may change on new chemistry and later regulatory releases"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles")
    parser.add_argument("--model", type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        result = infer(args.smiles, args.model)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
