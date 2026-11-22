"""Offline hepatotoxicity concern scoring for one or more SMILES."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from liver_dili.chemistry import identity

from .features import Features, FM_NAMES
from .release import release_model

STUDY = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong")
DEFAULT_PACKAGE = STUDY / "top3_tuning_v2/b/runs/release-01/models/release-0"
DEFAULT_FM = STUDY / "nextgen_5track/fm_store_v1/chemistry-v1"
DEFAULT_CB = STUDY / "liver_dili_v1/embeddings-full-v1/chemberta.parquet"


def run(args):
    model, card = release_model(args.model)
    bank = Features(args.fm, args.chemberta)
    if bank.hashes != card["embedding_sha256"]:
        raise ValueError("Offline representation SHA-256 differs from the frozen release")
    inputs = list(args.smiles)
    if args.smiles_file:
        inputs.extend(s for s in args.smiles_file.read_text().splitlines() if s.strip())
    if not inputs:
        raise ValueError("Provide at least one SMILES")
    columns = ("smiles", "inchikey", "connectivity", "scaffold")
    registries = pd.DataFrame([identity(s) for s in inputs], columns=columns)
    result = model.forecast(registries, bank)
    features = bank.arrays(registries)
    confidence = float(card["ood_cutoff"])
    output = []
    for i, row in result.iterrows():
        support = float(row.nearest_tanimoto)
        output.append({
            "input_smiles": inputs[i], "standardized_parent": registries.loc[i, "smiles"],
            "inchikey": registries.loc[i, "inchikey"], "version": "DILI v2",
            "endpoint": "probability_of_FDA_DILIrank_binary_concern",
            "probability": float(row.probability), "binary_decision": bool(row.decision),
            "threshold": float(model.calibrator.threshold),
            "nearest_tanimoto": support, "ood_below_training_support": support < confidence,
            "ood_threshold": confidence,
            "available_frozen_embedding_blocks": [f for f in FM_NAMES if np.isfinite(features[f][i]).all()],
        })
    document = json.dumps(output[0] if len(output)==1 else output, indent=2, allow_nan=False)
    if args.out:
        with args.out.open("x") as stream:
            stream.write(document + "\n")
    else:
        print(document)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("smiles", nargs="*")
    cli.add_argument("--smiles-file", type=Path)
    cli.add_argument("--model", type=Path, default=DEFAULT_PACKAGE)
    cli.add_argument("--fm", type=Path, default=DEFAULT_FM)
    cli.add_argument("--chemberta", type=Path, default=DEFAULT_CB)
    cli.add_argument("--out", type=Path)
    run(cli.parse_args())
