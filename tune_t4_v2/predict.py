"""Offline inference from registered 6 h differential expression and compound dose."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256
from src.r19_plan.t4_scaffold_transfer import scaffold

from .pretrain import load_pan


def release_models(root, lock, cell):
    table = read_json(lock)
    for group, hash_value in table["cell_groups"][cell].items():
        folder = root / cell / group
        receipt = folder / "receipt.json"
        if sha256(receipt) != hash_value:
            raise ValueError("Frozen release manifest verification failed")
        for key, record in read_json(receipt)["slices"].items():
            model_path = folder / f"{key}.joblib"
            if sha256(model_path) != record["model_sha256"]:
                raise ValueError("Offline T4 model is not the frozen model")
            yield joblib.load(model_path)


def run(args):
    raw = pd.read_csv(args.input, sep="\t")
    required = {"sample_id", "parent", "dose_uM"}
    if not required <= set(raw) or raw.sample_id.duplicated().any():
        raise ValueError("The input requires unique sample_id, parent SMILES, and dose_uM")
    if not np.isfinite(raw.dose_uM).all() or (raw.dose_uM <= 0).any() or raw.empty:
        raise ValueError("Expected finite, positive dose and nonempty 6 h expression")
    priors = read_json(args.pan / "receipt.json")
    if sha256(args.pan / "pan_model.joblib") != priors["model_sha256"]:
        raise ValueError("The offline pan checkpoint is not the frozen model")
    prior = load_pan(args.pan / "pan_model.joblib")
    keys = [str(gene) for gene in prior.genes]
    x = raw[keys].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Every registered 6 h gene value must be finite")
    identities = raw.parent.map(scaffold)
    total = np.zeros_like(x, dtype=np.float64)
    coverage = np.zeros(len(raw), dtype=int)
    for model in release_models(args.models, args.lock, args.cell):
        optional = ~identities.isin(model.train_scaffolds)
        if optional.any():
            total[optional] += model.predict(raw.loc[optional], x[optional], prior)
            coverage += optional.to_numpy(dtype=int)
    if (coverage == 0).any():
        raise ValueError("No scaffold-disjoint release head for some input structures")
    result = pd.DataFrame(total/coverage[:, None], columns=keys)
    result = pd.concat([raw[["sample_id", "parent", "dose_uM"]].assign(frozen_heads=coverage), result], axis=1)
    result.to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "models", "lock", "pan", "out"):
        cli.add_argument("--" + key, type=Path, required=True)
    cli.add_argument("--cell", choices=("HEPG2", "PHH"), required=True)
    run(cli.parse_args())
