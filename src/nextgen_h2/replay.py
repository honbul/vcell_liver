"""Replay a trusted, hash-pinned H2 prediction package without network I/O."""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vc_build.io import publish_json, sha256

from .assets import COLD, LIMORE, parquet_lock
from .run import META, offline_forecast, read_json


def network_deny(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("H2 replay is offline")


def run(root, out):
    sys.addaudithook(network_deny)
    torch.set_num_threads(5)
    source = read_json(COLD / "receipt.json")["files"]["rna.parquet"]
    rna = parquet_lock(COLD / "rna.parquet", source)
    inputs = read_json(LIMORE / "receipt.json")["artifacts"]
    held = parquet_lock(LIMORE / "test_inputs/rna.parquet", inputs["test_inputs/rna.parquet"]["sha256"])
    rna = pd.concat([rna, held.reindex(columns=rna.columns)], verify_integrity=True)
    checks = []
    for part in range(4):
        path = root / "forecasts" / str(part)
        proof = read_json(path / "receipt.json")
        table = parquet_lock(path / "predictions.parquet", proof["prediction_sha256"])
        package = path / "predictor.pkl"
        if sha256(package) != proof["predictor_sha256"]:
            raise ValueError("Prediction package checksum differs")
        with package.open("rb") as stream:
            loaded = pickle.load(stream)
        predicted = offline_forecast(loaded, table[META], rna)
        mismatch = max(float(np.abs(table[name].to_numpy() - value).max())
                       for name, value in predicted.items())
        missing = ~table.champion_observed
        fallback = np.abs(predicted["champion"][missing] - predicted["C1"][missing])
        fully_masked = np.abs(predicted["structure_only"] - predicted["C1"])
        checks.append({"partition": part, "rows": len(table), "missing_profile_rows": int(missing.sum()),
                       "max_abs_diff": mismatch, "missing_fallback_max_abs_diff": float(fallback.max(initial=0)),
                       "structure_only_fallback_max_abs_diff": float(fully_masked.max(initial=0))})
        if mismatch >= 1e-6 or fallback.max(initial=0) != 0 or fully_masked.max(initial=0) != 0:
            raise AssertionError(f"Offline replay failed on partition {part}")
    publish_json(out, {"partitions": checks, "network_connections_allowed": False, "tolerance": 1e-6})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.root, args.out)
