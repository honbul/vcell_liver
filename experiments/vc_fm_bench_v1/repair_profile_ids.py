"""Migrate verified v1 RNA vectors to portable Unicode IDs without unpickling."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vc_fm_bench.artifacts import file_record, new_directory, verify_record, write_json


def migrate(source, inputs, destination):
    receipt = json.loads((source / "receipt.json").read_text())
    path = source / "vectors.npz"
    verify_record(path, receipt["output"])
    for p, provenance in zip(inputs, receipt["input"], strict=True):
        verify_record(p, provenance)
    keys = pd.concat([pd.read_parquet(p, columns=[]) for p in inputs]).index.to_numpy(dtype=str)
    if len(keys) != receipt["rows"] or len(set(keys)) != len(keys):
        raise ValueError("Invalid source IDs")
    used = keys[~np.isin(keys, receipt["dropped_ids"])]
    with np.load(path, allow_pickle=False) as raw:
        vectors = raw["vectors"]
    if vectors.shape != (receipt["embedded"], receipt["dimension"]) or len(used) != len(vectors):
        raise ValueError("Vector receipt incompatible with reconstructed source IDs")
    with new_directory(destination) as tmp:
        file = tmp / "vectors.npz"
        np.savez(file, id=used, vectors=vectors)
        receipt.update({"original_encoding_receipt": file_record(source / "receipt.json"),
                        "output": file_record(file)})
        write_json(tmp / "receipt.json", receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path, nargs="+")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    migrate(a.source, a.input, a.output)


if __name__ == "__main__":
    main()
