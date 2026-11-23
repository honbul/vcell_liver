"""Offline CPU inference from an exact fixed K-anchor TSV."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from vc_build.io import sha256

from .models import Fit, KS


def bundle(path):
    selection = json.loads((path / "freeze.json").read_text())
    if sha256(path / "model.npz") != selection["artifacts"]["model.npz"]:
        raise ValueError("Model hash differs from the frozen manifest")
    arrays = dict(np.load(path / "model.npz", allow_pickle=False))
    return selection, Fit(**arrays)


def calibration(path):
    selection, fit = bundle(path)
    data = json.loads((path / "intervals.json").read_text())
    if (data["model_sha256"] != selection["artifacts"]["model.npz"]
            or data["freeze_sha256"] != sha256(path / "freeze.json")):
        raise ValueError("Uncertainty calibration does not match the frozen model")
    return selection, fit, data


def read_anchors(path, identifiers, bounds):
    data = pd.read_csv(path, sep="\t", dtype={"compound_id": str})
    if list(data.columns) != ["compound_id", "response"] or data.compound_id.duplicated().any():
        raise ValueError("Expected one row per anchor, with compound_id and response columns")
    if set(data.compound_id) != set(identifiers):
        raise ValueError("Input must contain exactly the fixed K anchor compound IDs")
    if not identifiers:
        return np.empty((1, 0))
    obs = pd.to_numeric(data.set_index("compound_id").loc[identifiers, "response"], errors="raise").to_numpy()
    if not np.isfinite(obs).all():
        raise ValueError("Missing or nonfinite anchor measurement")
    lo, hi = np.array(bounds, dtype=float).T
    if ((obs < lo) | (obs > hi)).any():
        raise ValueError("Anchor response lies outside the registered input range")
    return obs[None, :]


def infer_from_bundle(loaded, anchors, k):
    meta, fit, info = loaded
    obs = read_anchors(anchors, meta["anchors"][:k], info["bounds"][:k])
    pred = fit.predict(obs, k, meta["primary"])[0]
    q = np.setdiff1d(np.arange(len(pred)), fit.anchors)
    spread = np.array(info["half_width"][str(k)])
    return (pd.DataFrame({"compound_id": meta["query_panel"], "score": pred[q],
                          "interval_lower": pred[q] - spread, "interval_upper": pred[q] + spread})
            .sort_values(["score", "compound_id"], ascending=[False, True], kind="stable")
            .assign(rank=np.arange(1, len(q) + 1)))


def infer(path, anchors, k):
    return infer_from_bundle(calibration(path), anchors, k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--k", choices=KS, type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        table = infer(args.model, args.anchors, args.k)
    except (ValueError, FileNotFoundError) as e:
        parser.exit(2, f"{e}\n")
    destination = args.out or sys.stdout
    table.to_csv(destination, sep="\t", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
