"""Descriptive statistics for the data card. Read-outs only -- nothing here can change the split.

The nearest-train Tanimoto similarity is reported because this is a drug-identity holdout, not a
scaffold holdout: structurally close analogues may well sit in train. No drug is removed on the
basis of this number (plan s5.2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def tanimoto_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = A.astype(np.float32)
    b = B.astype(np.float32)
    inter = a @ b.T
    na = a.sum(1)[:, None]
    nb = b.sum(1)[None, :]
    return inter / np.maximum(na + nb - inter, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D, M = Path(args.data_dir), Path(args.manifest_dir)
    fp = np.load(D / "fp.npy")
    Y = np.load(D / "Y.npy")
    valid = np.load(D / "valid.npy")
    plate = np.load(D / "plate.npy")
    sp = pd.read_csv(M / "splits.tsv", sep="\t")
    cov = pd.read_csv(M / "coverage.tsv", sep="\t")
    keep = cov.keep.to_numpy()

    tr = (sp.split == "train").to_numpy()
    out = dict(nearest_train_tanimoto={})
    for s in ["val", "test"]:
        m = (sp.split == s).to_numpy()
        T = tanimoto_matrix(fp[m], fp[tr])
        nn = T.max(1)
        out["nearest_train_tanimoto"][s] = dict(
            n=int(m.sum()), mean=float(nn.mean()), median=float(np.median(nn)),
            p90=float(np.percentile(nn, 90)), max=float(nn.max()),
            frac_above_0p7=float((nn >= 0.7).mean()),
            frac_above_0p5=float((nn >= 0.5).mean()))

    # effect size of the truth itself: how large is the response we are asking a model to predict
    V = valid[keep]
    Yk = Y[keep]
    mag = np.sqrt((Yk[V] ** 2).mean(1))
    out["response_magnitude_rmse_of_truth"] = dict(
        mean=float(mag.mean()), median=float(np.median(mag)),
        p10=float(np.percentile(mag, 10)), p90=float(np.percentile(mag, 90)))

    # how many (line, plate) blocks the fixed plate diagnostic will have on each split
    blocks = {}
    for s in ["val", "test"]:
        cols = np.where((sp.split == s).to_numpy())[0]
        n = 0
        for li in range(V.shape[0]):
            pl = plate[keep][li][cols]
            pl = pl[V[li][cols]]
            vals, cnt = np.unique(pl[pl >= 0], return_counts=True)
            n += int((cnt >= 5).sum())
        blocks[s] = n
    out["plate_diagnostic_blocks"] = blocks

    # replicate-well agreement is the noise floor the truth itself carries, per the earlier study
    out["split_sizes"] = sp.split.value_counts().to_dict()
    out["lines_kept"] = int(keep.sum())
    out["lines_dropped"] = cov.loc[~keep, ["cell_line_id", "cell_name"]].to_dict("records")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
