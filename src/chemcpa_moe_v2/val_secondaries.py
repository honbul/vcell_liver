"""Post-hoc read-out of the pre-registered secondaries on the VALIDATION predictions a run saved.

Nothing is inferred, selected or refitted: every run already wrote `val_pred.npy`,
`val_truth.npy` and `val_line.npy` for its best state, and the baselines wrote
`pred_*_val.npy`. This script lines the two up on the same condition rows and scores both with
the same functions -- the baselines' own `drug_centred_correlation`, top-20 sign agreement, and
SNU-423 -- so the arms and the baselines are compared on equal terms before the test split is
opened. It is a diagnostic, labelled as such, and it never replaces the primary metric.

Row order. The evaluator walks validation drug columns (group_index order) and, inside each, cell
lines in line_index order, keeping pairs with a valid truth. The baselines walk lines then drugs
over the same valid pairs. Both orders are rebuilt from valid.npy and splits.tsv here, and the
rebuilt evaluator line sequence must equal the run's saved `val_line.npy` exactly, or the script
stops.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.moe_drug_v1.evaluate import drug_centred_correlation

SNU423 = "CVCL_0366"


def macro(P, T, L):
    r = np.sqrt(((P - T) ** 2).mean(1))
    return float(np.mean([r[L == l].mean() for l in np.unique(L)]))


def top20_direction(P, T):
    out = []
    for p, t in zip(P, T):
        k = np.argsort(-np.abs(t))[:20]
        out.append(float((np.sign(p[k]) == np.sign(t[k])).mean()))
    return float(np.mean(out))


def score(P, T, L, snu):
    m = L == snu
    return dict(
        macro_delta_rmse=round(macro(P, T, L), 6),
        drug_centred_r=round(float(drug_centred_correlation(P, T, L)[0]), 6),
        top20_direction=round(top20_direction(P, T), 6),
        rms_predicted_response=round(float(np.sqrt((P ** 2).mean())), 6),
        rms_true_response=round(float(np.sqrt((T ** 2).mean())), 6),
        snu423_delta_rmse=round(float(np.sqrt(((P[m] - T[m]) ** 2).mean(1)).mean()), 6),
        snu423_n=int(m.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--baselines-dir", required=True)
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    valid = np.load(Path(a.v2_dir) / "valid.npy")
    splits = pd.read_csv(Path(a.manifest_dir) / "splits.tsv", sep="\t").sort_values("group_index")
    lines = pd.read_csv(Path(a.manifest_dir) / "lines.tsv", sep="\t").sort_values("line_index")
    hit = np.where(lines.cell_line_id.astype(str).to_numpy() == SNU423)[0]
    if len(hit) != 1:
        raise SystemExit(f"{SNU423} (SNU-423) not found exactly once in lines.tsv")
    snu = int(hit[0])
    cols = np.where(splits.split.to_numpy() == "val")[0]

    # baseline order: lines, then drugs
    li, dj = np.where(valid[:, cols])
    base_pos = {(int(l), int(cols[j])): k for k, (l, j) in enumerate(zip(li, dj))}
    # evaluator order: drugs, then lines
    ev_pairs = [(int(l), int(d)) for d in cols for l in range(valid.shape[0]) if valid[l, d]]

    sel = json.loads(Path(a.selection).read_text())
    rep: dict = {"arms": {}, "baselines_same_rows": {}, "note":
                 "post-hoc validation read-out of pre-registered secondaries; not a selection "
                 "input; rows and functions identical for arms and baselines"}
    T_ref = L_ref = take = None
    for arm, s in sel["aggregates"].items():
        lrs = s["lr_scale"]
        per = {}
        for sd in a.seeds:
            run = Path(a.runs_dir) / f"{arm}_lr{lrs}_s{sd}"
            if not (run / "val_pred.npy").exists():
                raise SystemExit(f"{run}/val_pred.npy is missing")
            P = np.load(run / "val_pred.npy").astype(np.float64)
            T = np.load(run / "val_truth.npy").astype(np.float64)
            L = np.load(run / "val_line.npy").astype(int)
            if len(L) != len(ev_pairs) or not np.array_equal(L, [p[0] for p in ev_pairs]):
                raise SystemExit(f"{run}: saved rows do not match the rebuilt evaluator order "
                                 f"({len(L)} vs {len(ev_pairs)})")
            if T_ref is None:
                T_ref, L_ref = T, L
                take = np.array([base_pos[p] for p in ev_pairs])
            elif not np.allclose(T, T_ref):
                raise SystemExit(f"{run}: truth differs from the first run's")
            per[str(sd)] = score(P, T, L, snu)
        keys = list(next(iter(per.values())).keys())
        rep["arms"][arm] = dict(lr_scale=lrs, per_seed=per, seed_mean={
            k: round(float(np.mean([v[k] for v in per.values()])), 6) for k in keys})

    for name in ["no_change", "line_mean", "ridge"]:
        Pb = np.load(Path(a.baselines_dir) / f"pred_{name}_val.npy").astype(np.float64)[take]
        rep["baselines_same_rows"][name] = score(Pb, T_ref, L_ref, snu)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
