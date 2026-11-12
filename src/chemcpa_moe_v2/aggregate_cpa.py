"""Collect the chemCPA arm runs, pick one setting per arm on validation, and score the gate.

Selection reads validation and nothing else. The two seed aggregates are written under separate
names for the same reason as in `initfix_report.py`: the seed mean of the scores is the
pre-registered primary, and the score of the seed-averaged prediction is a three-model ensemble
that answers a different question.

The promotion gate for P3 is the one stated in the plan: P2 must beat both P0 (the published
chemCPA) and P1 (the size-matched Dense residual) by at least 2 % relative on validation, all three
seeds must agree in direction, and P2 must beat the minimum baselines. Anything short of that and
P3 is not trained, and the routing question is reported as not tested rather than answered.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ["P0", "P1", "P2", "P3"]


def load(runs_dir: Path) -> dict:
    out = {}
    for f in sorted(runs_dir.glob("*/record.json")):
        r = json.loads(f.read_text())
        out[(r["arm"], float(r["lr_scale"]), int(r["seed"]))] = (r, f.parent)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--gate-pct", type=float, default=2.0)
    a = ap.parse_args()

    runs = load(Path(a.runs_dir))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for (arm, lrs, seed), (r, _) in sorted(runs.items()):
        rows.append(dict(arm=arm, lr_scale=lrs, seed=seed,
                         val_macro_delta_rmse=round(r["val_macro_delta_rmse"], 6),
                         best_epoch=r["best_epoch"], epochs_run=r["epochs_run"],
                         params_total=r["params"]["total"],
                         params_residual=r["params"]["residual"],
                         params_router=r["params"]["router"],
                         dense_hidden=r.get("dense_hidden"),
                         train_seconds=r["train_seconds"],
                         peak_alloc_MB=round(r.get("peak_alloc_MB", 0), 1),
                         n_val_conditions=r["n_val_conditions"],
                         budget_limited=r["budget_limited"],
                         base_init_sha256=r["shared_init_sha256"][:16]))
    trials = pd.DataFrame(rows)
    trials.to_csv(out / "trials.tsv", sep="\t", index=False)

    # ---- one setting per arm, on seed-42 validation only ------------------------------------
    selected = {}
    for arm in ARMS:
        cand = trials[(trials.arm == arm) & (trials.seed == 42)]
        if cand.empty:
            continue
        best = cand.sort_values("val_macro_delta_rmse").iloc[0]
        selected[arm] = dict(lr_scale=float(best.lr_scale),
                             val_seed42=float(best.val_macro_delta_rmse))

    agg = {}
    for arm, sel in selected.items():
        per_seed, preds = {}, []
        for s in a.seeds:
            key = (arm, sel["lr_scale"], s)
            if key not in runs:
                continue
            r, d = runs[key]
            per_seed[str(s)] = round(r["val_macro_delta_rmse"], 6)
            p = d / "val_pred.npy"
            if p.exists():
                preds.append(np.load(p))
        entry = dict(lr_scale=sel["lr_scale"], per_seed_score=per_seed,
                     n_seeds=len(per_seed))
        if per_seed:
            entry["seed_mean_of_scores"] = round(st.mean(per_seed.values()), 6)
            if len(per_seed) > 1:
                entry["seed_sd_of_scores"] = round(st.stdev(per_seed.values()), 6)
        if len(preds) == len(a.seeds) and preds:
            d0 = runs[(arm, sel["lr_scale"], a.seeds[0])][1]
            T = np.load(d0 / "val_truth.npy")
            L = np.load(d0 / "val_line.npy")
            from .train_cpa import macro_delta_rmse
            entry["score_of_mean_prediction"] = round(
                macro_delta_rmse(np.mean(preds, 0), T, L), 6)
        agg[arm] = entry

    # `baselines_val.json` is one block per baseline; the gate compares the primary metric, so
    # flatten to {name: macro_delta_rmse} and fail loudly rather than silently comparing nothing.
    raw = json.loads(Path(a.baselines).read_text())
    base = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "macro_delta_rmse" in v:
            base[k] = float(v["macro_delta_rmse"])
        elif isinstance(v, (int, float)):
            base[k] = float(v)
    if not base:
        raise SystemExit(f"no baseline macro_delta_rmse found in {a.baselines}")

    def rel(a_, b_):
        return round((a_ - b_) / a_ * 100, 4) if a_ else None

    gate = {}
    if "P2" in agg and "seed_mean_of_scores" in agg["P2"]:
        p2 = agg["P2"]["seed_mean_of_scores"]
        for other in ["P0", "P1"]:
            if other in agg and "seed_mean_of_scores" in agg[other]:
                gate[f"P2_over_{other}_pct"] = rel(agg[other]["seed_mean_of_scores"], p2)
        signs = []
        for s in a.seeds:
            if (str(s) in agg["P2"]["per_seed_score"]
                    and "P1" in agg and str(s) in agg["P1"]["per_seed_score"]):
                signs.append(agg["P1"]["per_seed_score"][str(s)]
                             > agg["P2"]["per_seed_score"][str(s)])
        gate["seeds_agreeing_P2_beats_P1"] = int(sum(signs))
        gate["n_seeds"] = len(signs)
        beats = {k: bool(p2 < v) for k, v in base.items()
                 if isinstance(v, (int, float))}
        gate["beats_baselines"] = beats
        gate["threshold_pct"] = a.gate_pct
        gate["train_P3"] = bool(
            gate.get("P2_over_P0_pct", -1) >= a.gate_pct
            and gate.get("P2_over_P1_pct", -1) >= a.gate_pct
            and gate.get("seeds_agreeing_P2_beats_P1", 0) == len(a.seeds)
            and all(beats.values()))
    else:
        gate["train_P3"] = False
        gate["reason"] = "P2 has no seed-confirmed validation score yet"

    rep = dict(selected=selected, aggregates=agg, baselines=base, gate=gate,
               note="positive percentages mean P2 is better; the two seed aggregates are "
                    "different statistics, not two estimates of one")
    (out / "selection.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
