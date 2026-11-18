"""Observation budget, inference only (plan branch F3, the part that needs no training).

The confirmation models (trained with N = 64 control cells per set and p in {1, 0.5, 0.25}) are
evaluated on the source held folds with fewer cells and/or fewer molecules:
    N in {64, 32, 16, 8}   the first N of the 64 fixed eval_input cells (nested subsets)
    p in {1, 0.5, 0.25}    the stored thinning draws of those same cells
The product N * p is a rough "cell-equivalent" budget. Models were NOT trained for N < 64, so a loss
at small N mixes less information with a train/inference mismatch; this is stated with every table.
Nothing is selected from these numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.depth_robust_r6.ridge import eval_feats

from . import common as C
from . import metrics as M
from .oof_tables import mean_baselines, oof_conditions
from .scoring import DEF_TOTAL, DEF_WITHIN, Scorer
from .ridge_fixed import predict as r_predict
from .train_fixed import build_model, predict

NS = (64, 32, 16, 8)
PS = (1.0, 0.5, 0.25)


def ridge_predict(d, run: Path, held, n_cells: int, dev):
    z = np.load(run / "model.npz")
    W, b = torch.as_tensor(z["W"]).double(), torch.as_tensor(z["b"]).double()
    L, Dg = d.conditions(held)
    pid = C.pool_of(d, L, Dg)
    ev = eval_feats(d, take=n_cells)
    return L, Dg, {k: r_predict(W, b, (ev[k][pid] - d.x_mean) / d.x_sd, d.fp[Dg]) for k in C.VIEWS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/r6_followup_v1/observation_budget.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    dev = "cuda"
    d = C.load()
    L, Dg, fold = oof_conditions(d)
    pool = C.pool_of(d, L, Dg)
    sc = Scorer(d.Y[L, Dg], L, Dg, pool, fold, n_boot=a.n_boot, seed=0)
    kappa = json.loads(Path("manifests/r6_followup_v1/kappa_selection.json").read_text())["chosen_kappa"]
    sc.add("poolmean", [mean_baselines(d, L, Dg, fold, pool, kappa)["poolmean"]], retrieval=False)
    G = d.Y.shape[2]
    for arm, item in spec["arms"].items():
        models = {}
        for k in C.FOLDS:
            for s in C.SEEDS:
                run = C.assets() / "runs" / item["stage"] / item["pattern"].format(fold=k, seed=s)
                if arm == "R":
                    models[(k, s)] = run
                    continue
                ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
                assert ck["fold"] == k and not ck["inner"] and not (set(ck["fit_drugs"]) & set(C.source_split(d, k)[1]))
                m = build_model(d, ck["arm"], ck["hp"]).to(dev)
                m.load_state_dict(ck["state_dict"])
                models[(k, s)] = (m, ck["y_scale"])
        for N in NS:
            preds = {p: [] for p in PS}
            for s in C.SEEDS:
                per = {}
                for k in C.FOLDS:
                    held = C.source_split(d, k)[1]
                    if arm == "R":
                        Lk, Dk, pe = ridge_predict(d, models[(k, s)], held, N, dev)
                    else:
                        Lk, Dk, pe = predict(models[(k, s)][0], d, held, dev, models[(k, s)][1], n_cells=N)
                    assert (Lk == L[fold == k]).all() and (Dk == Dg[fold == k]).all()
                    per[k] = pe
                for (p, ts) in C.VIEWS:
                    A = np.zeros((len(L), G), np.float32)
                    for k in C.FOLDS:
                        A[fold == k] = per[k][(p, ts)]
                    preds[p].append(A)
            for p in PS:
                sc.add(f"{arm}|{N}|{p}", preds[p], retrieval=False)
            print("scored", arm, N, flush=True)
        del models
        torch.cuda.empty_cache()
    rows = []
    std = C.std_fields(split_role="source_oof_inference_only", fold="0,1,2", model_seed="42,43,44")
    arms = list(spec["arms"])
    for arm in arms:
        for N in NS:
            for p in PS:
                s = sc.summary(f"{arm}|{N}|{p}")
                pr = sc.pair(f"{arm}|{N}|{p}", "poolmean")
                rows.append(dict(std, row_type="arm", arm=arm, comparator="", n_cells=N, p=p, budget_cell_equivalents=N * p, n_drugs=s["n_drugs"], n_units=s["n_units"],
                                 total_rmse=s["total_rmse"], total_lo=s["total_lo"], total_hi=s["total_hi"], within_rmse=s["within_rmse"],
                                 within_lo=s["within_lo"], within_hi=s["within_hi"], total_minus_poolmean=pr["total_diff"],
                                 total_minus_poolmean_lo=pr["total_lo"], total_minus_poolmean_hi=pr["total_hi"]))

    def pair_row(kind, ca, cb, **kw):
        pr = sc.pair(ca, cb)
        rows.append(dict(std, row_type=kind, sign="candidate - comparator (negative = candidate better)", n_drugs=pr["n_drugs_total"], n_units=pr["n_units_total"],
                         total_diff=pr["total_diff"], total_diff_lo=pr["total_lo"], total_diff_hi=pr["total_hi"], total_rel=pr["total_rel"], total_category=pr["total_category"],
                         within_diff=pr["within_diff"], within_diff_lo=pr["within_lo"], within_diff_hi=pr["within_hi"], within_rel=pr["within_rel"],
                         within_category=pr["within_category"], **kw))

    for c, b in spec["pairs"]:
        for N in NS:
            for p in PS:
                pair_row("pair_between_arms", f"{c}|{N}|{p}", f"{b}|{N}|{p}", arm=c, comparator=b, n_cells=N, p=p, budget_cell_equivalents=N * p)
    for arm in arms:
        # same budget, two ways to spend it: many shallow cells (candidate) vs few deep cells (comparator)
        for (N1, p1), (N2, p2) in (((64, 0.25), (16, 1.0)), ((64, 0.5), (32, 1.0)), ((32, 0.5), (16, 1.0)), ((32, 0.25), (8, 1.0)), ((16, 0.5), (8, 1.0))):
            pair_row("pair_iso_budget", f"{arm}|{N1}|{p1}", f"{arm}|{N2}|{p2}", arm=arm, comparator=arm, n_cells=N1, p=p1, budget_cell_equivalents=N1 * p1,
                     candidate_setting=f"N={N1},p={p1}", comparator_setting=f"N={N2},p={p2}")
        # the 2x2: effect of cells at fixed depth, effect of depth at fixed cells
        for (N1, p1), (N2, p2) in (((16, 1.0), (64, 1.0)), ((16, 0.25), (64, 0.25)), ((64, 0.25), (64, 1.0)), ((16, 0.25), (16, 1.0)), ((8, 1.0), (64, 1.0)), ((8, 0.25), (64, 1.0))):
            pair_row("pair_within_arm", f"{arm}|{N1}|{p1}", f"{arm}|{N2}|{p2}", arm=arm, comparator=arm, n_cells=N1, p=p1, budget_cell_equivalents=N1 * p1,
                     candidate_setting=f"N={N1},p={p1}", comparator_setting=f"N={N2},p={p2}")
    out = pd.DataFrame(rows)
    out["metric_definition"] = f"total: {DEF_TOTAL} | within: {DEF_WITHIN}"
    out["note"] = "inference only: models were trained with N = 64; N < 64 uses the first N fixed eval_input cells"
    out.to_csv(spec["out"], sep="\t", index=False, float_format="%.6f")
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    print(out[out.row_type == "arm"][["arm", "n_cells", "p", "budget_cell_equivalents", "total_rmse", "total_lo", "total_hi", "within_rmse", "total_minus_poolmean"]].to_string())
    print(out[out.row_type != "arm"][["row_type", "arm", "comparator", "n_cells", "p", "candidate_setting", "comparator_setting", "total_diff", "total_diff_lo", "total_diff_hi",
                                      "total_rel", "total_category", "within_diff", "within_rel"]].to_string())


if __name__ == "__main__":
    main()
