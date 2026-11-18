"""Learning-curve and effect-range summaries from finished runs (no new model, no held-fold delta is opened here).

learning_curves.tsv   inner-validation source delta RMSE by update step, per arm, mean over the cells
                      (relative to the frozen backbone of the same cell), with best / last step summaries
effect_range.tsv      for each pre-specified contrast and K group: point, interval, and the half-width of the interval
                      (the smallest difference this design could have separated from zero), from the committed table
"""
import argparse, json, os
import numpy as np
import pandas as pd

from .common import fruns, pman, tag_cols, NULL_SEEDS
from .score_oof import run_dir
from .stats import THR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--keys", nargs="+", default=["M", "G0", "GR"] + [f"GS{n}" for n in NULL_SEEDS])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = (lambda f: pman(a.tag, f)) if a.tag else pman
    rows, summ = [], []
    for k in a.folds:
        for s in a.seeds:
            c = json.load(open(os.path.join(run_dir("C", f"fold{k}", s)[0], "record.json")))["best"]["score"]
            for key in a.keys:
                d, rid = run_dir(key, f"fold{k}", s)
                h = pd.read_csv(os.path.join(d, "history.tsv"), sep="\t")
                arm = "TDS" if key.startswith("TDS") else ("GS" if key.startswith("GS") else key)
                for r in h.itertuples():
                    rows.append(dict(arm=arm, key=key, fold=k, seed=s, step=r.step, sel_minus_backbone=r.sel_rmse - c, corr_rms=r.corr_rms, mean_gate=r.mean_gate))
                summ.append(dict(arm=arm, key=key, fold=k, seed=s, best_step=int(h.step[h.sel_rmse.idxmin()]), best_minus_backbone=h.sel_rmse.min() - c,
                                 last_minus_backbone=h.sel_rmse.iloc[-1] - c, last_minus_best=h.sel_rmse.iloc[-1] - h.sel_rmse.min()))
    L = pd.DataFrame(rows)
    g = L.groupby(["arm", "step"]).agg(sel_minus_backbone_mean=("sel_minus_backbone", "mean"), sel_minus_backbone_min=("sel_minus_backbone", "min"),
                                       sel_minus_backbone_max=("sel_minus_backbone", "max"), corr_rms_mean=("corr_rms", "mean"), mean_gate=("mean_gate", "mean"),
                                       n_runs=("key", "size")).reset_index()
    tag_cols(g, "inner_validation(checkpoint selection units of the source folds)", metric_definition="family-balanced source delta RMSE of the inner-validation units minus the same value of the frozen backbone of the cell; mean / min / max over runs").to_csv(
        out("learning_curves.tsv"), sep="\t", index=False)
    Sm = pd.DataFrame(summ)
    g2 = Sm.groupby("arm").agg(n_runs=("key", "size"), best_step_median=("best_step", "median"), best_step_min=("best_step", "min"), best_step_max=("best_step", "max"),
                               best_minus_backbone_mean=("best_minus_backbone", "mean"), last_minus_backbone_mean=("last_minus_backbone", "mean"),
                               last_minus_best_mean=("last_minus_best", "mean"), n_runs_best_is_last=("last_minus_best", lambda x: int((x == 0).sum()))).reset_index()
    tag_cols(g2, "inner_validation(checkpoint selection units of the source folds)", metric_definition="summary of learning_curves.tsv per arm; best_is_last > 0 would indicate too few updates").to_csv(
        out("learning_curve_summary.tsv"), sep="\t", index=False)
    f = out("capacity_attribution_contrasts.tsv")
    if os.path.exists(f):
        c = pd.read_csv(f, sep="\t"); c = c[c.scope == "pooled"]
        e = c[["k_group", "candidate", "comparator", "core", "diff", "lo95", "hi95", "category_ci95", "n_units", "n_drugs"]].copy()
        e["half_width_95"] = (e.hi95 - e.lo95) / 2
        e["operating_threshold"] = THR
        e["interval_excludes_improvement_of_threshold"] = e.lo95 > -THR
        tag_cols(e, "source_out_of_fold", metric_definition="half_width_95: half of the 95 % paired drug-group bootstrap interval; a true difference smaller than this could not have been separated from zero").to_csv(
            out("effect_range.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(g2.round(5).to_string())


if __name__ == "__main__":
    main()
