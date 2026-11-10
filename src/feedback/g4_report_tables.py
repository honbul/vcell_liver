#!/usr/bin/env python
"""G4: compact headline tables for G4_LOCO.md from the G4 outputs (no new computation).

state_headline.tsv      per fold, T3 (State ∩ PIE unseen_ctx ∩ F3 A targets), target-excluded and -included
                        metrics for State_final / State_best / PIE_official / F3 calibrated / no-change.
state_headline_boot.tsv paired bootstrap rows for State_final vs F3 calibrated / PIE / State_best / no-change.
state_effect_strata.tsv State vs baselines by true-effect tertile (T3, target excluded).
state_4fold_mean.tsv    mean ± SD over the 4 folds (fold = unit) of the headline metrics.
pie_seed_headline.tsv   (if seed scores exist) PIE metric spread over seeds vs official and best baseline.
"""
import argparse
from pathlib import Path

import pandas as pd

KEEP = ["State_final", "State_best", "State_final_pool43", "State_authors_best", "PIE_official",
        "PIEsplit_same_target_cal", "F3_calibrated_same_target", "F3_same_target_mean", "F3_ridge_pca", "F3_no_change"]
KEYS = ["rmse", "pearson", "pearson_pertcentered", "dir_top20", "deg_auprc", "retrieval_rank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4", default="reports/feedback/G4")
    ap.add_argument("--state-pred", default="assets/g4/state/pred")
    a = ap.parse_args()
    g4 = Path(a.g4)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    s = pd.read_csv(g4 / "state_f3metrics_summary.tsv", sep="\t")
    h = s[(s.target_set == "T3_state_pie_f3") & s.predictor.isin(KEEP)]
    h.to_csv(g4 / "state_headline.tsv", sep="\t", index=False, float_format="%.4f")
    for v in ("excl", "incl"):
        print(f"== T3 {v}")
        print(h[h.variant == v].drop(columns=["target_set", "variant"]).round(4).to_string(index=False))
    m = h.groupby(["variant", "predictor"])[KEYS].agg(["mean", "std"])
    m.columns = [f"{k}_{f}" for k, f in m.columns]
    m.reset_index().to_csv(g4 / "state_4fold_mean.tsv", sep="\t", index=False, float_format="%.4f")
    print("== 4-fold mean (excl)")
    print(m.loc["excl"].round(4).to_string())
    t2 = s[(s.target_set == "T2_state_f3") & (s.variant == "excl") &
           s.predictor.isin(["State_final", "State_best", "F3_calibrated_same_target", "F3_no_change"])]
    print("== T2 (all State targets) excl")
    print(t2.drop(columns=["target_set", "variant"]).round(4).to_string(index=False))
    b = pd.read_csv(g4 / "state_f3metrics_paired_bootstrap.tsv", sep="\t")
    hb = b[(b.target_set == "T3_state_pie_f3")]
    hb.to_csv(g4 / "state_headline_boot.tsv", sep="\t", index=False, float_format="%.4f")
    print("== paired bootstrap T3 excl")
    print(hb[hb.variant == "excl"].drop(columns=["target_set", "variant"]).round(4).to_string(index=False))
    # effect-size tertiles (true ||delta|| proxy = no-change RMSE), T3, target excluded
    pt = Path(a.state_pred) / "state_f3metrics_per_target_T3.tsv.gz"
    if pt.is_file():
        d = pd.read_csv(pt, sep="\t")
        nc = d[d.predictor == "F3_no_change"][["fold", "target", "rmse"]].rename(columns={"rmse": "eff"})
        d = d.merge(nc, on=["fold", "target"])
        d["effect_tertile"] = d.groupby(["fold", "predictor"])["eff"].transform(
            lambda x: pd.qcut(x, 3, labels=["low", "mid", "high"]).astype(str))
        g = (d[d.predictor.isin(["State_final", "State_best", "F3_calibrated_same_target", "PIE_official", "F3_no_change"])]
             .groupby(["fold", "effect_tertile", "predictor"])[["rmse", "pearson_pertcentered", "dir_top20", "deg_auprc"]]
             .mean().reset_index())
        g.to_csv(g4 / "state_effect_strata.tsv", sep="\t", index=False, float_format="%.4f")
        print("== effect strata (T3 excl)")
        print(g.round(4).to_string(index=False))
    f = g4 / "pie_seed_spread_hepg2.tsv"
    if f.is_file():
        sp = pd.read_csv(f, sep="\t")
        print("== PIE seed spread")
        print(sp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
