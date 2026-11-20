"""Build the small deliverable tables and figures from immutable run outputs.

Only small, per-condition (well-aggregate) tables are written; no cell-level arrays."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from . import aggregate as ag
from . import figures as fg

PAIR_METRICS = ["comp_tv", "within_mean_rmse", "within_ed", "ed", "mean_rmse_latent"]


def load(patterns):
    fs = sorted(f for p in patterns for f in glob.glob(p))
    if not fs:
        return pd.DataFrame(), []
    return ag.ensure_keys(pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True)), fs


def b1_tables(b1, out):
    p = pd.read_csv(f"{b1}/replicate_components_pairs.tsv.gz", sep="\t")
    p["n_sub"] = p.n_sub.astype(str)
    g = p.groupby(["line", "K", "pair_type", "n_sub"])
    s = g[PAIR_METRICS].agg(["median", "mean"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.insert(0, "n_pairs", g.size())
    s = s.reset_index()
    # signal-to-replicate ratios (full n): median metric of each pair type / median replicate metric
    ref = s[(s.pair_type == "replicate_same_condition") & (s.n_sub == "full")].set_index("line")
    for m in ("comp_tv", "within_mean_rmse", "ed"):
        s[f"{m}_median_over_replicate"] = s[f"{m}_median"] / s.line.map(ref[f"{m}_median"])
    s.round(5).to_csv(f"{out}/replicate_components.tsv", sep="\t", index=False)
    p.to_csv(f"{out}/replicate_components_pairs.tsv.gz", sep="\t", index=False, float_format="%.5g")
    dec = pd.read_csv(f"{b1}/decomposition.tsv.gz", sep="\t")
    dec.to_csv(f"{out}/decomposition_train_conditions.tsv.gz", sep="\t", index=False, float_format="%.5g")
    os.makedirs(f"{out}/figures", exist_ok=True)
    fg.fig_decomposition(dec, f"{out}/figures/fig1_mean_change_decomposition.png")
    for f in ("K_selection.tsv", "synthetic_controls.tsv", "atlas_K.json"):
        os.system(f"cp {b1}/{f} {out}/{f}")
    return p, dec


def condition_tables(ev, out):
    ev = ag.alias_d00(ev)
    K = [c for c in ev.columns if c.startswith("pi_pred_")]
    keys = ag.KEYS
    nonor = ev[ev.is_oracle == 0]
    a = nonor.groupby(keys + ["rep_endpoint", "rep_query"], as_index=False).mean(numeric_only=True)
    cond = a.groupby(keys, as_index=False).mean(numeric_only=True)
    for k in range(len(K)):
        cond[f"bias_state{k}"] = cond[f"pi_pred_{k}"] - cond[f"pi_obs_{k}"]
    bias = [c for c in cond.columns if c.startswith("bias_state")]
    cond[keys + ["n_low", "n_high", "n_query", "comp_tv", "comp_js", "comp_brier"] + bias].round(5).to_csv(
        f"{out}/composition_metrics.tsv", sep="\t", index=False)
    nq = [c for c in cond.columns if c.startswith("neff_q_")]
    cond[keys + ["within_mean_rmse", "within_ed", "within_n_states", "within_mass_scored", "ed", "mmd",
                 "dmean_rmse_gene", "decoder_recon_rmse_gene", "mean_rmse_latent", "query_frac_out_of_atlas",
                 "query_mean_max_membership"] + nq].round(5).to_csv(f"{out}/within_state_metrics.tsv", sep="\t",
                                                                     index=False)
    asm = cond[cond.head_comp_tv.notna()] if "head_comp_tv" in cond else cond.iloc[:0]
    asm[keys + ["head_comp_tv", "comp_tv", "assembly_tv_head_vs_reread", "assembly_mean_rmse_head_vs_reread"]].rename(
        columns={"comp_tv": "reread_comp_tv"}).round(5).to_csv(f"{out}/assembly_consistency.tsv", sep="\t", index=False)
    return ev, cond


INFO = {
    "control": ("non-learned", "vehicle cells of the endpoint replicate", "cells"),
    "nearest_low": ("non-learned", "low endpoint well", "cells"),
    "nearest_high": ("non-learned", "high endpoint well", "cells"),
    "mixture": ("non-learned (=D00)", "both endpoint wells, t", "weighted cells"),
    "D00": ("non-learned (=mixture)", "both endpoint wells, t", "weighted cells"),
    "global_moment": ("non-learned", "both endpoints: mean+cov (LW)", "1000 Gaussian samples"),
    "state_moment": ("non-learned", "both endpoints: state pi, shrunk state moments", "1000 samples, Gaussian per state"),
    "ot_displacement": ("non-learned", "both endpoint wells, exact OT coupling", "weighted displaced points"),
    "D10": ("learned pi-head", "endpoint+vehicle summaries", "re-weighted mixture cells"),
    "D01": ("learned Q-head", "endpoint+vehicle summaries", "state-shifted mixture cells"),
    "D11": ("learned pi+Q heads", "endpoint+vehicle summaries", "re-weighted, shifted mixture cells"),
    "W": ("learned whole-distribution", "endpoint+vehicle summaries", "affine-transformed mixture cells"),
    "mixture_t": ("1 train-fitted scalar per episode (C3, post-freeze)", "both endpoint wells, t*=0.325",
                  "weighted cells"),
}


def baselines_table(ev, summ, out, role="development"):
    cost = ev[ev.is_oracle == 0].groupby("method").agg(sec_per_triplet=("sec", "mean"), points=("n_points", "mean"))
    rows = []
    for m in sorted(set(summ.method)):
        fam, inf, form = INFO.get(m, ("", "", ""))
        r = dict(method=m, family=fam, endpoint_information=inf, output=form,
                 sec_per_triplet=round(float(cost.sec_per_triplet.get(m, np.nan)), 4),
                 mean_points=round(float(cost.points.get(m, np.nan)), 1))
        for ep in ("E1000", "E100"):
            for met in ("comp_tv", "within_mean_rmse", "ed", "dmean_rmse_gene"):
                x = summ[(summ.role == role) & (summ.episode == ep) & (summ.direction == "cross") & (summ.method == m)
                         & (summ.metric == met)]
                if len(x):
                    x = x.iloc[0]
                    r[f"{ep}_{met}"] = f"{x['mean']:.4f} [{x.lo:.4f}, {x.hi:.4f}]"
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{out}/BASELINES.tsv", sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", required=True)
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--budget", nargs="*", default=[])
    ap.add_argument("--baseline_role", default="development")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert not os.path.exists(f"{args.out}/summary_by_method.tsv"), "immutable report run exists"
    os.makedirs(f"{args.out}/figures", exist_ok=True)
    b1_tables(args.b1, args.out)
    ev, files = load(args.evals)
    json.dump(files, open(f"{args.out}/eval_inputs.json", "w"), indent=1)
    ev, cond = condition_tables(ev, args.out)
    drug = ag.per_drug(cond)
    methods = sorted(cond.method.unique())
    summ = pd.concat([ag.summary(drug, m, methods) for m in ag.METRICS], ignore_index=True)
    summ.round(5).to_csv(f"{args.out}/summary_by_method.tsv", sep="\t", index=False)
    baselines_table(ev, summ, args.out, args.baseline_role)
    learned = [m for m in ("D10", "D01", "D11", "W") if m in methods]
    base = [m for m in ("control", "nearest_low", "nearest_high", "global_moment", "state_moment", "ot_displacement",
                        "mixture_t") if m in methods]
    pairs = [(m, "D00") for m in learned + base]
    pairs += [(m, ref) for m in learned for ref in ("nearest_low", "state_moment", "mixture_t") if ref in methods]
    pr = pd.concat([ag.paired(drug, m, pairs) for m in ag.METRICS], ignore_index=True)
    pr.round(5).to_csv(f"{args.out}/paired_vs_D00.tsv", sep="\t", index=False)
    ag.per_line_summary(cond, ag.METRICS, methods).round(5).to_csv(f"{args.out}/summary_by_line.tsv", sep="\t",
                                                                    index=False)
    # seed spread of learned arms (development)
    if learned:
        s = ev[ev.method.isin(learned) & (ev.direction == "cross")]
        sd = (s.groupby(["role", "episode", "method", "seed", "line", "drug"], as_index=False)[ag.METRICS].mean()
              .groupby(["role", "episode", "method", "seed", "drug"], as_index=False)[ag.METRICS].mean()
              .groupby(["role", "episode", "method", "seed"], as_index=False)[ag.METRICS].mean())
        sd.round(5).to_csv(f"{args.out}/seed_level_summary.tsv", sep="\t", index=False)
    # oracle replicate limit and source failure map (train role, cross)
    orc = ev[(ev.is_oracle == 1)]
    if len(orc):
        o = orc.groupby(["role", "line", "drug", "episode", "rep_endpoint", "rep_query"], as_index=False)[ag.METRICS].mean()
        mix = ev[(ev.method == "mixture") & (ev.direction == "cross")]
        ctl = ev[(ev.method == "control") & (ev.direction == "cross")][
            ["role", "line", "drug", "episode", "rep_endpoint", "rep_query", "ed"]].rename(columns={"ed": "strength"})
        m = mix.merge(o, on=["role", "line", "drug", "episode", "rep_endpoint", "rep_query"], suffixes=("", "_oracle"))
        m = m.merge(ctl, on=["role", "line", "drug", "episode", "rep_endpoint", "rep_query"])
        for met in ("comp_tv", "within_mean_rmse", "ed"):
            m[met + "_excess"] = m[met] - m[met + "_oracle"]
        fm = (m.groupby(["role", "line", "drug", "episode"], as_index=False)
              [["comp_tv", "comp_tv_oracle", "comp_tv_excess", "within_mean_rmse", "within_mean_rmse_oracle",
                "within_mean_rmse_excess", "ed", "ed_oracle", "ed_excess", "strength"]].mean())
        # pre-fixed classification thresholds: the line's median replicate-pair metric (B1, full n)
        rc = pd.read_csv(f"{args.out}/replicate_components.tsv", sep="\t")
        rc = rc[(rc.pair_type == "replicate_same_condition") & (rc.n_sub == "full")].set_index("line")
        fm["comp_fail"] = fm.comp_tv_excess > fm.line.map(rc.comp_tv_median)
        fm["within_fail"] = fm.within_mean_rmse_excess > fm.line.map(rc.within_mean_rmse_median)
        fm["failure_class"] = np.select([fm.comp_fail & fm.within_fail, fm.comp_fail, fm.within_fail],
                                        ["both", "composition", "within_state"], "within_replicate_limit")
        fm.round(5).to_csv(f"{args.out}/mixture_failure_map.tsv", sep="\t", index=False)
        fm.groupby(["role", "line", "episode"]).failure_class.value_counts().unstack(fill_value=0).to_csv(
            f"{args.out}/mixture_failure_classes.tsv", sep="\t")
        tr = fm[fm.role == "train"]
        if len(tr):
            fg.fig_failure_map(tr, f"{args.out}/figures/fig2_direct_interpolation_failure_map.png")
    for role in sorted(pr.role.unique()):
        sub = pr[(pr.role == role) & (pr.direction == "cross") & pr.method.isin(learned + ["interaction"])
                 & pr.reference.isin(["D00", "D11-D10-D01+D00"])]
        if len(sub):
            fg.fig_forest(sub, f"{args.out}/figures/fig3_2x2_paired_vs_D00_{role}.png",
                          f"{role} drugs, cross-replicate: arm - D00 (drug-level bootstrap 95% CI)")
    if args.budget:
        bd, _ = load(args.budget)
        bd = bd[(bd.is_oracle == 0) & (bd.direction == "cross")]
        full = ev[(ev.role == "development") & (ev.direction == "cross") & ev.method.isin(bd.method.unique())
                  & (ev.episode.isin(bd.episode.unique()))].copy()
        full["endpoint_budget"] = 0
        bb = pd.concat([bd, full], ignore_index=True)
        bb["budget"] = bb.endpoint_budget.replace(0, 1024)
        rows = []
        for (b, m, ep), s in bb.groupby(["budget", "method", "episode"]):
            c = s.groupby(["line", "drug"], as_index=False)[["comp_tv", "within_mean_rmse", "ed"]].mean()
            d = c.groupby("drug")[["comp_tv", "within_mean_rmse", "ed"]].mean()
            r = dict(budget=b, method=m, episode=ep, n_drugs=len(d))
            for met in ("comp_tv", "within_mean_rmse", "ed"):
                x = ag.drug_block_bootstrap(d[[met]].dropna())
                r[met], r[met + "_lo"], r[met + "_hi"] = x["est"], x["lo"], x["hi"]
            rows.append(r)
        bt = pd.DataFrame(rows)
        bt.round(5).to_csv(f"{args.out}/endpoint_budget.tsv", sep="\t", index=False)
        fg.fig_budget(bt[bt.episode == "E1000"], f"{args.out}/figures/fig4_endpoint_budget_E1000.png")


if __name__ == "__main__":
    main()
