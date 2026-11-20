#!/usr/bin/env python3
"""Score internal out-of-fold predictions (stages A2 and A3).

Usage:
  evaluate_internal.py main     --run-dir RUN --out-dir reports/pathway_sensitivity_a1/tables
  evaluate_internal.py controls --run-dir RUN --controls-dir RUN2 --out-dir ...

`main` scores A0, A1, A1r, A2, A3 and evaluates the pre-registered A3 gate.
`controls` scores the permuted-pathway control (A4) and the 50% learning curve against `main`.
Every expected table must exist; a missing run is an error, never skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
import metrics as M  # noqa: E402

MAIN = ["A0", "A1", "A1r", "A2", "A3"]
PAIRS = [("A1", "A0"), ("A1r", "A0"), ("A2", "A0"), ("A3", "A0"), ("A3", "A1"), ("A3", "A1r"), ("A3", "A2"),
         ("A1r", "A1"), ("A2", "A1r")]


class Scorer:
    def __init__(self, cfg, main_dir):
        self.cfg = cfg
        self.n_rep = cfg["split"]["repeats"]
        reg = core.load_registry()
        self.family_of = dict(zip(reg.model_id, reg.family_id))
        self.long = core.load_limore_response(cfg)
        self.splits = core.load_outer_splits()
        self.repeats = sorted(self.splits.repeat.unique())
        self.families = sorted(self.splits.family_id.unique())
        self.drugs = sorted(self.long.drug_id.unique())
        a0 = M.load_oof(main_dir, "A0", self.n_rep)
        self.s_ref = a0[["repeat", "outer_fold", "drug_id", "s_train"]].drop_duplicates()
        assert not self.s_ref.duplicated(["repeat", "outer_fold", "drug_id"]).any()
        self.W = M.bootstrap_weights(len(self.families), cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"])

    def load(self, run_dir, name, check_s=True):
        pred = M.load_oof(run_dir, name, self.n_rep)
        # every table must cover exactly the split table
        key = pred[["repeat", "model_id"]].drop_duplicates()
        assert len(key) == len(self.splits) and len(pred) == len(self.splits) * len(self.drugs), name
        chk = pred.merge(self.splits, on=["repeat", "model_id"], validate="many_to_one")
        assert (chk.outer_fold_x == chk.outer_fold_y).all(), f"{name}: fold mismatch with split table"
        if check_s:   # full-training runs must share the reference SD
            c = pred.merge(self.s_ref, on=["repeat", "outer_fold", "drug_id"], suffixes=("", "_ref"))
            assert np.allclose(c.s_train, c.s_train_ref, rtol=1e-6), f"{name}: s_train differs from reference"
        df = M.attach_truth(pred, self.long, self.family_of, self.s_ref)
        F = M.error_cube(df, self.repeats, self.drugs, self.families)
        return df, F


def summarise(sc: Scorer, cubes: dict, names: list) -> pd.DataFrame:
    rows = []
    for n in names:
        F = cubes[n]
        per_rep = np.nanmean(M.drug_nrmse(F), axis=1)
        lo, hi = M.ci(M.boot_macro(F, sc.W), *sc.cfg["bootstrap"]["ci"])
        rows.append(dict(model=n, macro_drug_normalized_rmse=per_rep.mean(), ci_low=lo, ci_high=hi,
                         **{f"repeat{r}": v for r, v in zip(sc.repeats, per_rep)},
                         repeat_sd=per_rep.std(ddof=1)))
    return pd.DataFrame(rows)


def differences(sc: Scorer, cubes: dict, pairs: list) -> pd.DataFrame:
    rows = []
    boots = {}
    for a, b in pairs:
        for n in (a, b):
            if n not in boots:
                boots[n] = M.boot_macro(cubes[n], sc.W)
        d = boots[a] - boots[b]
        lo, hi = M.ci(d, *sc.cfg["bootstrap"]["ci"])
        da, db = M.drug_nrmse(cubes[a]).mean(axis=0), M.drug_nrmse(cubes[b]).mean(axis=0)
        rows.append(dict(model=a, reference=b, difference=M.macro_nrmse(cubes[a]) - M.macro_nrmse(cubes[b]),
                         ci_low=lo, ci_high=hi, boot_fraction_below_zero=float((d < 0).mean()),
                         drugs_model_better=int((da < db).sum()), drugs_reference_better=int((da > db).sum()),
                         n_drugs=len(da)))
    return pd.DataFrame(rows)


def cmd_main(args):
    cfg = core.load_config()
    sc = Scorer(cfg, args.run_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    dfs, cubes = {}, {}
    for n in MAIN:
        dfs[n], cubes[n] = sc.load(args.run_dir, n)
    summary = summarise(sc, cubes, MAIN)
    diffs = differences(sc, cubes, PAIRS)

    # ensemble (predictions averaged over repeats first) -- reported under its own name only
    ens_rows = []
    for n in MAIN:
        e = dfs[n].groupby(["model_id", "drug_id", "family_id"]).agg(pred=("pred", "mean"), y=("y", "first")).reset_index()
        # normalise by the repeat-mean reference SD of the drug
        sbar = sc.s_ref.groupby("drug_id").s_train.mean()
        e["z_err2"] = ((e.y - e.pred) / e.drug_id.map(sbar)) ** 2
        fam = e.groupby(["drug_id", "family_id"]).z_err2.mean().groupby("drug_id").mean()
        ens_rows.append(dict(model=n, ensemble_of_repeats_macro_nrmse=float(np.sqrt(fam).mean())))
    summary = summary.merge(pd.DataFrame(ens_rows), on="model")

    # per-drug table (mean over repeats) with exploratory bootstrap intervals for A3 - A0 and A3 - A1
    per = []
    for n in MAIN:
        t = M.per_drug_table(dfs[n])
        t["model"] = n
        per.append(t)
    per = pd.concat(per, ignore_index=True)
    per.to_csv(os.path.join(args.out_dir, "internal_per_drug_per_repeat.tsv"), sep="\t", index=False, float_format="%.6g")
    drug = per.groupby(["model", "drug_id"]).agg(n_families=("n_families", "first"), nrmse=("nrmse", "mean"),
                                                 rmse_raw=("rmse_raw", "mean"), mae_raw=("mae_raw", "mean"),
                                                 spearman=("spearman", "mean")).reset_index()
    wide = drug.pivot(index="drug_id", columns="model", values=["nrmse", "spearman", "rmse_raw", "mae_raw"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    b = {n: M.boot_drug_nrmse(cubes[n], sc.W).mean(axis=1) for n in ("A0", "A1", "A3")}     # [boot, drug]
    for ref in ("A0", "A1"):
        d = b["A3"] - b[ref]
        wide[f"A3_minus_{ref}_nrmse"] = wide["nrmse_A3"] - wide[f"nrmse_{ref}"]
        wide[f"A3_minus_{ref}_ci_low"] = np.nanquantile(d, 0.025, axis=0)
        wide[f"A3_minus_{ref}_ci_high"] = np.nanquantile(d, 0.975, axis=0)
    const = dfs["A0"].groupby("drug_id").constant.any()
    wide["constant_response"] = wide.drug_id.map(const).values
    wide.to_csv(os.path.join(args.out_dir, "internal_per_drug.tsv"), sep="\t", index=False, float_format="%.6g")

    # Spearman summary (secondary): mean over drugs of the repeat-mean per-drug Spearman
    sp = drug.groupby("model").agg(mean_spearman=("spearman", "mean"), median_spearman=("spearman", "median"),
                                   n_drugs_scored=("spearman", "count"), mean_rmse_raw=("rmse_raw", "mean"),
                                   mean_mae_raw=("mae_raw", "mean")).reset_index()
    summary = summary.merge(sp, on="model")
    summary.to_csv(os.path.join(args.out_dir, "internal_summary.tsv"), sep="\t", index=False, float_format="%.6g")
    diffs.to_csv(os.path.join(args.out_dir, "internal_differences.tsv"), sep="\t", index=False, float_format="%.6g")

    # per-family error (mean over drugs and repeats of the family's squared normalised error)
    fam = pd.DataFrame({n: np.sqrt(np.nanmean(cubes[n], axis=(0, 1))) for n in MAIN}, index=sc.families)
    fam.index.name = "family_id"
    reg = core.load_registry()
    members = reg[reg.study == "LIMORE"].groupby("family_id").source_name.apply(lambda s: ",".join(sorted(s)))
    fam.insert(0, "models", members.reindex(fam.index).values)
    fam.sort_values("A3", ascending=False).reset_index().to_csv(
        os.path.join(args.out_dir, "internal_per_family.tsv"), sep="\t", index=False, float_format="%.6g")

    # selected alphas and A2 dimensions
    al = []
    for n in MAIN[1:]:
        t = dfs[n].drop_duplicates(["repeat", "outer_fold", "drug_id"]).groupby("alpha").size().rename("n").reset_index()
        t.insert(0, "model", n)
        al.append(t)
    pd.concat(al).to_csv(os.path.join(args.out_dir, "internal_alpha_counts.tsv"), sep="\t", index=False)
    metas = {n: json.load(open(os.path.join(args.run_dir, "oof", f"{n}__meta.json"))) for n in MAIN}
    folds = pd.DataFrame(metas["A2"]["folds"])
    folds["pca_dim"] = folds.pca_dim.map(lambda v: ",".join(map(str, v)))
    folds.to_csv(os.path.join(args.out_dir, "internal_fold_sizes_and_a2_dims.tsv"), sep="\t", index=False)

    # pre-registered gate
    g = diffs.set_index(["model", "reference"])
    gate = dict(criterion="A3 lower than A0 and A1 with both 95% family-bootstrap intervals entirely below zero",
                A3_minus_A0=g.loc[("A3", "A0"), ["difference", "ci_low", "ci_high"]].to_dict(),
                A3_minus_A1=g.loc[("A3", "A1"), ["difference", "ci_low", "ci_high"]].to_dict())
    gate["met"] = bool(gate["A3_minus_A0"]["ci_high"] < 0 and gate["A3_minus_A1"]["ci_high"] < 0)
    res = dict(run_dir=os.path.basename(os.path.normpath(args.run_dir)), cache_key=core.cache_key(cfg),
               n_models=int(sc.splits.model_id.nunique()), n_families=len(sc.families), n_drugs=len(sc.drugs),
               n_constant_response_drugs=int(const.sum()), n_features={n: metas[n]["n_features"] for n in MAIN},
               bootstrap=cfg["bootstrap"], gate=gate,
               summary=summary.to_dict(orient="records"), differences=diffs.to_dict(orient="records"))
    with open(os.path.join(args.out_dir, "internal_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(summary.to_string())
    print(diffs.to_string())
    print(json.dumps(gate, indent=1))


def cmd_controls(args):
    cfg = core.load_config()
    sc = Scorer(cfg, args.run_dir)
    cubes = {n: sc.load(args.run_dir, n)[1] for n in MAIN}
    gate = json.load(open(os.path.join(args.out_dir, "internal_results.json")))["gate"]
    label = "confirmatory_control" if gate["met"] else "small_diagnostic_gate_not_met"

    # ---- A4: all permutations reported
    seeds = cfg["a4"]["seeds"]
    perm = {s: sc.load(args.controls_dir, f"A4_perm{s:03d}")[1] for s in seeds}
    b3 = M.boot_macro(cubes["A3"], sc.W)
    rows, bperm = [], []
    for s in seeds:
        bp = M.boot_macro(perm[s], sc.W)
        bperm.append(bp)
        lo, hi = M.ci(b3 - bp)
        rows.append(dict(perm_seed=s, macro_drug_normalized_rmse=M.macro_nrmse(perm[s]),
                         A3_minus_perm=M.macro_nrmse(cubes["A3"]) - M.macro_nrmse(perm[s]), ci_low=lo, ci_high=hi))
    a4 = pd.DataFrame(rows)
    a4.to_csv(os.path.join(args.out_dir, "a4_permutations.tsv"), sep="\t", index=False, float_format="%.6g")
    dmean = b3 - np.mean(bperm, axis=0)
    lo, hi = M.ci(dmean)
    a3 = M.macro_nrmse(cubes["A3"])
    drug3 = M.drug_nrmse(cubes["A3"]).mean(axis=0)
    drugp = np.mean([M.drug_nrmse(perm[s]).mean(axis=0) for s in seeds], axis=0)
    a4_res = dict(label=label, n_permutations=len(seeds), A3=a3, perm_mean=float(a4.macro_drug_normalized_rmse.mean()),
                  perm_min=float(a4.macro_drug_normalized_rmse.min()), perm_max=float(a4.macro_drug_normalized_rmse.max()),
                  perm_sd=float(a4.macro_drug_normalized_rmse.std(ddof=1)),
                  n_perm_better_or_equal_than_A3=int((a4.macro_drug_normalized_rmse <= a3).sum()),
                  A3_minus_perm_mean=float(a3 - a4.macro_drug_normalized_rmse.mean()), ci_low=lo, ci_high=hi,
                  drugs_A3_better_than_perm_mean=int((drug3 < drugp).sum()), n_drugs=len(drug3),
                  A0=M.macro_nrmse(cubes["A0"]))

    # ---- learning curve
    lc_rows = []
    frac_cubes = {}
    for n in cfg["learning_curve"]["representations"]:
        frac_cubes[n] = sc.load(args.controls_dir, f"{n}_frac050", check_s=False)[1]
        for frac, F in ((0.5, frac_cubes[n]), (1.0, cubes[n])):
            lo_, hi_ = M.ci(M.boot_macro(F, sc.W))
            lc_rows.append(dict(model=n, train_fraction=frac, macro_drug_normalized_rmse=M.macro_nrmse(F),
                                ci_low=lo_, ci_high=hi_))
    lc = pd.DataFrame(lc_rows)
    lc.to_csv(os.path.join(args.out_dir, "learning_curve.tsv"), sep="\t", index=False, float_format="%.6g")
    lcd = []
    for a, b in [("A3", "A0"), ("A3", "A1"), ("A3", "A1r"), ("A3", "A2"), ("A1", "A0")]:
        d = M.boot_macro(frac_cubes[a], sc.W) - M.boot_macro(frac_cubes[b], sc.W)
        lo_, hi_ = M.ci(d)
        lcd.append(dict(train_fraction=0.5, model=a, reference=b,
                        difference=M.macro_nrmse(frac_cubes[a]) - M.macro_nrmse(frac_cubes[b]), ci_low=lo_, ci_high=hi_))
    # does A3's advantage over A1 grow when training data is halved? (difference of differences)
    dd = (M.boot_macro(frac_cubes["A3"], sc.W) - M.boot_macro(frac_cubes["A1"], sc.W)) - \
         (M.boot_macro(cubes["A3"], sc.W) - M.boot_macro(cubes["A1"], sc.W))
    lo_, hi_ = M.ci(dd)
    interaction = dict(quantity="(A3-A1 at 50%) - (A3-A1 at 100%)",
                       value=(M.macro_nrmse(frac_cubes["A3"]) - M.macro_nrmse(frac_cubes["A1"]))
                       - (M.macro_nrmse(cubes["A3"]) - M.macro_nrmse(cubes["A1"])), ci_low=lo_, ci_high=hi_)
    pd.DataFrame(lcd).to_csv(os.path.join(args.out_dir, "learning_curve_differences.tsv"), sep="\t", index=False,
                             float_format="%.6g")
    meta = json.load(open(os.path.join(args.controls_dir, "oof", "A3_frac050__meta.json")))
    res = dict(label=label, main_run=os.path.basename(os.path.normpath(args.run_dir)),
               controls_run=os.path.basename(os.path.normpath(args.controls_dir)), a4=a4_res,
               learning_curve=lc.to_dict(orient="records"), learning_curve_differences=lcd,
               learning_curve_interaction=interaction,
               train_families_at_50pct=sorted({f["n_train_families"] for f in meta["folds"]}))
    with open(os.path.join(args.out_dir, "controls_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "controls"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--controls-dir")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    {"main": cmd_main, "controls": cmd_controls}[a.cmd](a)
