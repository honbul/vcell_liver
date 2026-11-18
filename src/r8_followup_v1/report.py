"""Assemble the P1 / P2 tables of the follow-up from immutable run directories (no prediction is recomputed).

usage: python -m src.r8_followup_v1.report --p1 <run> --select <inner0> <inner1> <inner2> <final_inner> --evaluate <outer0> <outer1> <outer2>
Writes covariance_repair_versions.tsv, covariance_effect_paths.tsv, fixed_hyper_vs_retuned.tsv, matched_plate_repaired.tsv,
residual_components.tsv, kernel_components.tsv, source_oof_summary.tsv, source_oof_contrasts.tsv and report_inputs.json.
"""
import argparse, json, os
import numpy as np
import pandas as pd

from .common import pub_man, dump, sha256
from . import stats as S

SETS = ("source_oof", "E1", "E2")
METRICS = ["delta_rmse", "centred_rmse", "centred_pearson", "delta_pearson", "mrr_tie_centred", "mrr_legacy_centred", "mrr_tie", "top5_tie_centred",
           "pred_rms", "truth_rms", "mrr_chance"]


def load_eval(dirs):
    gen, mat, infl, n0, perm, nui = [], [], [], [], [], {}
    for d in dirs:
        gen.append(pd.read_csv(os.path.join(d, "episodes.tsv.gz"), sep="\t"))
        mat.append(pd.read_csv(os.path.join(d, "matched_episodes.tsv.gz"), sep="\t"))
        infl.append(pd.read_csv(os.path.join(d, "influence.tsv"), sep="\t"))
        n0.append(pd.read_csv(os.path.join(d, "n0_strata.tsv"), sep="\t"))
        perm.append(pd.read_csv(os.path.join(d, "row_permutation_check.tsv"), sep="\t"))
        r = json.load(open(os.path.join(d, "result.json")))
        nui[r["partition"]] = json.load(open(os.path.join(d, "outer_nuisance.json")))
    return [pd.concat(x, ignore_index=True) for x in (gen, mat, infl, n0, perm)] + [nui]


def as_sets(gen, mat):
    """One frame with a 'set' column: source_oof general episodes, E1 = in_block, E2 = other_block (matched episodes averaged per
    ordered plate pair first, so that a pair is one episode as in the original matched-plate table)."""
    g = gen.assign(set="source_oof")
    m = mat[mat.block.isin(["in_block", "other_block"])].copy()
    num = [c for c in m.columns if c in METRICS or c in ("k_eff", "alpha", "alpha_omega", "alpha_iid", "w_min_xK", "w_max_xK", "n_negative_w", "cond",
                                                          "eig_min", "eig_max", "trace_over_K", "b_rms", "r_rms", "r_query_var", "fstar_rms", "corr_mean_rms",
                                                          "corr_spec_rms", "ctrl_shift_rms", "n_query")]
    m = m.groupby(["fold", "cell", "family", "plate_query", "plate_other", "support_seed", "block", "K", "method"])[num].mean().reset_index()
    m["set"] = np.where(m.block == "in_block", "E1", "E2")
    m["support_seed"] = m.plate_query + "|" + m.plate_other            # an ordered plate pair plays the role of a support seed
    return pd.concat([g, m.drop(columns=["block", "plate_query", "plate_other"])], ignore_index=True)


def fold_map(df):
    return df.drop_duplicates("family").set_index("family").fold.to_dict()


def contrasts(df, pairs, value="delta_rmse", sets=SETS, ks=(8, 16, 32)):
    fm = fold_map(df)
    out = []
    for s in sets:
        d = df[(df.set == s) & df.K.isin(ks)]
        if not len(d):
            continue
        fam = S.family_table(d, value, by=("K", "method"))
        for a, b, label in pairs:
            c = S.paired(fam, a, b, value=value, fold_of=fm, extra_by=("K",))
            if len(c):
                c.insert(0, "set", s)
                c.insert(1, "label", label)
                out.append(c)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summary(df, methods, sets=SETS):
    out = []
    for s in sets:
        d = df[(df.set == s) & df.method.isin(methods)]
        if not len(d):
            continue
        r = None
        for v in METRICS:
            if v not in d:
                continue
            fam = S.family_table(d, v, by=("K", "method"))
            a = fam.groupby(["K", "method"])[v].agg(["mean", "count"]).reset_index().rename(columns={"mean": v, "count": "n_families"})
            r = a if r is None else r.merge(a.drop(columns="n_families"), on=["K", "method"])
        r.insert(0, "set", s)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def cov_versions_table(p1, sel_dirs, ev_dirs):
    rows = []
    g = json.load(open(os.path.join(p1, "cov_versions_global.json")))
    for v in g.values():
        rows.append(dict(partition="global", role="original V0 reproduction / fixed-hyper reference (kappa_f = 8 for V3)", **v))
    for d in sel_dirs:
        c = json.load(open(os.path.join(d, "constants.json")))
        for v in c["cov"].values():
            rows.append(dict(partition=c["partition"], role="inner selection (nuisance on inner-train families)", **v))
    for d in ev_dirs:
        r = json.load(open(os.path.join(d, "result.json")))
        n = json.load(open(os.path.join(d, "outer_nuisance.json")))
        for k, v in n["cov"].items():
            if k == "V3_fx":
                continue
            rows.append(dict(partition=r["partition"], role="outer refit (nuisance on outer-train families)", **dict(v, version=k if not k.startswith("V3_kf") else k)))
    t = pd.DataFrame(rows)
    first = ["partition", "version", "role", "n_families", "n_cells", "n_units", "n_units_overlap_summary", "unit_list_sha256", "residual_reference", "kappa_f",
             "v_tot", "q_mean", "o_diag_mean", "o_off_mean", "v_e", "v_s", "v_s_raw", "v_r_raw", "v_r_floor", "v_r", "floor_applied",
             "frac_cells_raw_negative", "frac_cells_below_floor", "frac_families_raw_negative", "n_pairs", "n_pair_cells", "n_pair_families", "sufficient"]
    return t[[c for c in first if c in t] + [c for c in t if c not in first]]


def path_diagnostics(df):
    d = df[df.method.str.startswith("P_both_")].copy()
    d["version"] = d.method.str.replace("P_both_", "", regex=False)
    cols = ["k_eff", "alpha_iid", "alpha_omega", "w_min_xK", "w_max_xK", "n_negative_w", "cond", "eig_min", "eig_max", "trace_over_K"]
    a = d.groupby(["set", "K", "version"])[cols].agg(["mean", "min", "max"])
    a.columns = [f"{x}_{y}" for x, y in a.columns]
    return a.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", required=True)
    ap.add_argument("--select", nargs=4, required=True)
    ap.add_argument("--evaluate", nargs=3, required=True)
    a = ap.parse_args()
    gen, mat, infl, n0, perm, nui = load_eval(a.evaluate)
    df = as_sets(gen, mat)
    W = lambda name, t: t.to_csv(pub_man(name), sep="\t", index=False, float_format="%.6g")

    # ---- covariance versions
    W("covariance_repair_versions.tsv", cov_versions_table(a.p1, a.select, a.evaluate))

    # ---- C0 four paths + Omega diagnostics
    pairs = [(f"P_{p}_{v}", "C0_iid", f"C0 path {p} ({v}) vs C0-iid; common kappa_b = retuned iid value") for v in ("V0", "V3", "V3s")
             for p in ("weights_only", "shrink_only", "both")]
    pairs += [(f"P_both_{v}", f"P_shrink_only_{v}", f"both vs shrink-only ({v})") for v in ("V0", "V3")]
    pairs += [(f"P_both_{v}", f"P_weights_only_{v}", f"both vs weights-only ({v})") for v in ("V0", "V3")]
    pairs += [("P_both_V3", "P_both_V0", "V3 vs V0 (both paths)"), ("P_both_V3s", "P_both_V3", "V3 trace-matched to V0 vs V3")]
    cp = contrasts(df, pairs)
    dg = path_diagnostics(df)
    cp = cp.merge(dg.assign(a=lambda x: "P_both_" + x.version).drop(columns="version"), on=["set", "K", "a"], how="left")
    W("covariance_effect_paths.tsv", cp)

    # ---- fixed-hyper vs retuned
    fx = [("fx_C0_V0", "fx_C0_iid", "fixed-hyper"), ("fx_C0_V3", "fx_C0_iid", "fixed-hyper"), ("fx_C0_V3s", "fx_C0_iid", "fixed-hyper"),
          ("fx_C0_V3", "fx_C0_V0", "fixed-hyper"), ("fx_C1_V0", "fx_C1_iid", "fixed-hyper"), ("fx_C1_V3", "fx_C1_iid", "fixed-hyper"),
          ("fx_C1_V3s", "fx_C1_iid", "fixed-hyper"), ("fx_C1_V3", "fx_C1_V0", "fixed-hyper"), ("fx_C1_iid", "fx_C0_iid", "fixed-hyper"),
          ("C0_V0", "C0_iid", "source-retuned (nested)"), ("C0_V3", "C0_iid", "source-retuned (nested)"), ("C0_V3", "C0_V0", "source-retuned (nested)"),
          ("C1_V0", "C1_iid", "source-retuned (nested)"), ("C1_V3", "C1_iid", "source-retuned (nested)"), ("C1_V3", "C1_V0", "source-retuned (nested)"),
          ("C1_iid", "C0_iid", "source-retuned (nested)"), ("C1_V3", "C0_V3", "source-retuned (nested)"),
          ("C0_iid", "fx_C0_iid", "retuned vs fixed constants"), ("C1_iid", "fx_C1_iid", "retuned vs fixed constants")]
    fh = pd.concat([contrasts(df, fx), contrasts(df, fx, value="centred_rmse")], ignore_index=True)
    fh["note"] = np.where(fh.label == "fixed-hyper", "source development analysis: original constants were selected on all source-train families (incl. the held fold)",
                          "strict out-of-fold: constants from the inner split of the outer-train families, nuisance refit on outer-train")
    W("fixed_hyper_vs_retuned.tsv", fh)

    # ---- matched plate E1 / E2
    mm = ["shrunk_mean", "C0_iid", "C1_iid", "C0_V0", "C0_V3", "C1_V0", "C1_V3", "fx_C0_iid", "fx_C0_V0", "fx_C0_V3", "fx_C1_iid", "fx_C1_V0", "fx_C1_V3",
          "P_weights_only_V3", "P_shrink_only_V3", "P_both_V3", "J1", "J2", "kn_all", "kn_chem+f0", "C1_iid_shuf1"]
    E = df[df.set.isin(["E1", "E2"])]
    rows = []
    fm = fold_map(E)
    for v in ("delta_rmse", "centred_rmse"):
        fam = S.family_table(E[E.method.isin(mm)], v, by=("K", "set", "method"))
        for (K, m), h in fam.groupby(["K", "method"]):
            p = h.pivot(index="family", columns="set", values=v).dropna()
            if len(p) < 2:
                continue
            d = (p["E1"] - p["E2"]).to_numpy()
            r = dict(row="method_level", K=K, name=m, metric=v, E1=float(p["E1"].mean()), E2=float(p["E2"].mean()), n_families=len(d))
            r.update({k: x for k, x in S.summarise_diff(d, float(p["E2"].mean())).items() if k in ("diff", "lo", "hi")})
            rows.append(r)
    mpairs = [("C0_iid", "shrunk_mean", "common correction"), ("C1_iid", "C0_iid", "per-drug correction"), ("C0_V3", "C0_iid", "covariance V3 (C0, retuned)"),
              ("C0_V0", "C0_iid", "covariance V0 (C0, retuned)"), ("fx_C0_V0", "fx_C0_iid", "covariance V0 (C0, fixed)"), ("fx_C0_V3", "fx_C0_iid", "covariance V3 (C0, fixed)"),
              ("C1_V3", "C1_iid", "covariance V3 (C1, retuned)"), ("fx_C1_V3", "fx_C1_iid", "covariance V3 (C1, fixed)"), ("C1_V0", "C1_iid", "covariance V0 (C1, retuned)"),
              ("P_shrink_only_V3", "C0_iid", "shrink-only V3"), ("P_weights_only_V3", "C0_iid", "weights-only V3"), ("P_both_V3", "C0_iid", "both V3"),
              ("J1", "C0_iid", "J1 (query overlap)"), ("J2", "C0_iid", "J2 (joint)"), ("J2", "J1", "J2 vs J1"), ("C1_iid_shuf1", "C1_iid", "C1 residual shuffle 1")]
    for v in ("delta_rmse", "centred_rmse"):
        c = contrasts(E, mpairs, value=v, sets=("E1", "E2"))
        for r in c.itertuples():
            rows.append(dict(row=f"contrast_{r.set}", K=r.K, name=f"{r.a}-{r.b}", label=r.label, metric=v, diff=r.diff, lo=r.lo, hi=r.hi, n_families=r.n_families,
                             n_folds_a_lower=r.n_folds_a_lower, fold_diffs=r.fold_diffs))
        fam = S.family_table(E, v, by=("K", "set", "method"))
        for a_, b_, lab in mpairs:
            for K, h in fam.groupby("K"):
                p1_ = h[h.set == "E1"].pivot(index="family", columns="method", values=v)
                p2_ = h[h.set == "E2"].pivot(index="family", columns="method", values=v)
                if not all(x in p1_ and x in p2_ for x in (a_, b_)):
                    continue
                d = ((p1_[a_] - p1_[b_]) - (p2_[a_] - p2_[b_])).dropna().to_numpy()
                r = dict(row="contrast_E1_minus_E2", K=K, name=f"{a_}-{b_}", label=lab, metric=v, n_families=len(d))
                r.update({k: x for k, x in S.summarise_diff(d, 1.0).items() if k in ("diff", "lo", "hi")})
                rows.append(r)
    # positive control (replicate) and the common K = 32 cohort
    pc = mat[mat.block == "replicate"].groupby(["fold", "cell", "family", "plate_query", "plate_other", "method"])[["delta_rmse", "centred_rmse", "delta_pearson",
                                                                                                                   "centred_pearson", "mrr_tie_centred", "mrr_chance"]].mean().reset_index()
    for v in ("delta_rmse", "centred_rmse", "delta_pearson", "centred_pearson", "mrr_tie_centred", "mrr_chance"):
        fam = pc.groupby(["method", "family", "cell"])[v].mean().groupby(level=[0, 1]).mean().reset_index()
        for m, h in fam.groupby("method"):
            rows.append(dict(row="positive_control", K=0, name=m, metric=v, E2=float(h[v].mean()), n_families=len(h)))
    k32 = E[E.K == 32][["cell", "support_seed"]].drop_duplicates()
    Ec = E.merge(k32, on=["cell", "support_seed"])
    c = contrasts(Ec, [("C1_iid", "C0_iid", "per-drug correction (K=32 common cohort)"), ("C0_iid", "shrunk_mean", "common correction (K=32 common cohort)"),
                       ("C0_V3", "C0_iid", "covariance V3 (K=32 common cohort)")], sets=("E1", "E2"))
    for r in c.itertuples():
        rows.append(dict(row=f"common_cohort_k32_{r.set}", K=r.K, name=f"{r.a}-{r.b}", label=r.label, metric="delta_rmse", diff=r.diff, lo=r.lo, hi=r.hi, n_families=r.n_families))
    mt = pd.DataFrame(rows)
    pk = mat.drop_duplicates(["cell", "plate_query", "plate_other"]).groupby("pair_kind").size().to_dict()
    cs = mat[mat.block == "in_block"].drop_duplicates(["cell", "plate_query", "plate_other"]).ctrl_shift_rms
    mt["pair_kinds"] = json.dumps(pk)
    mt["support_ctrl_shift_rms_mean"] = float(cs.mean())
    mt["note"] = "E1 = support on the query plate; E2 = support on the other X-tag replicate plate (one compound plate; not documented independent cultures); E2 truth is the noisy plate-A measurement, not the latent mean"
    W("matched_plate_repaired.tsv", mt)

    # ---- residual components, shuffle, influence
    comp_cols = ["b_rms", "r_rms", "r_query_var", "fstar_rms", "corr_mean_rms"]
    rc = []
    for s in SETS:
        d = df[(df.set == s) & (df.method == "C1_iid")]
        for v in comp_cols:
            fam = S.family_table(d, v, by=("K",))
            for K, h in fam.groupby("K"):
                rc.append(dict(set=s, K=K, row="component_C1_iid", name=v, value=float(h[v].mean()), n_families=len(h)))
        d = df[(df.set == s) & (df.method == "J2")]
        for v in ("corr_mean_rms", "corr_spec_rms"):
            if v in d:
                fam = S.family_table(d, v, by=("K",))
                for K, h in fam.groupby("K"):
                    rc.append(dict(set=s, K=K, row="component_J2", name=v, value=float(h[v].mean()), n_families=len(h)))
        for m in ("shrunk_mean", "C0_iid", "C1_iid", "J1", "J2", "C1_iid_shuf1", "C1_iid_shuf2", "C1_iid_shuf3"):
            dd = df[(df.set == s) & (df.method == m)]
            for v in ("delta_rmse", "centred_rmse", "pred_rms", "truth_rms", "centred_pearson", "mrr_tie_centred", "mrr_legacy_centred"):
                fam = S.family_table(dd, v, by=("K",))
                for K, h in fam.groupby("K"):
                    rc.append(dict(set=s, K=K, row=f"metric_{m}", name=v, value=float(h[v].mean()), n_families=len(h)))
    rct = pd.DataFrame(rc)
    sh = contrasts(df, [(f"C1_iid_shuf{j}", "C1_iid", f"C1 direct residual shuffle, permutation {j}") for j in (1, 2, 3)] +
                   [("C1_iid", "C0_iid", "correct correspondence vs no residual correction"),
                    ("C1_iid_shuf1", "C0_iid", "shuffled correspondence vs no residual correction")], value="delta_rmse")
    sh2 = contrasts(df, [(f"C1_iid_shuf{j}", "C1_iid", f"C1 direct residual shuffle, permutation {j}") for j in (1, 2, 3)] +
                    [("C1_iid", "C0_iid", "correct correspondence vs no residual correction")], value="centred_rmse")
    shc = pd.concat([sh, sh2], ignore_index=True)
    shc = shc.rename(columns={"diff": "value"}).assign(row="contrast", name=lambda x: x.a + "-" + x.b)
    inf = infl.groupby("family")[[c for c in infl.columns if c not in ("fold", "cell", "family", "support_seed", "K")]].mean()
    inr = [dict(set="source_oof", K=16, row="influence_C1_iid", name=c, value=float(inf[c].mean()), n_families=len(inf)) for c in inf.columns]
    n0a = n0.groupby(["family", "method", "n0_bin"])[["sum_rmse", "n"]].sum().reset_index()
    n0a["mean_rmse"] = n0a.sum_rmse / n0a.n
    n0r = [dict(set="source_oof", K=16, row=f"n0_stratum_{m}", name=b, value=float(h.mean_rmse.mean()), n_families=h.family.nunique(), n_units=int(h.n.sum()))
           for (m, b), h in n0a.groupby(["method", "n0_bin"])]
    ref = contrasts(df[df.set == "source_oof"], [("C0_iid_refhalf0", "C0_iid", "reference from half 0 of training families"),
                                                  ("C0_iid_refhalf1", "C0_iid", "reference from half 1"), ("C1_iid_refhalf0", "C1_iid", "reference half 0"),
                                                  ("C1_iid_refhalf1", "C1_iid", "reference half 1"), ("C1_iid_refhalf0", "C0_iid_refhalf0", "C1-C0 with half-0 reference"),
                                                  ("C1_iid_refhalf1", "C0_iid_refhalf1", "C1-C0 with half-1 reference")], sets=("source_oof",), ks=(16,))
    ref = ref.rename(columns={"diff": "value"}).assign(row="reference_half", name=lambda x: x.a + "-" + x.b)
    pm = perm[[c for c in perm.columns if c.startswith("maxabs_")]].max()
    pmr = [dict(set="source_oof", K=16, row="full_row_permutation_max_abs_change", name=c[7:], value=float(v), n_families=int(perm.family.nunique())) for c, v in pm.items()]
    W("residual_components.tsv", pd.concat([rct, shc, pd.DataFrame(inr), pd.DataFrame(n0r), ref, pd.DataFrame(pmr)], ignore_index=True))

    # ---- kernel components
    cfgs = ["chem", "f0", "ctrl", "chem+f0", "all"]
    kp = [(f"kr_{c}", "C0_iid", f"original scale, C1-iid constants (sensitivity): {c}") for c in cfgs if c != "all"]
    kp += [("C1_iid", "C0_iid", "original scale: all (= C1-iid)")]
    kp += [(f"kn_{c}", "C0_iid", f"block-normalised, own lambda/eta (information value): {c}") for c in cfgs]
    kp += [(f"kn_{c}", "kn_all", f"normalised {c} vs normalised all") for c in cfgs if c != "all"]
    kp += [("kn_all", "C1_iid", "normalised all vs original-scale all"), ("kr_chem+f0", "C1_iid", "drop ctrl block (original scale)")]
    kc = pd.concat([contrasts(df, kp), contrasts(df, kp, value="centred_rmse")], ignore_index=True)
    sc = []
    for part, n in nui.items():
        for K, s in n["scales"].items():
            sc.append(dict(partition=part, K=int(K), **{f"scale_{b}": v for b, v in s["block"].items()}, **{f"comb_{c}": v for c, v in s["comb"].items()},
                           **{f"zero_trace_{b}": v for b, v in s["zero_trace_fraction"].items()}, n_scale_episodes=s["n_episodes"]))
    kc = kc.merge(pd.DataFrame(sc).groupby("K").mean(numeric_only=True).add_prefix("mean_").reset_index().rename(columns={"mean_K": "K"}), on="K", how="left") if len(sc) else kc
    W("kernel_components.tsv", kc)
    pd.DataFrame(sc).to_csv(pub_man("kernel_scales_by_fold.tsv"), sep="\t", index=False, float_format="%.6g")

    # ---- primary source table and the main contrasts
    main_m = ["no_change", "same_drug_mean", "shrunk_mean", "C0_iid", "C1_iid", "C0_V0", "C0_V3", "C1_V0", "C1_V3", "fx_C0_iid", "fx_C1_iid", "fx_C0_V0", "fx_C1_V0",
              "fx_C0_V3", "fx_C1_V3", "J1", "J2", "J1_unknown_env", "J2_unknown_env", "kn_all", "kn_chem+f0", "kn_f0", "kn_chem", "kn_ctrl"]
    W("source_oof_summary.tsv", summary(df, main_m))
    mc = [("C0_iid", "shrunk_mean", "common correction"), ("C1_iid", "C0_iid", "per-drug correction"), ("J1", "C0_iid", "J1 vs J0 (C0-iid)"),
          ("J2", "C0_iid", "J2 vs J0 (C0-iid)"), ("J2", "J1", "J2 vs J1 (same metadata)"), ("J1", "C0_V3", "J1 vs J0 (C0-V3)"), ("J2", "C0_V3", "J2 vs J0 (C0-V3)"),
          ("J2_unknown_env", "C0_iid", "J2 unknown-environment mode vs C0-iid"), ("J1_unknown_env", "C0_iid", "J1 unknown-environment mode vs C0-iid"),
          ("J2", "C1_iid", "J2 vs C1-iid")]
    W("source_oof_contrasts.tsv", pd.concat([contrasts(df, mc), contrasts(df, mc, value="centred_rmse")], ignore_index=True))
    dump(dict(p1=a.p1, select=a.select, evaluate=a.evaluate, results={d: json.load(open(os.path.join(d, "result.json"))) for d in a.evaluate + a.select + [a.p1]}),
         pub_man("report_inputs.json"))
    print("done")


if __name__ == "__main__":
    main()
