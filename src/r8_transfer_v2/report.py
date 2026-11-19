"""Tables of R8 v2 from immutable run directories (no prediction is recomputed).

  p0   P0-A anchor_matched rows and decomposition, P0-B K-consistency (keys, constant changes, panel impact), P0-C tie impact
  e1   E1 / E2-A source tables, primary decisions (Holm), compound-plate sensitivity, selected constants, episode manifest
  hcc  frozen HCC follow-up table (after the source freeze)
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

from src.r8_followup_v1 import stats as S
from src.r8_followup_v1.report import contrasts as v1_contrasts
from src.r8_followup_v1.final_report import hcc_tables
from .common import pub_man, dump, sha256, v1_man, CFG, MAN, log
import yaml

BOOT_SEED, N_BOOT = 20261117, 2000
W = lambda name, t: t.to_csv(pub_man(name), sep="\t", index=False, float_format="%.7g")


def boot_p(draws):
    return float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))))


def fam_values(df, value):
    """episodes -> cells -> families (equal weights)."""
    return S.family_table(df, value, by=("method",))


def pair_stat(fam, a, b, value, fold_of=None):
    p = fam.pivot(index="family", columns="method", values=value)
    if a not in p or b not in p:
        return None
    p = p[[a, b]].dropna()
    d = (p[a] - p[b]).to_numpy()
    if len(d) < 2:
        return None
    return diff_summary(pd.Series(d, index=p.index), float(p[b].mean()), fold_of, mean_a=float(p[a].mean()))


def diff_summary(ds, ref, fold_of=None, **extra):
    d = ds.to_numpy()
    bs = S.boot_mean(d)
    r = dict(n_families=len(d), n_families_a_lower=int((d < 0).sum()), mean_b=ref, **extra)
    r.update(S.summarise_diff(d, ref, draws=bs))
    r["p_boot"] = boot_p(bs)
    if fold_of is not None:
        f = ds.groupby(ds.index.map(fold_of)).mean()
        r.update(fold_diffs=";".join(f"{k}:{v:+.5f}" for k, v in f.items()), n_folds_a_lower=int((f < 0).sum()), n_folds=int(len(f)))
        r["adoption_rule_met"] = bool(r["diff"] < 0 and r["ci_below_0"] and r["n_folds_a_lower"] >= 2)
    return r


def cluster_boot(units, n_boot=N_BOOT, seed=BOOT_SEED):
    """units: DataFrame (family, cluster, d). Resample clusters; statistic = mean over families of their drawn-unit mean."""
    ci, cl = pd.factorize(units.cluster, sort=True)
    fi, fa = pd.factorize(units.family, sort=True)
    d = units.d.to_numpy(float)
    rs = np.random.RandomState(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rs.randint(len(cl), size=len(cl)), minlength=len(cl))[ci].astype(float)
        sw = np.bincount(fi, weights=w, minlength=len(fa))
        swd = np.bincount(fi, weights=w * d, minlength=len(fa))
        ok = sw > 0
        out[i] = (swd[ok] / sw[ok]).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return dict(cluster_n=len(cl), cluster_lo=float(lo), cluster_hi=float(hi), cluster_p=boot_p(out))


# ================================================================= P0
def load_runs(dirs, name, **kw):
    out = []
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            out.append(pd.read_csv(p, sep="\t", **kw))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def p0(a):
    sel = a.p0_select
    # ---- P0-C tie impact
    t1 = load_runs(sel, "tie_log_v1grids.tsv").assign(grid_run="v1 nuisance, v1 grids (the selections v1 actually made)")
    t2 = load_runs(sel, "tie_log_v2grids.tsv").assign(grid_run="v2 nuisance, v2 tie rule (corrected selection)")
    tt = pd.concat([t1, t2], ignore_index=True)
    tt["exact_tie"] = tt.n_exact_tie_at_best > 1
    tt["affected_results"] = np.where(~tt.exact_tie, "none: no exact tie at the best unrounded score",
                                      np.where(tt.selection_changed, "CHANGED: dependent rows must be recomputed", "exact tie but v1 and v2 rules pick the same candidate"))
    W("tie_rule_impact.tsv", tt)
    # ---- P0-B
    checks = {}
    for d in sel:
        c = json.load(open(os.path.join(d, "checks.json")))
        checks[c["partition"]] = c
    keys = pd.concat([load_runs(sel, "nuisance_keys.tsv").assign(context="inner selection"),
                      load_runs([a.p0_k8], "nuisance_keys_global.tsv").assign(context="global (confirmation / HCC evaluation)")], ignore_index=True)
    W("k8_nuisance_keys.tsv", keys)
    cd = []
    for P, c in checks.items():
        for k in ("replay_const_diffs", "replay_j_diffs", "budget_replay_diffs", "v2_vs_v1_const_diffs", "v2_vs_v1_j_diffs"):
            for x in c.get(k, []) or ["(none)"]:
                cd.append(dict(partition=P, comparison=k, difference=x))
    W("k8_constant_changes.tsv", pd.DataFrame(cd))
    # panel impact (exposed panels; correction report only)
    k8 = pd.read_csv(os.path.join(a.p0_k8, "panel_episodes.tsv.gz"), sep="\t")
    k8["method_v"] = k8.method + "|" + k8.variant
    rows = []
    for st_, d in k8.groupby("set"):
        for v in ("delta_rmse", "centred_rmse"):
            fam = S.family_table(d.assign(method=d.method_v), v, by=("K", "method"))
            for (K, m), h in fam.groupby(["K", "method"]):
                rows.append(dict(panel=st_, target="", K=K, method=m.split("|")[0], variant=m.split("|")[1], metric=v, value=float(h[v].mean()), n_families=len(h)))
            for K in (8, 16, 32):
                fk = fam[fam.K == K]
                for m in sorted(d.method.unique()):
                    for var in ("fixed", "reselected"):
                        r = pair_stat(fk, f"{m}|{var}", f"{m}|replay", v)
                        if r:
                            rows.append(dict(panel=st_, target="", K=K, method=m, variant=f"{var}-replay", metric=v, value=r["diff"], lo=r["lo"], hi=r["hi"], n_families=r["n_families"]))
    u = pd.read_csv(os.path.join(a.p0_k8, "hcc_per_unit.tsv.gz"), sep="\t")
    u["method"] = u.method + "|" + u.variant
    pairs = [(f"{m}|{var}", f"{m}|replay") for m in sorted(set(x.split("|")[0] for x in u.method.unique())) for var in ("fixed", "reselected")]
    res, con, _ = hcc_tables(u, pairs, "test")
    for r in res[res.metric.isin(["delta_rmse", "centred_rmse"])].itertuples():
        rows.append(dict(panel="hcc", target=r.target, K=r.K, method=r.method.split("|")[0], variant=r.method.split("|")[1], metric=r.metric, value=r.value, lo=r.lo, hi=r.hi))
    for r in con[con.metric.isin(["delta_rmse", "centred_rmse"])].itertuples():
        m1, m2 = r.contrast.split("-", 1)
        rows.append(dict(panel="hcc", target=r.target, K=r.K, method=m1.split("|")[0], variant=f"{m1.split('|')[1]}-replay", metric=r.metric, value=r.diff, lo=r.lo, hi=r.hi))
    W("k8_panel_impact.tsv", pd.DataFrame(rows))
    k8chk = json.load(open(os.path.join(a.p0_k8, "checks.json")))
    # ---- P0-A
    ep = load_runs(a.p0_anchor, "episodes.tsv.gz")
    ep["set2"] = np.where(ep.fold == "global", np.where(ep.set == "general", "confirmation", "confirmation_" + ep.set), np.where(ep.set == "general", "source_oof", "source_" + ep.set))
    df = ep.drop(columns="set").rename(columns={"set2": "set"})
    df["support_seed"] = df.support_seed.astype(str) + "|" + df.get("plate_query", pd.Series([""] * len(df))).fillna("").astype(str) + "|" + df.get("plate_other", pd.Series([""] * len(df))).fillna("").astype(str)
    sets = ("source_oof", "source_E1", "source_E2", "confirmation", "confirmation_E1", "confirmation_E2")
    meta = df.groupby(["set", "K", "method"]).agg(anchor_kappa_f=("anchor_kappa_f", lambda x: ";".join(sorted(set(f"{v:g}" for v in x)))),
                                                   anchor_contract_id=("anchor_contract_id", lambda x: ";".join(sorted(set(x)))),
                                                   selection_partition=("selection_partition", lambda x: ";".join(sorted(set(x))))).reset_index()
    summ = []
    for s in sets:
        d = df[df.set == s]
        for v in ("delta_rmse", "centred_rmse", "centred_pearson", "mrr_tie_centred", "pred_rms"):
            fam = S.family_table(d, v, by=("K", "method"))
            for (K, m), h in fam.groupby(["K", "method"]):
                summ.append(dict(set=s, K=K, method=m, metric=v, value=float(h[v].mean()), n_families=len(h)))
    summ = pd.DataFrame(summ).merge(meta, on=["set", "K", "method"], how="left")
    hu = load_runs(a.p0_anchor, "hcc_per_unit.tsv.gz")
    dec_pairs = [("C0_iid", "anchor_matched"), ("C1_iid", "C0_iid"), ("C1_iid", "anchor_matched"), ("C0_iid", "shrunk_mean"), ("shrunk_mean", "anchor_matched")]
    hres, hcon, _ = hcc_tables(hu, dec_pairs, "test")
    kf_h = hu.groupby(["K"]).anchor_kappa_f.first().to_dict()
    for r in hres[hres.metric.isin(["delta_rmse", "centred_rmse", "centred_pearson", "mrr_tie_centred"])].itertuples():
        summ = pd.concat([summ, pd.DataFrame([dict(set=f"hcc_{r.target}", K=r.K, method=r.method, metric=r.metric, value=r.value, lo=r.lo, hi=r.hi,
                                                   anchor_kappa_f=f"{kf_h[r.K]:g}" if r.method in ("anchor_matched", "C0_iid", "C1_iid") else "",
                                                   anchor_contract_id=f"kappa_f={kf_h[r.K]:g}|final_inner" if r.method in ("anchor_matched", "C0_iid", "C1_iid") else "",
                                                   selection_partition="final_inner")])], ignore_index=True)
    summ.loc[summ.method == "shrunk_mean", "anchor_contract_id"] = "independently selected shrunk kappa_f (different anchor)"
    W("anchor_matched_rows.tsv", summ)
    labels = {("C0_iid", "anchor_matched"): "common correction (same anchor)", ("C1_iid", "C0_iid"): "per-drug correction",
              ("C1_iid", "anchor_matched"): "total over same anchor", ("C0_iid", "shrunk_mean"): "v1 'common correction' row (different anchors)",
              ("shrunk_mean", "anchor_matched"): "anchor choice (shrunk kappa_f vs C0 kappa_f)"}
    dec = []
    for v in ("delta_rmse", "centred_rmse"):
        c = v1_contrasts(df, [(x, y, labels[(x, y)]) for x, y in dec_pairs], value=v, sets=sets)
        for r in c.to_dict("records"):
            dec.append(r)
    for r in hcon[hcon.metric.isin(["delta_rmse", "centred_rmse"])].itertuples():
        x, y = r.contrast.split("-", 1)
        dec.append(dict(set=f"hcc_{r.target}", label=labels.get((x, y), ""), K=r.K, a=x, b=y, metric=r.metric, diff=r.diff, lo=r.lo, hi=r.hi))
    W("anchor_decomposition.tsv", pd.DataFrame(dec))
    # reproduction rows relabelled
    rp = pd.read_csv(v1_man("v0_reproduction_existing_panels.tsv"), sep="\t")
    rp["row_kind"] = np.where(rp.followup.str.startswith("fx_"), "original-constant reproduction (C0/C1 with the original R8 constants)",
                              np.where(rp.followup == "shrunk_mean", "NOT a reproduction: shrunk_mean with an independently reselected kappa_f (v1 final_inner 4 vs original 8)",
                                       "constant-free baseline (reproduction)"))
    W("reproduction_row_kinds.tsv", rp)
    ach = {os.path.basename(d): json.load(open(os.path.join(d, "replay_checks.json"))) for d in a.p0_anchor}
    dump(dict(p0_select=sel, p0_anchor=a.p0_anchor, p0_k8=a.p0_k8, select_checks=checks, anchor_replay=ach, k8_checks=k8chk,
              reproduction_counts=rp.row_kind.value_counts().to_dict()), pub_man("p0_report_inputs.json"))
    log("p0 done")


# ================================================================= E1
SETS_E1 = [("G", "planned"), ("G", "missing_reselected"), ("G", "missing_zero_overlap_fallback"), ("M1", "planned"), ("M1", "fixed_gen_constants"),
           ("M1", "missing_reselected"), ("M1", "missing_zero_overlap_fallback"), ("M2", "known_separate"), ("M2", "fixed_gen_constants"), ("M2", "missing_reselected")]
E1_PAIRS = [("J3", "C1"), ("J3", "C1J1"), ("C1J1", "C1"), ("C1J1_full", "C1J1"), ("C1J1_full", "C1"), ("J1", "C0"), ("J2", "J1"), ("J2", "C0"), ("J3", "J2"),
            ("C0", "anchor_matched"), ("C1", "C0"), ("C1", "anchor_matched"), ("anchor_shrunk", "anchor_matched"), ("J2", "C1"), ("J1", "C1")]
E2A_PAIRS = [("kn_all", "kn_chem+ctrl", "f0_given_chem"), ("kn_all", "kn_f0+ctrl", "chem_given_f0"), ("kn_f0+ctrl", "C0", "f0_alone"), ("kn_chem+ctrl", "C0", "chem_alone"),
             ("kn_ctrl", "C0", "ctrl_alone"), ("kn_all", "C0", "all_vs_C0"), ("kn_all", "C1", "normalised_all_vs_C1"), ("kn_all_resshuf1", "kn_all", "residual_shuffle_1"),
             ("kn_all_resshuf2", "kn_all", "residual_shuffle_2"), ("kn_all_resshuf3", "kn_all", "residual_shuffle_3"), ("kn_all_f0perm", "kn_all", "f0_stratified_perm_all"),
             ("kn_f0+ctrl_f0perm", "kn_f0+ctrl", "f0_stratified_perm_f0ctrl")]
METRICS = ("delta_rmse", "centred_rmse", "centred_pearson", "mrr_tie_centred", "pred_rms", "truth_rms")


def e1(a):
    sp = yaml.safe_load(open(CFG))
    ep = load_runs(a.evaluate, "episodes.tsv.gz")
    ep["support_seed"] = ep.support_seed.astype(str)
    fold_of = ep.drop_duplicates("family").set_index("family").fold.to_dict()
    summ, con = [], []
    fams = {}
    for view, contract in SETS_E1:
        d = ep[(ep.view == view) & (ep.contract == contract)]
        if not len(d):
            continue
        for v in METRICS:
            fam = S.family_table(d, v, by=("K", "method"))
            fams[(view, contract, v)] = fam
            for (K, m), h in fam.groupby(["K", "method"]):
                summ.append(dict(view=view, contract=contract, K=K, method=m, metric=v, value=float(h[v].mean()), n_families=len(h)))
            if v not in ("delta_rmse", "centred_rmse", "centred_pearson", "mrr_tie_centred"):
                continue
            for K in sorted(fam.K.unique()):
                fk = fam[fam.K == K]
                for x, y in E1_PAIRS:
                    r = pair_stat(fk, x, y, v, fold_of)
                    if r:
                        con.append(dict(view=view, contract=contract, K=K, a=x, b=y, metric=v, family="E1", **r))
                for x, y, lab in E2A_PAIRS:
                    r = pair_stat(fk, x, y, v, fold_of)
                    if r:
                        con.append(dict(view=view, contract=contract, K=K, a=x, b=y, metric=v, family="E2A:" + lab, **r))
    # cross-contract contrasts within a view (same episodes)
    cross = [("M1", "planned", "fixed_gen_constants"), ("M2", "known_separate", "fixed_gen_constants"), ("G", "missing_reselected", "missing_zero_overlap_fallback"),
             ("M1", "missing_reselected", "missing_zero_overlap_fallback"), ("G", "planned", "missing_reselected"), ("M1", "planned", "missing_reselected"),
             ("M2", "known_separate", "missing_reselected")]
    for view, c1_, c2_ in cross:
        for v in ("delta_rmse", "centred_rmse"):
            f1, f2 = fams.get((view, c1_, v)), fams.get((view, c2_, v))
            if f1 is None or f2 is None:
                continue
            ff = pd.concat([f1.assign(method=f1.method + "@" + c1_), f2.assign(method=f2.method + "@" + c2_)])
            for K in sorted(ff.K.unique()):
                fk = ff[ff.K == K]
                for m in ("C0", "C1", "J1", "C1J1", "C1J1_full", "J2", "J3"):
                    r = pair_stat(fk, f"{m}@{c1_}", f"{m}@{c2_}", v, fold_of)
                    if r:
                        con.append(dict(view=view, contract=f"{c1_} vs {c2_}", K=K, a=m, b=m, metric=v, family="cross_contract", **r))
    summ, con = pd.DataFrame(summ), pd.DataFrame(con)
    W("e1_summary.tsv", summ)
    W("e1_contrasts.tsv", con[~con.family.str.startswith("E2A")])
    W("information_ablation.tsv", con[con.family.str.startswith("E2A")])
    # ---- H1 environment transfer (M1 planned vs M2 known_separate, paired by family) + compound-plate sensitivity
    env = []
    mk = ep[ep.view.isin(["M1", "M2"]) & (((ep.view == "M1") & (ep.contract == "planned")) | ((ep.view == "M2") & (ep.contract == "known_separate")))]
    key = ["fold", "cell", "family", "support_seed", "pert_plate_query", "K"]
    for v in ("delta_rmse", "centred_rmse"):
        pv = mk.pivot_table(index=key, columns=["view", "method"], values=v)
        q = pd.DataFrame(index=pv.index)
        for vw in ("M1", "M2"):
            q[f"common_{vw}"] = pv[(vw, "C0")] - pv[(vw, "anchor_matched")]
            q[f"perdrug_{vw}"] = pv[(vw, "C1")] - pv[(vw, "C0")]
            q[f"total_{vw}"] = pv[(vw, "C1")] - pv[(vw, "anchor_matched")]
            q[f"J3_minus_C1_{vw}"] = pv[(vw, "J3")] - pv[(vw, "C1")]
            q[f"C1J1_minus_C1_{vw}"] = pv[(vw, "C1J1")] - pv[(vw, "C1")]
        q["dc"] = q.common_M2 - q.common_M1
        q["dr"] = q.perdrug_M2 - q.perdrug_M1
        q["dc_minus_dr"] = q.dc - q.dr
        q = q.reset_index()
        for K in sorted(q.K.unique()):
            qk = q[q.K == K]
            for col in [c for c in q.columns if c not in key]:
                fam = qk.groupby(["family", "cell"])[col].mean().groupby("family").mean()
                r = diff_summary(fam, 1.0, fold_of)
                u = qk.assign(cluster=qk.pert_plate_query, d=qk[col])[["family", "cluster", "d"]].dropna()
                r.update(cluster_boot(u.groupby(["family", "cluster"]).d.mean().reset_index()))
                env.append(dict(K=K, metric=v, quantity=col, **{k: r[k] for k in ("diff", "lo", "hi", "lo90", "hi90", "p_boot", "n_families", "n_families_a_lower",
                                                                                  "fold_diffs", "n_folds_a_lower", "cluster_n", "cluster_lo", "cluster_hi", "cluster_p")}))
    env = pd.DataFrame(env)
    W("environment_transfer.tsv", env)
    # ---- primary decisions
    cK = con[(con.K == 16)]
    def get(view, contract, x, y, v="delta_rmse"):
        r = cK[(cK.view == view) & (cK.contract == contract) & (cK.a == x) & (cK.b == y) & (cK.metric == v)]
        return r.iloc[0].to_dict() if len(r) else None
    def iut(view, contract):
        r1, r2 = get(view, contract, "J3", "C1"), get(view, contract, "J3", "C1J1")
        c1, c2 = get(view, contract, "J3", "C1", "centred_rmse"), get(view, contract, "J3", "C1J1", "centred_rmse")
        if r1 is None or r2 is None:
            return None
        strong = "C1" if r1["mean_b"] <= r2["mean_b"] else "C1J1"
        rs, cs = (r1, c1) if strong == "C1" else (r2, c2)
        met = bool(r1["adoption_rule_met"] and r2["adoption_rule_met"])
        lab = "not_adopted" if not met else ("adopted_transfer" if cs["ci_below_0"] else "observation_correction_only")
        return dict(view=view, contract=contract, p_iut=max(r1["p_boot"], r2["p_boot"]), adoption_rule_met_both=met, label=lab, stronger_baseline=strong,
                    J3_minus_C1=r1, J3_minus_C1J1=r2, centred_J3_minus_C1=c1, centred_J3_minus_C1J1=c2, rel_vs_stronger=rs["rel_diff"],
                    meets_0p5pct_vs_stronger=bool(rs["meets_0p5pct"]))
    h2 = iut("M2", "known_separate")
    e16 = env[(env.K == 16) & (env.metric == "delta_rmse")].set_index("quantity")
    h1 = dict(dc_minus_dr=e16.loc["dc_minus_dr"].to_dict(), dc=e16.loc["dc"].to_dict(), dr=e16.loc["dr"].to_dict(), perdrug_M2=e16.loc["perdrug_M2"].to_dict(),
              common_M1=e16.loc["common_M1"].to_dict(), common_M2=e16.loc["common_M2"].to_dict(), perdrug_M1=e16.loc["perdrug_M1"].to_dict())
    h1["p"] = h1["dc_minus_dr"]["p_boot"]
    ps = sorted([("H1", h1["p"]), ("H2", h2["p_iut"])], key=lambda x: x[1])
    holm = {}
    for i, (n, p) in enumerate(ps):
        holm[n] = bool(p <= 0.05 / (len(ps) - i)) and all(holm.get(m, True) for m, _ in ps[:i])
    h1["holm_reject"] = holm["H1"]
    h1["supported"] = bool(holm["H1"] and h1["dc_minus_dr"]["lo"] > 0 and h1["perdrug_M2"]["hi"] < 0)
    h2["holm_reject"] = holm["H2"]
    h2["final_label"] = h2["label"] if holm["H2"] else ("not_adopted" if not h2["adoption_rule_met_both"] else h2["label"] + " (Holm not passed -> not adopted)")
    sec = {f"{v}/{c}": iut(v, c) for v, c in SETS_E1 if v != "M2" or c != "known_separate"}
    # E2-A labels (M2 known_separate, K = 16, centred RMSE)
    def lab(view, contract):
        g_ = lambda x, y: get(view, contract, x, y, "centred_rmse")
        f0c, chf, perm = g_("kn_all", "kn_chem+ctrl"), g_("kn_all", "kn_f0+ctrl"), g_("kn_all_f0perm", "kn_all")
        if f0c is None:
            return None
        return dict(view=view, contract=contract, f0_given_chem=f0c, chem_given_f0=chf, f0perm_minus_all=perm,
                    same_drug_source_response_assisted=bool(f0c["ci_below_0"] and not chf["ci_below_0"]), chemistry_adds_beyond_f0=bool(chf["ci_below_0"]),
                    f0_correspondence_used=bool(perm["ci_above_0"]))
    e2a = {f"{v}/{c}": lab(v, c) for v, c in (("M2", "known_separate"), ("G", "planned"), ("M1", "planned"))}
    m2 = e2a["M2/known_separate"]
    e2b = dict(condition_a="false: HCC has 0 queries without same-drug source response (v1 final_eval groups_without_source_family)",
               condition_b=bool(m2["same_drug_source_response_assisted"] and m2["f0_correspondence_used"]))
    e2b["enter"] = bool(e2b["condition_b"])
    # simple-vs-joint rule and the same-plate-only rule (plan 5.2 'decisions_after_E1')
    c1j1_vs_j3 = {f"{v}/{c}": get(v, c, "J3", "C1J1") for v, c in (("G", "planned"), ("M1", "planned"), ("M2", "known_separate"))}
    dec = dict(H1=h1, H2=h2, holm_order=ps, secondary_iut=sec, e2a=e2a, e2b_entry=e2b, J3_vs_C1J1=c1j1_vs_j3, spec_sha256=sha256(CFG))
    dump(dec, pub_man("e1_source_decisions.json"))
    # ---- constants and manifests
    rows = []
    for d in a.select:
        cc = json.load(open(os.path.join(d, "context_constants.json")))
        for ctxn in ("gen_planned", "gen_missing", "M1", "M2"):
            for arm, t in cc[ctxn].items():
                if isinstance(t, dict) and "16" in t:
                    for K, v in t.items():
                        rows.append(dict(partition=cc["partition"], context=ctxn, arm=arm, K=int(K), **{k: x for k, x in v.items()}))
            rows.append(dict(partition=cc["partition"], context=ctxn, arm="n_selection_episodes", value=cc[ctxn]["n_selection_episodes"], n_families=cc[ctxn]["n_selection_families"]))
    W("e1_selected_constants.tsv", pd.DataFrame(rows))
    tl = load_runs(a.select, "tie_log.tsv")
    W("e1_selection_tie_log.tsv", tl)
    man = load_runs(a.evaluate, "episode_manifest.tsv")
    W("e1_episode_manifest.tsv", man)
    pm = load_runs(a.evaluate, "row_permutation_check.tsv")
    mx = pm[[c for c in pm.columns if c.startswith("maxabs_")]].max()
    mm = man[man.set == "matched"]
    elig = dict(n_general_episodes=int((man.set == "general").sum()), n_matched_episodes=int(len(mm)), n_matched_families=int(mm.family.nunique()),
                n_matched_cells=int(mm.cell.nunique()), pair_kinds=mm.drop_duplicates(["cell", "plate_query", "plate_other"]).pair_kind.value_counts().to_dict(),
                n_compound_plates=int(mm.pert_plate_query.nunique()), compound_plates_shared_by_families=int((mm.groupby("pert_plate_query").family.nunique() > 1).sum()),
                same_support_groups_E1_E2=bool(mm.same_support_groups_E1_E2.all()), same_truth_E1_E2=bool(mm.same_truth_E1_E2.all()),
                full_row_permutation_max_abs=float(mx.max()) if len(mx) else None,
                different_compound_plate_or_date_stratum="none: every eligible pair is an X-tag replicate of one compound plate" if set(mm.pair_kind) == {"x_tag_same_compound_plate"} else "present")
    dump(dict(eligibility=elig, evaluate=a.evaluate, select=a.select, results={d: json.load(open(os.path.join(d, "result.json"))) for d in a.evaluate + a.select}),
         pub_man("e1_report_inputs.json"))
    log(json.dumps(dict(H1=dict(supported=h1["supported"], p=h1["p"]), H2=dict(label=h2["final_label"], p=h2["p_iut"]), e2b=e2b, elig=elig), indent=1, default=str))


# ================================================================= HCC follow-up
def hcc(a):
    fz = json.load(open(os.path.join(MAN, "e1_source_freeze.json")))
    res = json.load(open(os.path.join(a.run, "result.json")))
    assert sha256(res["hcc_per_unit"]) == res["hcc_per_unit_sha256"]
    u = pd.read_csv(res["hcc_per_unit"], sep="\t")
    pairs = [tuple(x) for x in fz["hcc_contrasts"]]
    tabs, cons = [], []
    for qset in ("test", "test_plus_val"):
        t, c, n = hcc_tables(u, pairs, qset)
        tabs.append(t)
        cons.append(c)
    W("hcc_e1_results.tsv", pd.concat(tabs, ignore_index=True))
    W("hcc_e1_contrasts.tsv", pd.concat(cons, ignore_index=True))
    dump(dict(run=a.run, pairs=pairs, n_union_query_drugs_test=int(u[u.query_set == "test"].drug_group.nunique())), pub_man("hcc_e1_report_inputs.json"))
    log("hcc done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["p0", "e1", "hcc"])
    ap.add_argument("--p0-select", nargs="*", default=[])
    ap.add_argument("--p0-anchor", nargs="*", default=[])
    ap.add_argument("--p0-k8", default="")
    ap.add_argument("--select", nargs="*", default=[])
    ap.add_argument("--evaluate", nargs="*", default=[])
    ap.add_argument("--run", default="")
    a = ap.parse_args()
    {"p0": p0, "e1": e1, "hcc": hcc}[a.stage](a)
