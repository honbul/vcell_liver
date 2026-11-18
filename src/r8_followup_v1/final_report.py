"""Tables of the post-freeze evaluation (no prediction is recomputed).

confirmation : family-equal means and paired family-bootstrap contrasts (9 families), plus matched-plate E1 / E2 (5 families)
hcc          : per target = mean over split seeds of the query-drug mean; EQUAL_MEAN_3 = equal mean of the three targets.
               Intervals: synchronised drug-ID bootstrap (one multinomial weight per drug ID over the union of all targets' and
               splits' query drugs, the equal mean computed inside every draw). Per-target intervals use independent SeedSequence
               streams over that target's own drug union (conditional on the target).
reproduction : fx_* rows (original constants, original V0, original feature basis) next to the original R8 tables.
"""
import argparse, json, os
import numpy as np
import pandas as pd

from .common import pub_man, orig_man, dump, sha256, BOOT_SEED, N_BOOT, TARGETS
from . import stats as S
from .report import as_sets, contrasts, summary

METRICS = ["delta_rmse", "centred_rmse", "centred_pearson", "delta_pearson", "mrr_tie_centred", "mrr_legacy_centred"]


def hcc_tables(u, pairs, qset="test"):
    u = u[u.query_set == qset]
    keys = sorted({(t, s) for t, s in zip(u.target, u.split_seed)})
    drugs = {k: sorted(u[(u.target == k[0]) & (u.split_seed == k[1])].drug_group.unique()) for k in keys}
    Wsync, n_union = S.sync_drug_weights({k: drugs[k] for k in keys})
    streams = S.independent_streams(list(TARGETS))
    Wind = {}
    for t in TARGETS:
        tk = [k for k in keys if k[0] == t]
        union = sorted(set().union(*[set(drugs[k]) for k in tk]))
        pos = {g: i for i, g in enumerate(union)}
        W = streams[t].multinomial(len(union), np.ones(len(union)) / len(union), size=N_BOOT).astype(np.float64)
        for k in tk:
            Wind[k] = W[:, [pos[g] for g in drugs[k]]]
    piv = {}
    for v in METRICS:
        p = u.pivot_table(index=["target", "split_seed", "K", "drug_group"], columns="method", values=v, aggfunc="mean")
        piv[v] = p
    res, con = [], []
    for K in sorted(u.K.unique()):
        methods = sorted(u[u.K == K].method.unique())
        for v in METRICS:
            P = piv[v].xs(K, level="K")

            def stat(vec_of_key):
                pt_t, sy_t, ind_t = {}, {}, {}
                for t in TARGETS:
                    tk = [k for k in keys if k[0] == t]
                    pts, sy, ind = [], [], []
                    for k in tk:
                        x = vec_of_key(P.loc[(k[0], k[1])].reindex(drugs[k]))
                        ok = np.isfinite(x)
                        pts.append(x[ok].mean())
                        sy.append(S.weighted_means(x, Wsync[k]))
                        ind.append(S.weighted_means(x, Wind[k]))
                    pt_t[t], sy_t[t], ind_t[t] = float(np.mean(pts)), np.nanmean(np.stack(sy), 0), np.nanmean(np.stack(ind), 0)
                eq_pt = float(np.mean([pt_t[t] for t in TARGETS]))
                eq_dr = np.mean([sy_t[t] for t in TARGETS], 0)
                return pt_t, sy_t, ind_t, eq_pt, eq_dr
            for m in methods:
                pt_t, sy_t, ind_t, eq_pt, eq_dr = stat(lambda df, m=m: df[m].to_numpy(float))
                for t in TARGETS:
                    lo, hi = np.nanpercentile(ind_t[t], [2.5, 97.5])
                    res.append(dict(target=t, query_set=qset, K=K, method=m, metric=v, value=pt_t[t], lo=lo, hi=hi, ci="per-target conditional (independent stream)"))
                lo, hi = np.nanpercentile(eq_dr, [2.5, 97.5])
                res.append(dict(target="EQUAL_MEAN_3", query_set=qset, K=K, method=m, metric=v, value=eq_pt, lo=lo, hi=hi, ci="synchronised drug-ID bootstrap"))
            for a_, b_ in pairs:
                if a_ not in methods or b_ not in methods:
                    continue
                pt_t, sy_t, ind_t, eq_pt, eq_dr = stat(lambda df, a_=a_, b_=b_: df[a_].to_numpy(float) - df[b_].to_numpy(float))
                ref = float(np.mean([P[b_].groupby(level=[0, 1]).mean().loc[t].mean() for t in TARGETS]))
                for t in TARGETS:
                    lo, hi = np.nanpercentile(ind_t[t], [2.5, 97.5])
                    con.append(dict(target=t, query_set=qset, K=K, contrast=f"{a_}-{b_}", metric=v, diff=pt_t[t], lo=lo, hi=hi, ci="per-target conditional (independent stream)"))
                lo, hi = np.nanpercentile(eq_dr, [2.5, 97.5])
                lo9, hi9 = np.nanpercentile(eq_dr, [5, 95])
                tol = S.REL * ref if v in ("delta_rmse", "centred_rmse") else np.nan
                con.append(dict(target="EQUAL_MEAN_3", query_set=qset, K=K, contrast=f"{a_}-{b_}", metric=v, diff=eq_pt, lo=lo, hi=hi, lo90=lo9, hi90=hi9,
                                rel_diff=eq_pt / ref if v in ("delta_rmse", "centred_rmse") else np.nan, ref_mean=ref,
                                equivalent_0p5pct=bool(lo9 > -tol and hi9 < tol) if np.isfinite(tol) else np.nan,
                                n_targets_a_lower=int(sum(pt_t[t] < 0 for t in TARGETS)), ci="synchronised drug-ID bootstrap"))
    return pd.DataFrame(res), pd.DataFrame(con), n_union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    fz = json.load(open(os.path.join("manifests", "r8_followup_v1", "followup_freeze.json")))
    cand, base = fz["candidate"]["method"], fz["baselines"]
    pairs = [(cand, b) for b in base if b != cand] + [tuple(x) for x in fz.get("extra_contrasts", [])]
    out = {}
    # ---- confirmation families
    ce = os.path.join(a.run, "confirmation_episodes.tsv.gz")
    if os.path.exists(ce):
        g = pd.read_csv(ce, sep="\t").assign(fold="global")
        me = os.path.join(a.run, "matched_episodes.tsv.gz")
        mat = pd.read_csv(me, sep="\t") if os.path.exists(me) else pd.DataFrame(columns=g.columns)
        df = as_sets(g, mat).replace({"set": {"source_oof": "confirmation"}})
        sets = ("confirmation", "E1", "E2")
        allm = sorted(set([cand] + base + [x for p in pairs for x in p] + ["fx_C0_iid", "fx_C1_iid", "fx_C0_V0", "fx_C1_V0"]))
        sm = summary(df, allm, sets=sets)
        lp = [(x, y, f"{x}-{y}") for x, y in pairs]
        cs = pd.concat([contrasts(df, lp, sets=sets), contrasts(df, lp, value="centred_rmse", sets=sets)], ignore_index=True)
        sm.to_csv(pub_man("confirmation_followup_results.tsv"), sep="\t", index=False, float_format="%.6g")
        cs.to_csv(pub_man("confirmation_followup_contrasts.tsv"), sep="\t", index=False, float_format="%.6g")
        o = pd.read_csv(orig_man("source_eval_summary.tsv"), sep="\t")
        rep = []
        for fxm, om in (("fx_C0_iid", "C0_iid"), ("fx_C1_iid", "C1_iid"), ("fx_C0_V0", "C0_cov"), ("fx_C1_V0", "C1_cov"), ("shrunk_mean", "shrunk_same_drug_mean"),
                        ("no_change", "no_change"), ("same_drug_mean", "same_drug_mean")):
            for K in (8, 16, 32):
                r = sm[(sm.set == "confirmation") & (sm.method == fxm) & (sm.K == K)]
                q = o[(o.method == om) & (o.K == K)]
                if len(r) and len(q):
                    rep.append(dict(panel="confirmation", K=K, followup=fxm, original=om, followup_value=float(r.delta_rmse.iloc[0]), original_value=float(q.delta_rmse.iloc[0])))
        out["confirmation_reproduction"] = rep
    # ---- HCC
    res = json.load(open(os.path.join(a.run, "result.json")))
    if "hcc_per_unit" in res:
        assert sha256(res["hcc_per_unit"]) == res["hcc_per_unit_sha256"]
        u = pd.read_csv(res["hcc_per_unit"], sep="\t")
        tabs, cons = [], []
        for qset in ("test", "test_plus_val"):
            t, c, n_union = hcc_tables(u, pairs, qset)
            tabs.append(t)
            cons.append(c)
        tab, con = pd.concat(tabs, ignore_index=True), pd.concat(cons, ignore_index=True)
        tab.to_csv(pub_man("hcc_followup_results.tsv"), sep="\t", index=False, float_format="%.6g")
        con.to_csv(pub_man("hcc_followup_contrasts.tsv"), sep="\t", index=False, float_format="%.6g")
        o = pd.read_csv(orig_man("hcc_results.tsv"), sep="\t")
        rep = []
        for fxm, om in (("fx_C0_iid", "C0_iid"), ("fx_C1_iid", "C1_iid"), ("fx_C0_V0", "C0_cov"), ("fx_C1_V0", "C1_cov"), ("shrunk_mean", "shrunk_same_drug_mean"),
                        ("no_change", "no_change"), ("same_drug_mean", "same_drug_mean")):
            for t in list(TARGETS) + ["EQUAL_MEAN_3"]:
                for K in (8, 16, 32):
                    r = tab[(tab.target == t) & (tab.method == fxm) & (tab.K == K) & (tab.metric == "delta_rmse") & (tab.query_set == "test")]
                    q = o[(o.target == t) & (o.method == om) & (o.K == K) & (o.query_set == "test")]
                    if len(r) and len(q):
                        rep.append(dict(panel="hcc", target=t, K=K, followup=fxm, original=om, followup_value=float(r["value"].iloc[0]), original_value=float(q.delta_rmse.iloc[0])))
        out["hcc_reproduction"] = rep
        out["hcc_union_query_drugs_test"] = int(u[u.query_set == "test"].drug_group.nunique())
        # cost of the support (original support_cost.tsv; identical support lists)
        out["support_cost_source"] = "manifests/replicate_calibrated_r8/support_cost.tsv (identical frozen support lists)"
    rp = pd.DataFrame(out.get("confirmation_reproduction", []) + out.get("hcc_reproduction", []))
    if len(rp):
        rp["abs_diff"] = (rp.followup_value - rp.original_value).abs()
        rp.to_csv(pub_man("v0_reproduction_existing_panels.tsv"), sep="\t", index=False, float_format="%.7g")
    dump(dict(run=a.run, candidate=cand, baselines=base, pairs=pairs, **{k: v for k, v in out.items() if not k.endswith("reproduction")}), pub_man("final_report_inputs.json"))
    print(rp.to_string() if len(rp) else "no reproduction rows")


if __name__ == "__main__":
    main()
