"""R10-E1a / E2 (v2): train x evaluation contract matrix and declared/actual assembly audit.

Reads immutable evaluation outputs (evaluate.py v2 rows carry train_contract / assembly_version /
atlas_version) and writes small tables. Contrasts, thresholds and unlock rules come from the frozen
spec configs/r10/contract_matrix_v2.json; nothing here chooses a contrast after seeing results.

Aggregation: seeds -> triplet -> replicate direction -> line (equal weight) -> drug; drug bootstrap.
Pairing: inside one (role, episode, evaluation contract) cell, on the triplet intersection where both
arms are finite.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from . import aggregate as ag

TRIP = ["role", "line", "drug", "episode", "direction", "rep_endpoint", "rep_query"]
METRICS = ["comp_tv", "within_mean_rmse", "within_ed", "ed", "mmd", "dmean_rmse_gene"]
E2COLS = ["head_comp_tv", "assembly_tv_head_vs_reread", "diag_membership_entropy", "diag_neff_low_min",
          "diag_neff_high_min", "diag_lowsupport_mass", "diag_pi_shift_tv", "diag_q_shift_rms"]
TCS = ("same", "cross")
B, SEED = 2000, 20261117


# ------------------------------------------------------------------ loading
def load(patterns, triplets):
    fs = sorted(f for p in patterns for f in glob.glob(p))
    df = ag.arm_label(pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True))
    df = df[df.is_oracle == 0]
    df["arm"] = df.arm.replace({"mixture": "D00"}).str.replace("mixture_t[", "tstar[", regex=False)
    dup = df.duplicated(TRIP + ["arm", "seed"]).sum()
    assert dup == 0, f"{dup} duplicated triplet x arm x seed rows"
    for c in E2COLS:
        if c not in df:
            df[c] = np.nan
    tr = pd.read_csv(triplets, sep="\t")[TRIP[1:] + ["plate_low", "plate_high", "plate_query"]].drop_duplicates()
    rel = np.select([tr.plate_query == tr.plate_low, tr.plate_query == tr.plate_high],
                    ["query_on_low_plate", "query_on_high_plate"], "query_on_neither")
    tr = tr.assign(plate_relation=rel)[TRIP[1:] + ["plate_relation"]]
    df = df.merge(tr, on=TRIP[1:], how="left", validate="many_to_one")
    return df, fs


def triplet_level(df):
    cols = [c for c in METRICS + E2COLS if c in df]
    return df.groupby(TRIP + ["plate_relation", "arm"], as_index=False)[cols].mean()   # seeds


# ------------------------------------------------------------------ statistics
def boot(x, B=B, seed=SEED, n_family=6):
    """Drug bootstrap of a mean: estimate, 95/90% and Bonferroni (1-0.05/n_family) percentile
    intervals and a two-sided percentile p-value."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, lo90=np.nan, hi90=np.nan, lo_bonf=np.nan, hi_bonf=np.nan,
                    p_boot=np.nan, n_drugs=0)
    rng = np.random.default_rng(seed)
    bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    a = 0.05 / n_family
    q = lambda p: float(np.quantile(bs, p))
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return dict(est=float(x.mean()), lo=q(0.025), hi=q(0.975), lo90=q(0.05), hi90=q(0.95), lo_bonf=q(a / 2),
                hi_bonf=q(1 - a / 2), p_boot=float(p), n_drugs=n)


def to_drug(t, value="d"):
    """triplet rows -> mean over replicate directions per (line, drug) -> mean over lines per drug."""
    return t.groupby(["line", "drug"])[value].mean().groupby("drug").mean()


def paired(T, role, ep, ev, a, b, metric, extra=None):
    s = T[(T.role == role) & (T.episode == ep) & (T.direction == ev) & T.arm.isin([a, b])]
    if extra is not None:
        s = s[extra(s)]
    w = s.pivot_table(index=["line", "drug", "rep_endpoint", "rep_query"], columns="arm", values=metric)
    if a not in w or b not in w:
        return None
    d = (w[a] - w[b]).dropna().rename("d").reset_index()
    if len(d) == 0:
        return None
    x = to_drug(d)
    r = boot(x.values)
    r.update(n_triplets=len(d), frac_drugs_a_lower=float((x < 0).mean()))
    return r


def arm_summary(T, role, ep, ev, arm, metric, common=None):
    s = T[(T.role == role) & (T.episode == ep) & (T.direction == ev) & (T.arm == arm)].dropna(subset=[metric])
    if common is not None:
        s = s.merge(common, on=["line", "drug", "rep_endpoint", "rep_query"])
    if len(s) == 0:
        return None
    x = to_drug(s, metric)
    r = boot(x.values)
    r.update(n_triplets=len(s))
    return r


# ------------------------------------------------------------------ E1a
def secondary_contrasts(arms):
    out = []
    for tc in TCS:
        oth = "cross" if tc == "same" else "same"
        for X in ("D10", "D01", "D11", "W", "tstar"):
            out.append((f"{X}[{tc}]", "D00", f"{X}-D00"))
        out += [(f"D10[{tc}]", f"tstar[{tc}]", "D10-tstar_matched"), (f"D10[{tc}]", f"tstar[{oth}]", "D10-tstar_other"),
                (f"D11[{tc}]", f"D10[{tc}]", "D11-D10"), (f"D11[{tc}]", f"W[{tc}]", "D11-W"),
                (f"D11[{tc}]", f"tstar[{tc}]", "D11-tstar_matched"), (f"W[{tc}]", f"tstar[{tc}]", "W-tstar_matched")]
    for X in ("D10", "D01", "D11", "W", "tstar"):
        out.append((f"{X}[same]", f"{X}[cross]", "train_contract_effect"))
    return [c for c in out if c[0] in arms and c[1] in arms]


def label(r, thr, margin):
    if r["p_holm"] < 0.05:
        return "practical_difference_established" if abs(r["est"]) >= thr else "difference_below_practical_threshold"
    if r["lo90"] > -margin and r["hi90"] < margin:
        return "equivalent_within_margin"
    return "not_established"


def e1a(T, spec, out):
    arms = set(T.arm)
    pr = spec["primary"]
    thr, margin = pr["practical_threshold"]["comp_tv"], pr["equivalence"]["margin"]["comp_tv"]
    # primary family
    rows = []
    for c in pr["family"]:
        r = paired(T, pr["role"], pr["episode"], c["eval"], c["a"], c["b"], pr["metric"])
        rows.append(dict(id=c["id"], question=c["question"], a=c["a"], b=c["b"], eval_contract=c["eval"],
                         role=pr["role"], episode=pr["episode"], metric=pr["metric"], **(r or {})))
    P = pd.DataFrame(rows)
    order = np.argsort(P.p_boot.values)
    m = len(P)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * P.p_boot.values[i]))
        adj[i] = run
    P["p_holm"] = adj
    P["practical_threshold"] = thr
    P["decision"] = [label(r, thr, margin) for r in P.to_dict("records")]
    P.to_csv(f"{out}/primary_contrasts.tsv", sep="\t", index=False, float_format="%.6g")
    # secondary: all cells, both roles/episodes, all metrics
    rows = []
    for role in sorted(T.role.unique()):
        for ep in ("E1000", "E100"):
            for ev in TCS:
                for a, b, kind in secondary_contrasts(arms):
                    for met in METRICS:
                        r = paired(T, role, ep, ev, a, b, met)
                        if r is None:
                            continue
                        eq = (met == "comp_tv") and (r["lo90"] > -margin and r["hi90"] < margin)
                        rows.append(dict(role=role, episode=ep, eval_contract=ev, kind=kind, a=a, b=b, metric=met,
                                         **r, ci95_excludes_0=bool(r["lo"] > 0 or r["hi"] < 0),
                                         equivalent_within_margin=eq if met == "comp_tv" else np.nan))
                # arm interaction per training contract
                for tc in TCS:
                    s = T[(T.role == role) & (T.episode == ep) & (T.direction == ev)]
                    need = ["D00", f"D10[{tc}]", f"D01[{tc}]", f"D11[{tc}]"]
                    for met in METRICS:
                        w = s[s.arm.isin(need)].pivot_table(index=["line", "drug", "rep_endpoint", "rep_query"],
                                                            columns="arm", values=met)
                        if not all(x in w for x in need):
                            continue
                        d = (w[need[3]] - w[need[1]] - w[need[2]] + w["D00"]).dropna().rename("d").reset_index()
                        r = boot(to_drug(d).values)
                        rows.append(dict(role=role, episode=ep, eval_contract=ev, kind="arm_interaction",
                                         a=f"D11-D10-D01+D00[{tc}]", b="", metric=met, **r, n_triplets=len(d),
                                         ci95_excludes_0=bool(r["lo"] > 0 or r["hi"] < 0)))
    S = pd.DataFrame(rows)
    S.to_csv(f"{out}/secondary_contrasts.tsv", sep="\t", index=False, float_format="%.6g")
    # cell summaries (own triplets and common triplets)
    rows = []
    for role in sorted(T.role.unique()):
        for ep in ("E1000", "E100"):
            for ev in TCS:
                s = T[(T.role == role) & (T.episode == ep) & (T.direction == ev)]
                w = s.pivot_table(index=["line", "drug", "rep_endpoint", "rep_query"], columns="arm", values="comp_tv")
                common = w.dropna().reset_index()[["line", "drug", "rep_endpoint", "rep_query"]]
                for arm in sorted(s.arm.unique()):
                    for met in METRICS:
                        for setname, cm in (("own", None), ("common_all_arms", common)):
                            r = arm_summary(T, role, ep, ev, arm, met, cm)
                            if r:
                                rows.append(dict(role=role, episode=ep, eval_contract=ev, arm=arm, metric=met,
                                                 triplet_set=setname, mean=r["est"], lo=r["lo"], hi=r["hi"],
                                                 n_drugs=r["n_drugs"], n_triplets=r["n_triplets"]))
    C = pd.DataFrame(rows)
    C.to_csv(f"{out}/cell_summary.tsv", sep="\t", index=False, float_format="%.6g")
    matrix_2x2(C, out)
    contract_interaction(T, out)
    strata(T, spec, out)
    return P, S, C


def matrix_2x2(C, out):
    """Readable 2x2: rows = training contract of the learned arm / t*, columns = evaluation contract."""
    c = C[(C.metric == "comp_tv") & (C.triplet_set == "common_all_arms")]
    rows = []
    for (role, ep), s in c.groupby(["role", "episode"]):
        for tc in ("same", "cross", "none"):
            for base in ("D00", "tstar", "D10", "D01", "D11", "W"):
                arm = base if base == "D00" else f"{base}[{tc}]"
                if (base == "D00") != (tc == "none"):
                    continue
                r = dict(role=role, episode=ep, train_contract=tc, arm=arm)
                for ev in TCS:
                    x = s[(s.arm == arm) & (s.eval_contract == ev)]
                    if len(x):
                        x = x.iloc[0]
                        r[f"eval_{ev}"] = f"{x['mean']:.4f} [{x.lo:.4f}, {x.hi:.4f}]"
                        r[f"n_drugs_{ev}"] = int(x.n_drugs)
                rows.append(r)
    pd.DataFrame(rows).to_csv(f"{out}/matrix_2x2_comp_tv.tsv", sep="\t", index=False)


def contract_interaction(T, out):
    """Across evaluation contracts: drug-paired on the common eligible (line, drug, episode) set."""
    rows = []
    for role in sorted(T.role.unique()):
        for ep in ("E1000", "E100"):
            s = T[(T.role == role) & (T.episode == ep)]
            for met in METRICS:
                g = s.groupby(["line", "drug", "direction", "arm"])[met].mean().unstack(["direction", "arm"])
                for X in ("D10", "D01", "D11", "W", "tstar"):
                    need = [(ev, f"{X}[{tc}]") for ev in TCS for tc in TCS]
                    if not all(n in g.columns for n in need):
                        continue
                    h = g[need].dropna()
                    ss, sc = h[("same", f"{X}[same]")], h[("same", f"{X}[cross]")]
                    cs, cc = h[("cross", f"{X}[same]")], h[("cross", f"{X}[cross]")]
                    for name, v in (("contract_interaction", (ss - sc) - (cs - cc)),
                                    ("train_effect_at_eval_same", ss - sc), ("train_effect_at_eval_cross", cs - cc),
                                    (f"eval_cross_minus_same[{X}[same]]", cs - ss),
                                    (f"eval_cross_minus_same[{X}[cross]]", cc - sc)):
                        x = v.groupby("drug").mean()
                        r = boot(x.values)
                        rows.append(dict(role=role, episode=ep, metric=met, arm=X, contrast=name,
                                         n_line_drug=len(h), **r))
                if all(n in g.columns for n in [("same", "D00"), ("cross", "D00")]):
                    h = g[[("same", "D00"), ("cross", "D00")]].dropna()
                    x = (h[("cross", "D00")] - h[("same", "D00")]).groupby("drug").mean()
                    rows.append(dict(role=role, episode=ep, metric=met, arm="D00",
                                     contrast="eval_cross_minus_same[D00]", n_line_drug=len(h), **boot(x.values)))
    pd.DataFrame(rows).to_csv(f"{out}/contract_interaction.tsv", sep="\t", index=False, float_format="%.6g")


def strata(T, spec, out):
    pr = spec["primary"]
    rows = []
    for c in pr["family"]:
        for role in sorted(T.role.unique()):
            for ep in ("E1000", "E100"):
                defs = [("line", l, (lambda l: lambda s: s.line == l)(l)) for l in sorted(T.line.unique())]
                defs += [("replicate_direction", f"{a}->{b}", (lambda a, b: lambda s: (s.rep_endpoint == a)
                                                                & (s.rep_query == b))(a, b))
                         for a, b in sorted(set(zip(T.rep_endpoint, T.rep_query)))]
                defs += [("plate_relation", p, (lambda p: lambda s: s.plate_relation == p)(p))
                         for p in sorted(T.plate_relation.dropna().unique())]
                for kind, val, f in defs:
                    r = paired(T, role, ep, c["eval"], c["a"], c["b"], "comp_tv", f)
                    if r:
                        rows.append(dict(id=c["id"], a=c["a"], b=c["b"], eval_contract=c["eval"], role=role,
                                         episode=ep, stratum=kind, value=val, **r))
    pd.DataFrame(rows).to_csv(f"{out}/strata_primary_contrasts.tsv", sep="\t", index=False, float_format="%.6g")
    pc = (T[T.arm == "D00"].groupby(["role", "episode", "direction", "plate_relation"]).size()
          .rename("n_triplets").reset_index())
    pc.to_csv(f"{out}/plate_relation_counts.tsv", sep="\t", index=False)


# ------------------------------------------------------------------ E2
def e2(df, T, spec, out):
    dec = T[T.head_comp_tv.notna()].copy()
    dec["gap_actual_minus_declared"] = dec.comp_tv - dec.head_comp_tv
    tol = spec["e2_audit"]["d00_tolerance"]
    d00 = dec[dec.arm == "D00"]
    tol_check = dict(n=len(d00), max_assembly_tv=float(d00.assembly_tv_head_vs_reread.abs().max()),
                     max_abs_declared_minus_actual=float((d00.head_comp_tv - d00.comp_tv).abs().max()))
    tol_check["pass"] = bool(tol_check["max_assembly_tv"] <= tol and tol_check["max_abs_declared_minus_actual"] <= tol)
    rows = []
    for (role, ep, ev, arm), s in dec.groupby(["role", "episode", "direction", "arm"]):
        r = dict(role=role, episode=ep, eval_contract=ev, arm=arm, n_triplets=len(s))
        for v, name in (("head_comp_tv", "tv_declared_query"), ("comp_tv", "tv_actual_query"),
                        ("assembly_tv_head_vs_reread", "tv_declared_actual"),
                        ("gap_actual_minus_declared", "gap_actual_minus_declared")):
            b = boot(to_drug(s, v).values)
            r.update({f"{name}": b["est"], f"{name}_lo": b["lo"], f"{name}_hi": b["hi"]})
            r["n_drugs"] = b["n_drugs"]
        q = s.assembly_tv_head_vs_reread
        r.update(tv_declared_actual_p50=q.median(), tv_declared_actual_p90=q.quantile(0.9),
                 tv_declared_actual_max=q.max(),
                 spearman_triplet_mismatch_vs_gap=s[["assembly_tv_head_vs_reread", "gap_actual_minus_declared"]]
                 .corr("spearman").iloc[0, 1] if len(s) > 2 else np.nan)
        rows.append(r)
    DA = pd.DataFrame(rows)
    DA.to_csv(f"{out}/declared_vs_actual.tsv", sep="\t", index=False, float_format="%.6g")
    # selection disagreement: declared-TV vs actual-TV contrasts and arm ranking
    dec2 = T.copy()
    fixed = dec2.arm.isin(["D00"]) | dec2.arm.str.startswith("tstar[")
    dec2.loc[fixed, "head_comp_tv"] = dec2.loc[fixed, "comp_tv"]          # fixed memberships: declared = actual
    rows = []
    for role in sorted(T.role.unique()):
        for ep in ("E1000", "E100"):
            for ev in TCS:
                for tc in TCS:
                    pairs = [(f"D10[{tc}]", "D00"), (f"D11[{tc}]", "D00"), (f"D01[{tc}]", "D00"),
                             (f"D10[{tc}]", f"tstar[{tc}]"), (f"D11[{tc}]", f"tstar[{tc}]"),
                             (f"D11[{tc}]", f"D10[{tc}]")]
                    for a, b in pairs:
                        ra = paired(dec2, role, ep, ev, a, b, "head_comp_tv")
                        rb = paired(dec2, role, ep, ev, a, b, "comp_tv")
                        if ra is None or rb is None:
                            continue
                        rows.append(dict(role=role, episode=ep, eval_contract=ev, train_contract=tc, a=a, b=b,
                                         declared_est=ra["est"], declared_lo=ra["lo"], declared_hi=ra["hi"],
                                         actual_est=rb["est"], actual_lo=rb["lo"], actual_hi=rb["hi"],
                                         n_drugs=rb["n_drugs"],
                                         sign_change=bool(np.sign(ra["est"]) != np.sign(rb["est"])),
                                         any_ci_excludes_0=bool(ra["lo"] > 0 or ra["hi"] < 0 or rb["lo"] > 0 or rb["hi"] < 0)))
                # arm ranking within the cell (drug-level means on the common set of declared-capable arms)
                s = dec2[(dec2.role == role) & (dec2.episode == ep) & (dec2.direction == ev) & dec2.head_comp_tv.notna()]
                for v in ("head_comp_tv", "comp_tv"):
                    w = s.pivot_table(index=["line", "drug", "rep_endpoint", "rep_query"], columns="arm", values=v).dropna()
                    mean = w.reset_index().groupby(["line", "drug"]).mean(numeric_only=True).groupby("drug").mean().mean()
                    for rank, (arm, val) in enumerate(mean.sort_values().items(), 1):
                        rows.append(dict(role=role, episode=ep, eval_contract=ev, train_contract="",
                                         a=arm, b=f"rank_by_{'declared' if v == 'head_comp_tv' else 'actual'}",
                                         declared_est=val if v == "head_comp_tv" else np.nan,
                                         actual_est=val if v == "comp_tv" else np.nan, rank=rank))
    SD = pd.DataFrame(rows)
    # seed-level ranking (no selection among seeds is made; descriptive only)
    sr = []
    sd = df[df.head_comp_tv.notna() & df.arm.str.match(r"D1[01]\[")]
    for (role, ep, ev, arm, seed), s in sd.groupby(["role", "episode", "direction", "arm", "seed"]):
        sr.append(dict(role=role, episode=ep, eval_contract=ev, train_contract="", a=arm, b=f"seed{seed}",
                       declared_est=to_drug(s, "head_comp_tv").mean(), actual_est=to_drug(s, "comp_tv").mean()))
    SD = pd.concat([SD, pd.DataFrame(sr)], ignore_index=True)
    SD.to_csv(f"{out}/selection_disagreement.tsv", sep="\t", index=False, float_format="%.6g")
    # endpoint support: diagnostics of the endpoint (from D00 rows) vs mismatch of each learned arm
    diag = T[T.arm == "D00"][TRIP + ["diag_membership_entropy", "diag_neff_low_min", "diag_neff_high_min",
                                     "diag_lowsupport_mass"]]
    L = dec[dec.arm != "D00"].drop(columns=["diag_membership_entropy", "diag_neff_low_min", "diag_neff_high_min",
                                            "diag_lowsupport_mass"]).merge(diag, on=TRIP)
    L["diag_neff_min"] = L[["diag_neff_low_min", "diag_neff_high_min"]].min(1)
    rows = []
    for (role, ep), s0 in L.groupby(["role", "episode"]):
        base = diag[(diag.role == role) & (diag.episode == ep)].copy()
        base["diag_neff_min"] = base[["diag_neff_low_min", "diag_neff_high_min"]].min(1)
        for dv in ("diag_membership_entropy", "diag_neff_min", "diag_lowsupport_mass", "diag_q_shift_rms",
                   "diag_pi_shift_tv"):
            src = base if dv in base else s0
            cuts = np.unique(np.quantile(src[dv].dropna(), [1 / 3, 2 / 3]))
            for (ev, arm), s in s0.groupby(["direction", "arm"]):
                bins = np.digitize(s[dv], cuts)
                for bi in np.unique(bins):
                    ss = s[bins == bi]
                    rows.append(dict(role=role, episode=ep, eval_contract=ev, arm=arm, diagnostic=dv,
                                     tertile=int(bi) + 1, n_triplets=len(ss),
                                     n_drugs=ss.drug.nunique(), diag_mean=ss[dv].mean(),
                                     tv_declared_actual=ss.assembly_tv_head_vs_reread.mean(),
                                     gap_actual_minus_declared=ss.gap_actual_minus_declared.mean(),
                                     tv_actual_query=ss.comp_tv.mean()))
                rows.append(dict(role=role, episode=ep, eval_contract=ev, arm=arm, diagnostic=dv, tertile=0,
                                 n_triplets=len(s), spearman_vs_mismatch=s[[dv, "assembly_tv_head_vs_reread"]]
                                 .corr("spearman").iloc[0, 1]))
    pd.DataFrame(rows).to_csv(f"{out}/endpoint_support.tsv", sep="\t", index=False, float_format="%.6g")
    return DA, SD, tol_check


def unlocks(DA, SD, spec, T):
    u = spec["conditional_unlocks"]
    role, ep = "development", "E1000"
    u1 = {}
    for tc in TCS:
        s = T[(T.role == role) & (T.episode == ep) & (T.direction == tc) & (T.arm == f"D10[{tc}]")].copy()
        s["g"] = s.comp_tv - s.head_comp_tv
        r = boot(to_drug(s, "g").values) if len(s) else None
        u1[tc] = dict(r or {}, met=bool(r and r["est"] >= 0.005 and r["lo"] > 0))
    sd = SD[(SD.role == role) & (SD.episode == ep) & (SD.eval_contract == SD.train_contract)
            & SD.b.isin(["D00"] + [f"tstar[{tc}]" for tc in TCS]) & SD.a.str.match(r"D1[01]\[")]
    u2_rows = sd[sd.sign_change & sd.any_ci_excludes_0]
    res = dict(rule=u, U1=u1, U1_met=all(v["met"] for v in u1.values()),
               U2_pairs=sd[["eval_contract", "a", "b", "declared_est", "declared_lo", "declared_hi", "actual_est",
                            "actual_lo", "actual_hi", "sign_change", "any_ci_excludes_0"]].to_dict("records"),
               U2_met=bool(len(u2_rows) > 0), epoch_ranking="not assessable from v1 artifacts")
    res["unlock_7_2_3"] = bool(res["U1_met"] or res["U2_met"])
    res["unlock_7_2_4"] = res["unlock_7_2_3"]
    res["unlock_7_1_5"] = res["unlock_7_2_3"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert not os.path.exists(f"{args.out}/primary_contrasts.tsv"), "immutable report run exists"
    os.makedirs(args.out, exist_ok=True)
    spec = json.load(open(args.spec))
    df, fs = load(args.evals, args.triplets)
    json.dump(fs, open(f"{args.out}/eval_inputs.json", "w"), indent=1)
    T = triplet_level(df)
    keys = df.groupby(["arm", "train_contract", "assembly_version", "atlas_version", "role", "direction"]).size()
    keys.rename("n_rows").reset_index().to_csv(f"{args.out}/arm_keys.tsv", sep="\t", index=False)
    P, S, C = e1a(T, spec, args.out)
    DA, SD, tol = e2(df, T, spec, args.out)
    U = unlocks(DA, SD, spec, T)
    U["d00_tolerance_check"] = tol
    json.dump(U, open(f"{args.out}/e2_unlock_evaluation.json", "w"), indent=1, default=float)
    print(P.to_string())
    print(json.dumps({k: v for k, v in U.items() if k not in ("rule", "U2_pairs")}, indent=1, default=float))


if __name__ == "__main__":
    main()
