"""R6 v3 sensitivity correction (private plan 2026-11-20 section 6.2; spec configs/r6_sensitivity_v3/spec.json).

Rebuilds the v2 R6-R sensitivity analyses S5 (identical / salt-variant compounds only) and S7 (drug
leave-one-out) TOGETHER WITH their wrong-match comparator sets:

    v2 S5   query links filtered to identical/salt drugs, but the wrong-match value of each link still
            averaged over ALL disjoint pairs of the stratum (other-class drugs included).
    v3 S5   the same filter is applied to the wrong-match candidates: a candidate pair (d', e') is kept
            only when both d' and e' are identical/salt in that plate pair. Links left without any
            candidate are excluded (counted).
    v2 S7   leaving drug g out removed the links whose query pair contains g, but g stayed inside the
            wrong-match candidates of the remaining links.
    v3 S7   g is removed as query AND as wrong-match candidate; links left without a candidate are excluded.
    v3 S5-S1  common-anchor centring with the query AND the anchor set restricted to identical/salt drugs
            (at least 2 anchors, the v2 rule).

The drug-block bootstrap of v2 keeps the comparator value of each link fixed inside a replicate, so its
CIs are conditional on the comparator set (labelled `conditional_fixed_comparator`). The pre-specified
secondary bootstrap `rebuilt_comparator` recomputes, inside every replicate, the wrong-match mean of each
link with candidate weights m[d'] * m[e'] (the replicate's drug multiplicities); a link whose candidate
weights are all zero is dropped from that replicate. It uses the same multiplicity matrix as v2.

Modes
    --mode membership   metadata only (no response value is read): comparator_membership.tsv
    --mode score        reads the v1 treated-well means once: sensitivity_corrected.tsv, link tables,
                        v2 reproduction check, receipt
    --wiring-test       score on random arrays, outputs under $R6F_ASSETS/scratch (no response is read)

Nothing is trained. The v2 code (src/r6_specificity_v2/repeat_audit.py) is imported, never modified.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.r6_specificity_v2.repeat_audit import Boot, cos, rmse, sha256

MAN2 = Path("manifests/r6_specificity_v2")
REP2 = Path("reports/r6_specificity_v2")
MAN3 = Path("manifests/r6_sensitivity_v3")
REP3 = Path("reports/r6_sensitivity_correction_v3")
OK = lambda c: c != "other"  # identical or salt_variant


# ------------------------------------------------------------------------------------------------
# metadata: strata, name classes, comparator sets
# ------------------------------------------------------------------------------------------------
def load_meta():
    pairs = pd.read_csv(MAN2 / "eligible_pair_links.tsv", sep="\t")
    anch = pd.read_csv(MAN2 / "eligible_anchor_links.tsv", sep="\t")
    same = pd.read_csv(MAN2 / "eligible_sameplate_wells.tsv", sep="\t")
    wells = pd.read_csv(MAN2 / "usable_wells.tsv", sep="\t")
    strata = {k: sorted(set(g.drug_d) | set(g.drug_e)) for k, g in pairs.groupby(["line", "plate_P", "plate_Q"])}
    nc = {}
    for col_d, col_c in (("drug_d", "name_class_d"), ("drug_e", "name_class_e")):
        for pp, d, c in zip(pairs.plate_pair, pairs[col_d], pairs[col_c]):
            assert nc.setdefault((pp, int(d)), c) == c, "name class must be constant per (plate pair, drug)"
    for pp, d, c in zip(anch.plate_pair, anch.drug, anch.name_class):
        assert nc.setdefault((pp, int(d)), c) == c
    return pairs, anch, same, wells, strata, nc


def candidates(common, d, e, pp, nc, filt_ok=False, drop=None):
    """disjoint wrong-match pairs (d' < e') of the stratum for query (d, e)."""
    oth = [x for x in common if x not in (d, e) and x != drop and (not filt_ok or OK(nc[(pp, x)]))]
    return list(itertools.combinations(sorted(oth), 2))


def analyses(prim_drugs):
    """(analysis id, version, kind, dropped drug, query filter, candidate filter, candidate drop)"""
    out = [("primary", "v2=v3", "pair", None, False, False, False),
           ("S5", "v2", "pair", None, True, False, False),
           ("S5", "v3", "pair", None, True, True, False)]
    for g in prim_drugs:
        out.append(("S7", "v2", "pair", g, False, False, False))
        out.append(("S7", "v3", "pair", g, False, False, True))
    return out


def membership(pairs, anch, strata, nc) -> pd.DataFrame:
    prim = pairs[pairs.n_disjoint_pairs > 0]
    prim_drugs = sorted(int(x) for x in set(prim.drug_d) | set(prim.drug_e))
    rows = []
    for an, ver, _, g, qf, cf, cd in analyses(prim_drugs):
        for r in prim.itertuples():
            l, P, Q, pp, d, e = int(r.line), int(r.plate_P), int(r.plate_Q), r.plate_pair, int(r.drug_d), int(r.drug_e)
            if g is not None and g in (d, e):
                continue
            if qf and not (OK(r.name_class_d) and OK(r.name_class_e)):
                continue
            cand = candidates(strata[(l, P, Q)], d, e, pp, nc, filt_ok=cf, drop=g if cd else None)
            cdrugs = sorted(set(itertools.chain.from_iterable(cand)))
            rows.append(dict(analysis=an, version=ver, comparator_kind="wrong_match_pairs", dropped_drug=g if g is not None else "",
                             plate_pair=pp, line=l, query_d=d, query_e=e, name_class_d=r.name_class_d, name_class_e=r.name_class_e,
                             members=";".join(f"{a}:{b}" for a, b in cand), n_members=len(cand),
                             member_drugs=",".join(map(str, cdrugs)),
                             members_include_other_class=any(not OK(nc[(pp, x)]) for x in cdrugs),
                             members_include_dropped_drug=bool(g is not None and g in cdrugs),
                             link_included=len(cand) > 0, exclusion=("" if cand else "no wrong-match candidate left")))
    # S5 single-drug common-anchor centring
    for ver, af in (("v2", False), ("v3", True)):
        for r in anch.itertuples():
            l, P, Q, pp, d = int(r.line), int(r.plate_P), int(r.plate_Q), r.plate_pair, int(r.drug)
            if not OK(r.name_class):
                continue
            oth = [x for x in strata[(l, P, Q)] if x != d and (not af or OK(nc[(pp, x)]))]
            rows.append(dict(analysis="S5-S1", version=ver, comparator_kind="anchor_drugs", dropped_drug="", plate_pair=pp, line=l,
                             query_d=d, query_e="", name_class_d=r.name_class, name_class_e="", members=",".join(map(str, oth)),
                             n_members=len(oth), member_drugs=",".join(map(str, oth)),
                             members_include_other_class=any(not OK(nc[(pp, x)]) for x in oth), members_include_dropped_drug=False,
                             link_included=len(oth) >= 2, exclusion=("" if len(oth) >= 2 else "fewer than 2 anchors")))
    M = pd.DataFrame(rows)
    # collapse over cell lines (one row per analysis x plate pair x query); assert identical membership across lines
    key = ["analysis", "version", "comparator_kind", "dropped_drug", "plate_pair", "query_d", "query_e"]
    chk = M.groupby(key, dropna=False).members.nunique()
    assert (chk == 1).all(), "comparator membership differs across cell lines of a plate pair"
    C = (M.groupby(key + ["name_class_d", "name_class_e", "members", "n_members", "member_drugs", "members_include_other_class",
                          "members_include_dropped_drug", "link_included", "exclusion"], dropna=False)
         .line.agg(n_lines="size").reset_index())
    return M, C


# ------------------------------------------------------------------------------------------------
# bootstrap helpers
# ------------------------------------------------------------------------------------------------
def pair_matrix(bt: Boot, V, valid, a, b, mult):
    """drug-macro of link values with per-replicate values V [R, L] and validity [R, L]."""
    ai = np.array([bt.ix[int(x)] for x in a]); bi = np.array([bt.ix[int(x)] for x in b])
    n, L = len(bt.du), len(a)
    Ia = np.zeros((L, n)); Ia[np.arange(L), ai] = 1
    Ib = np.zeros((L, n)); Ib[np.arange(L), bi] = 1
    V0 = np.where(valid, V, 0.0)
    wa, wb = mult[:, bi] * valid, mult[:, ai] * valid
    num = (wa * V0) @ Ia + (wb * V0) @ Ib
    den = wa @ Ia + wb @ Ib
    has = den > 0
    dm = np.where(has, num / np.maximum(den, 1e-12), 0.0)
    W = mult * has
    return np.where(W.sum(1) > 0, (W * dm).sum(1) / np.maximum(W.sum(1), 1e-12), np.nan)


def rebuilt_stat(bt: Boot, matched, cand_link, cand_d2, cand_e2, cand_val, a, b, chunk=250):
    """matched - rebuilt wrong-match, drug-block bootstrap with the comparator recomputed per replicate."""
    L = len(matched)
    order = np.argsort(cand_link, kind="stable")
    cl, cv = cand_link[order], cand_val[order]
    i2 = np.array([bt.ix[int(x)] for x in cand_d2[order]]); i3 = np.array([bt.ix[int(x)] for x in cand_e2[order]])
    starts = np.r_[0, np.flatnonzero(np.diff(cl)) + 1]
    assert np.array_equal(cl[starts], np.arange(L)), "every link needs >= 1 candidate at the point estimate"

    def wrong(mult):
        w = mult[:, i2] * mult[:, i3]
        num = np.add.reduceat(w * cv, starts, axis=1); den = np.add.reduceat(w, starts, axis=1)
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan), den > 0

    one = np.ones((1, len(bt.du)))
    wpt, _ = wrong(one)
    pt = pair_matrix(bt, matched[None, :] - wpt, np.ones((1, L), bool), a, b, one)[0]
    bs, n_drop = [], []
    for s in range(0, bt.M.shape[0], chunk):
        m = bt.M[s:s + chunk]
        wb, ok = wrong(m)
        bs.append(pair_matrix(bt, matched[None, :] - np.where(ok, wb, 0.0), ok, a, b, m))
        n_drop.append((~ok).sum(1))
    bs = np.concatenate(bs); n_drop = np.concatenate(n_drop)
    st = Boot._pack(pt, bs, None, L)
    st["n_drugs"] = len(set(map(int, a)) | set(map(int, b)))
    st["mean_links_dropped_per_replicate"] = float(n_drop.mean())
    return st


# ------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/r6_sensitivity_v3/spec.json")
    ap.add_argument("--mode", choices=["membership", "score"], required=True)
    ap.add_argument("--wiring-test", action="store_true")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    pairs, anch, same, wells, strata, nc = load_meta()
    Mlong, Mc = membership(pairs, anch, strata, nc)
    if a.mode == "membership":
        REP3.mkdir(parents=True, exist_ok=True)
        Mc.to_csv(REP3 / "comparator_membership.tsv", sep="\t", index=False)
        print(Mc.groupby(["analysis", "version"]).agg(n_query=("n_lines", "size"), n_links=("n_lines", "sum"),
                                                      n_included=("link_included", "sum"),
                                                      any_other=("members_include_other_class", "sum"),
                                                      any_dropped=("members_include_dropped_drug", "sum")).to_string())
        return

    out_dir = REP3
    if a.wiring_test:
        out_dir = Path(os.environ["R6F_ASSETS"]) / "scratch" / "wiring_sensitivity_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    B, seed = int(spec["n_boot"]), int(spec["boot_seed"])
    halves = Path(os.environ["R6F_ASSETS"]) / "cache" / "halves_v1"
    tm = np.load(halves / "treated_halves.npy", mmap_mode="r")
    if a.wiring_test:  # same random generator as the v2 wiring test; no response value is read
        tm = np.random.default_rng(1).normal(size=(3, int(wells.wid.max()) + 1, tm.shape[2])).astype(np.float32)

    # ---- per (line, drug, plate) full-well responses, cell-weighted (v2 rule) -------------------------
    X = {}
    for (l, d, P), w in wells.groupby(["line", "drug", "plate_index"]):
        n = w["n_full"].to_numpy().astype(np.float64)
        X[(int(l), int(d), int(P))] = (np.asarray(tm[2, w.wid.to_numpy()], np.float64) * n[:, None]).sum(0) / n.sum()
    con = lambda l, d, e, P: X[(l, d, P)] - X[(l, e, P)]

    # ---- per link: matched; per candidate x direction: wrong cos / wrong rmse ---------------------------
    prim = pairs[pairs.n_disjoint_pairs > 0].reset_index(drop=True)
    links, cands = [], []
    for li, r in enumerate(prim.itertuples()):
        l, P, Q, d, e = int(r.line), int(r.plate_P), int(r.plate_Q), int(r.drug_d), int(r.drug_e)
        cP, cQ = con(l, d, e, P), con(l, d, e, Q)
        links.append(dict(li=li, line=l, plate_P=P, plate_Q=Q, plate_pair=r.plate_pair, drug_d=d, drug_e=e,
                          name_class_d=r.name_class_d, name_class_e=r.name_class_e,
                          cos_matched=cos(cP, cQ), rmse_matched=rmse(cP, cQ)))
        others = [x for x in strata[(l, P, Q)] if x not in (d, e)]
        for d2, e2 in itertools.combinations(others, 2):
            for S, T in ((P, Q), (Q, P)):
                u, v = con(l, d, e, S), con(l, d2, e2, T)
                cands.append((li, d2, e2, cos(u, v), float(np.sqrt(np.mean(u ** 2) + np.mean(v ** 2)))))
    Lk = pd.DataFrame(links)
    Cd = pd.DataFrame(cands, columns=["li", "d2", "e2", "cos_w", "rmse_w"])
    Cd["ok2"] = [OK(nc[(pp, x)]) and OK(nc[(pp, y)]) for pp, x, y in zip(Lk.plate_pair.to_numpy()[Cd.li], Cd.d2, Cd.e2)]

    drugs_all = set(pairs.drug_d) | set(pairs.drug_e) | set(anch.drug) | set(same.drug)
    bt = Boot(drugs_all, B, seed)
    rows = []

    def label(st, fav_sign):
        lo, hi = st.get("lo", np.nan), st.get("hi", np.nan)
        if not np.isfinite(lo):
            return "CI 없음"
        if lo <= 0 <= hi:
            return "차이 미확립 (CI 0 포함)"
        return "유리 방향, CI 0 제외" if np.sign(st["estimate"]) == fav_sign else "불리 방향, CI 0 제외"

    def comp(cmask, lmask):
        """links of lmask with their wrong-match mean over candidates in cmask; drop links without candidate."""
        sub = Cd[cmask & lmask[Cd.li.to_numpy()]]
        g = sub.groupby("li").agg(cos_wrong=("cos_w", "mean"), rmse_wrong=("rmse_w", "mean"))
        D = Lk[lmask].join(g, on="li", how="inner").copy()
        D["d_cos"], D["d_rmse"] = D.cos_matched - D.cos_wrong, D.rmse_matched - D.rmse_wrong
        return D, sub, int(lmask.sum() - len(D))

    def emit(analysis, version, quantity, st, ci_type, fav_sign=None, **kw):
        rec = dict(analysis=analysis, version=version, quantity=quantity, estimate=st.get("estimate"), lo=st.get("lo"), hi=st.get("hi"),
                   p_boot=st.get("p_boot"), n_drugs=st.get("n_drugs"), n_links=st.get("n_links"), ci_type=ci_type)
        if fav_sign is not None:
            rec["status"] = label(st, fav_sign)
        rec.update(kw)
        rows.append(rec)

    def block(analysis, version, D, sub, n_excl, rebuilt=False, **kw):
        a_, b_ = D.drug_d.to_numpy(), D.drug_e.to_numpy()
        p1 = bt.pair_stat(D.d_cos.to_numpy(), a_, b_); p2 = bt.pair_stat(D.d_rmse.to_numpy(), a_, b_)
        wr = bt.pair_stat(D.rmse_wrong.to_numpy(), a_, b_)["estimate"]
        meta = dict(n_plate_pairs=int(D.plate_pair.nunique()), n_links_excluded_no_comparator=n_excl,
                    n_wrong_candidates=int(len(sub)) // 2, **kw)
        emit(analysis, version, "P1_cos (matched - wrong cos)", p1, "conditional_fixed_comparator", +1, **meta)
        emit(analysis, version, "P2_rmse (matched - wrong RMSE)", p2, "conditional_fixed_comparator", -1,
             rel_to_wrong=p2["estimate"] / wr, rel_lo=p2["lo"] / wr, rel_hi=p2["hi"] / wr, **meta)
        for q in ("cos_matched", "cos_wrong", "rmse_matched", "rmse_wrong"):
            emit(analysis, version, q, bt.pair_stat(D[q].to_numpy(), a_, b_),
                 "conditional_fixed_comparator" if "wrong" in q else "drug_block", **meta)
        res = dict(p1=p1, p2=p2, wr=wr)
        if rebuilt:
            idx = {li: k for k, li in enumerate(D.li.to_numpy())}
            s = sub[sub.li.isin(idx)]
            cl = s.li.map(idx).to_numpy()
            for qn, mcol, ccol, fs in (("P1_cos (matched - wrong cos)", "cos_matched", "cos_w", +1),
                                       ("P2_rmse (matched - wrong RMSE)", "rmse_matched", "rmse_w", -1)):
                st = rebuilt_stat(bt, D[mcol].to_numpy(), cl, s.d2.to_numpy(), s.e2.to_numpy(), s[ccol].to_numpy(), a_, b_)
                ex = dict(rel_to_wrong=st["estimate"] / wr, rel_lo=st["lo"] / wr, rel_hi=st["hi"] / wr) if fs < 0 else {}
                emit(analysis, version, qn, st, "rebuilt_comparator", fs, n_plate_pairs=int(D.plate_pair.nunique()),
                     mean_links_dropped_per_replicate=st["mean_links_dropped_per_replicate"], **ex)
                res["rb_" + qn[:2]] = st
        return res

    allL = np.ones(len(Lk), bool)
    qok = (Lk.name_class_d.map(OK) & Lk.name_class_e.map(OK)).to_numpy()
    allC = np.ones(len(Cd), bool)

    # primary (reproduction of v2 + rebuilt bootstrap)
    Dp, subp, _ = comp(allC, allL)
    prim_res = block("primary", "v2=v3", Dp, subp, 0, rebuilt=True)
    # S5
    D5a, sub5a, ex5a = comp(allC, qok)
    s5_before = block("S5", "v2 (before)", D5a, sub5a, ex5a)
    D5b, sub5b, ex5b = comp(Cd.ok2.to_numpy(), qok)
    s5_after = block("S5", "v3 (after)", D5b, sub5b, ex5b, rebuilt=True)
    # S7
    prim_drugs = sorted(set(Lk.drug_d) | set(Lk.drug_e))
    loo = {"v2 (before)": {"P1": [], "P2": []}, "v3 (after)": {"P1": [], "P2": []}}
    for g in prim_drugs:
        lm = ((Lk.drug_d != g) & (Lk.drug_e != g)).to_numpy()
        for ver, cm in (("v2 (before)", allC), ("v3 (after)", ((Cd.d2 != g) & (Cd.e2 != g)).to_numpy())):
            D, sub, ex = comp(cm, lm)
            a_, b_ = D.drug_d.to_numpy(), D.drug_e.to_numpy()
            p1 = bt.pair_stat(D.d_cos.to_numpy(), a_, b_); p2 = bt.pair_stat(D.d_rmse.to_numpy(), a_, b_)
            meta = dict(dropped_drug=int(g), n_plate_pairs=int(D.plate_pair.nunique()), n_links_excluded_no_comparator=ex,
                        candidates_with_dropped_drug=int(((sub.d2 == g) | (sub.e2 == g)).sum()) // 2)
            emit("S7 drug LOO", ver, "P1_cos (matched - wrong cos)", p1, "conditional_fixed_comparator", +1, **meta)
            emit("S7 drug LOO", ver, "P2_rmse (matched - wrong RMSE)", p2, "conditional_fixed_comparator", -1, **meta)
            loo[ver]["P1"].append(p1["estimate"]); loo[ver]["P2"].append(p2["estimate"])
    for ver, dd in loo.items():
        for k, fs in (("P1", +1), ("P2", -1)):
            v = np.array(dd[k])
            n_fav = int((np.sign(v) == fs).sum())
            emit("S7 drug LOO range", ver, f"{k} median [min, max] over {len(v)} drugs", dict(estimate=float(np.median(v)), lo=float(v.min()), hi=float(v.max()),
                                                                                         n_drugs=len(v)), "point_range",
                 status=f"유리 방향 {n_fav}/{len(v)}")
    # S3 plate LOO (comparator unaffected: dropping a plate drops whole strata) -- needed to re-derive the v2 verdict
    s3_sign = []
    for Xp in sorted(set(Lk.plate_P) | set(Lk.plate_Q)):
        lm = ((Lk.plate_P != Xp) & (Lk.plate_Q != Xp)).to_numpy()
        if lm.sum() == 0:
            emit("S3 plate LOO", "v2=v3", f"drop plate {Xp}", dict(n_drugs=0, n_links=0), "none", note="no primary link left")
            continue
        D, sub, ex = comp(allC, lm)
        a_, b_ = D.drug_d.to_numpy(), D.drug_e.to_numpy()
        p1 = bt.pair_stat(D.d_cos.to_numpy(), a_, b_); p2 = bt.pair_stat(D.d_rmse.to_numpy(), a_, b_)
        emit("S3 plate LOO", "v2=v3", f"drop plate {Xp}: P1_cos", p1, "conditional_fixed_comparator", +1)
        emit("S3 plate LOO", "v2=v3", f"drop plate {Xp}: P2_rmse", p2, "conditional_fixed_comparator", -1)
        s3_sign.append((p1["estimate"] > 0) and (p2["estimate"] < 0))

    # S5-S1 common-anchor centring, before / after (anchor = reference set held fixed under resampling)
    def s1(anchor_filter):
        vals, dr, n_ex = [], [], 0
        for r in anch.itertuples():
            l, P, Q, pp, d = int(r.line), int(r.plate_P), int(r.plate_Q), r.plate_pair, int(r.drug)
            if not OK(r.name_class):
                continue
            oth = [x for x in strata[(l, P, Q)] if x != d and (not anchor_filter or OK(nc[(pp, x)]))]
            if len(oth) < 2:
                n_ex += 1
                continue
            aP = X[(l, d, P)] - np.mean([X[(l, x, P)] for x in oth], 0)
            aQ = X[(l, d, Q)] - np.mean([X[(l, x, Q)] for x in oth], 0)
            vals.append(cos(aP, aQ)); dr.append(d)
        return bt.single_stat(np.array(vals), np.array(dr)), n_ex
    for ver, af in (("v2 (before)", False), ("v3 (after)", True)):
        st, n_ex = s1(af)
        emit("S5-S1 anchor", ver, "cos_anchor", st, "conditional_fixed_reference", +1, n_links_excluded_no_comparator=n_ex)

    # ---- verdict re-derivation with the frozen v2 rule ---------------------------------------------------
    def verdict(p1, p2, wr):
        ps = sorted([("P1", p1["p_boot"]), ("P2", p2["p_boot"])], key=lambda t: t[1])
        holm = {}
        for i, (k, p) in enumerate(ps):
            holm[k] = bool(p <= 0.05 / (2 - i)) and (i == 0 or holm[ps[0][0]])
        sig = {"P1": holm["P1"] and p1["estimate"] > 0, "P2": holm["P2"] and p2["estimate"] < 0}
        practical = abs(p2["estimate"] / wr) >= 0.005
        if sig["P1"] and sig["P2"] and practical and all(s3_sign):
            v = "repeatable"
        elif any(sig.values()):
            v = "weak_or_uncertain"
        else:
            v = "not_established"
        return v, dict(holm=holm, sig_favourable=sig, p2_practical=practical, plate_loo_all_favourable=all(s3_sign))
    order = ["not_established", "weak_or_uncertain", "repeatable"]
    verd = {}
    # the v2 rule is defined for the primary set only; S5/S7 rows carry per-row status labels instead
    verd["primary | conditional"] = verdict(prim_res["p1"], prim_res["p2"], prim_res["wr"])
    verd["primary | rebuilt"] = verdict(prim_res["rb_P1"], prim_res["rb_P2"], prim_res["wr"])
    for k, (v, info) in verd.items():
        emit("verdict (v2 rule)", k, v, dict(), "rule", note=json.dumps(info))
    v3_r6r = min(verd["primary | conditional"][0], verd["primary | rebuilt"][0], key=order.index)
    emit("verdict (v2 rule)", "R6-R v3 record (conservative of conditional / rebuilt)", v3_r6r, dict(), "rule")

    T = pd.DataFrame(rows)
    T.insert(0, "package", "P0-R6 v3")
    T.to_csv(out_dir / "sensitivity_corrected.tsv", sep="\t", index=False, float_format="%.6f")
    Lfull = Dp.drop(columns=["li"])
    Lfull.to_csv(out_dir / "link_values_primary.tsv", sep="\t", index=False, float_format="%.6f")
    D5b.drop(columns=["li"]).to_csv(out_dir / "link_values_S5_v3.tsv", sep="\t", index=False, float_format="%.6f")

    # ---- v2 reproduction check (raw-table regeneration, plan section 7.2 item 4) -------------------------
    chk = []
    if not a.wiring_test:
        v2 = pd.read_csv(REP2 / "repeat_contrast_audit.tsv", sep="\t")
        def v2row(q, s):
            r = v2[(v2.quantity == q) & (v2.set == s)]
            assert len(r) == 1, (q, s)
            return r.iloc[0]
        tests = [("primary P1", v2row("P1_cos: matched cos - wrong-match cos", "primary (disjoint wrong-match available)"), prim_res["p1"]),
                 ("primary P2", v2row("P2_rmse: matched RMSE - wrong-match RMSE", "primary (disjoint wrong-match available)"), prim_res["p2"]),
                 ("S5 v2 d_cos", v2row("d_cos", "S5 primary, identical or salt-variant compounds only"), s5_before["p1"]),
                 ("S5 v2 d_rmse", v2row("d_rmse", "S5 primary, identical or salt-variant compounds only"), s5_before["p2"])]
        for name, r, st in tests:
            for f in ("estimate", "lo", "hi", "p_boot"):
                chk.append(dict(item=name, field=f, v2=float(r[f]), v3_recomputed=float(st[f]), abs_diff=abs(float(r[f]) - float(st[f]))))
        for k in ("P1", "P2"):
            r = v2row(f"S7 drug LOO range: {k}", "primary"); v = np.array(loo["v2 (before)"][k])
            for f, val in (("estimate", np.median(v)), ("lo", v.min()), ("hi", v.max())):
                chk.append(dict(item=f"S7 v2 range {k}", field=f, v2=float(r[f]), v3_recomputed=float(val), abs_diff=abs(float(r[f]) - float(val))))
        r = v2row("S1 cos_anchor", "S5 identical or salt-variant only")
        st = T[(T.analysis == "S5-S1 anchor") & (T.version == "v2 (before)")].iloc[0]
        for f in ("estimate", "lo", "hi"):
            chk.append(dict(item="S5-S1 v2 cos_anchor", field=f, v2=float(r[f]), v3_recomputed=float(st[f]), abs_diff=abs(float(r[f]) - float(st[f]))))
        v2L = pd.read_csv(REP2 / "paired_plate_contrasts.tsv", sep="\t")
        m = Dp.merge(v2L[v2L.primary_set], on=["line", "plate_P", "plate_Q", "drug_d", "drug_e"], suffixes=("", "_v2"))
        assert len(m) == len(Dp)
        for q in ("cos_matched", "rmse_matched", "cos_wrong", "rmse_wrong", "d_cos", "d_rmse"):
            chk.append(dict(item=f"per-link {q} ({len(m)} links)", field="max_abs_diff", v2=np.nan, v3_recomputed=np.nan,
                            abs_diff=float(np.abs(m[q] - m[q + "_v2"]).max())))
        chk.append(dict(item="verdict primary conditional", field="label", v2=np.nan, v3_recomputed=np.nan,
                        abs_diff=float(verd["primary | conditional"][0] != v2[v2.row_type == "verdict"].quantity.iloc[0])))
    C = pd.DataFrame(chk)
    tol = float(spec["reproduction_tolerance"])
    repro_ok = bool(len(C) and (C.abs_diff <= tol).all())
    C["pass"] = C.abs_diff <= tol
    C.to_csv(out_dir / "v2_reproduction_check.tsv", sep="\t", index=False, float_format="%.7f")

    rec = dict(spec=a.spec, spec_sha256=sha256(a.spec), commit=os.popen("git rev-parse HEAD").read().strip(),
               dirty=os.popen("git status --porcelain -uno | wc -l").read().strip(), wiring_test=a.wiring_test,
               inputs={str(MAN2 / f): sha256(MAN2 / f) for f in ("usable_wells.tsv", "eligible_pair_links.tsv", "eligible_anchor_links.tsv",
                                                                 "eligible_sameplate_wells.tsv")},
               comparator_membership_sha256=sha256(REP3 / "comparator_membership.tsv") if (REP3 / "comparator_membership.tsv").exists() else None,
               treated_halves=str((halves / "treated_halves.npy").resolve()), treated_halves_bytes=os.path.getsize(halves / "treated_halves.npy"),
               n_boot=B, boot_seed=seed, bootstrap_drugs=[int(x) for x in bt.du],
               n_primary_links=int(len(Dp)), n_s5_v2_links=int(len(D5a)), n_s5_v3_links=int(len(D5b)), n_s5_v3_excluded=ex5b,
               verdicts={k: v for k, (v, _) in verd.items()}, r6r_v3_record=v3_r6r, v2_reproduction_pass=repro_ok)
    rpath = out_dir / "receipt.json" if a.wiring_test else MAN3 / "score_receipt.json"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(rec, indent=1, default=float))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 50)
    print(T.drop(columns=["package"]).to_string())
    print(C.to_string())
    print(json.dumps({k: rec[k] for k in ("verdicts", "r6r_v3_record", "v2_reproduction_pass", "n_s5_v3_links", "n_s5_v3_excluded")}, indent=1))


if __name__ == "__main__":
    main()
