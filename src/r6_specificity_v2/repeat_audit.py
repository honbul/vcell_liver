"""R6-R repeat-contrast audit (private plan 2026-11-19 section 3.2; spec configs/r6_specificity_v2/r6r_spec.json).

Uses ONLY the eligible sets written by repeat_eligibility.py (metadata) and the v1 treated-well means
(halves_v1/treated_halves.npy: [A, B, full] x wells x 2000 genes, read-only). Nothing is trained.
Every primary quantity is a difference of two treated wells inside one control pool, so the control
reference mean cancels and is not read.

Outputs
    reports/r6_specificity_v2/repeat_contrast_audit.tsv     summary rows (primary, secondary, sensitivity)
    reports/r6_specificity_v2/paired_plate_contrasts.tsv    one row per eligible pair link
    reports/r6_specificity_v2/repeat_contrast_by_drug.tsv   drug-level values
    manifests/r6_specificity_v2/repeat_audit_receipt.json   input hashes, verdict, counts
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

MAN = Path("manifests/r6_specificity_v2")
REP = Path("reports/r6_specificity_v2")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cos(u, v) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def rmse(u, v) -> float:
    return float(np.sqrt(np.mean((u - v) ** 2)))


# ------------------------------------------------------------------------------------------------
# drug-block bootstrap of drug-macro means
# ------------------------------------------------------------------------------------------------
class Boot:
    def __init__(self, drugs, n_boot: int, seed: int):
        self.du = np.array(sorted(set(int(x) for x in drugs)))
        self.ix = {int(g): i for i, g in enumerate(self.du)}
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(self.du), size=(n_boot, len(self.du)))
        self.M = np.stack([np.bincount(r, minlength=len(self.du)) for r in idx]).astype(np.float64)

    def _pair(self, v, a, b, mult):
        """drug-macro of link values v over links (a, b); mult [R, n_drug] (R = 1 for the point)."""
        ok = ~np.isnan(v)
        v, ai, bi = v[ok], np.array([self.ix[int(x)] for x in a[ok]]), np.array([self.ix[int(x)] for x in b[ok]])
        n = len(self.du)
        Ia = np.zeros((len(v), n)); Ia[np.arange(len(v)), ai] = 1
        Ib = np.zeros((len(v), n)); Ib[np.arange(len(v)), bi] = 1
        wa, wb = mult[:, bi], mult[:, ai]                             # partner multiplicities
        num = (wa * v) @ Ia + (wb * v) @ Ib
        den = wa @ Ia + wb @ Ib
        has = den > 0
        dm = np.where(has, num / np.maximum(den, 1e-12), 0.0)
        W = mult * has
        return np.where(W.sum(1) > 0, (W * dm).sum(1) / np.maximum(W.sum(1), 1e-12), np.nan), int(has[0].sum()) if mult.shape[0] == 1 else None

    def pair_stat(self, v, a, b) -> dict:
        pt, nd = self._pair(v, a, b, np.ones((1, len(self.du))))
        bs, _ = self._pair(v, a, b, self.M)
        return self._pack(pt[0], bs, nd, int((~np.isnan(v)).sum()))

    def single_stat(self, v, d) -> dict:
        ok = ~np.isnan(v)
        v, di = v[ok], np.array([self.ix[int(x)] for x in d[ok]])
        n = len(self.du)
        I = np.zeros((len(v), n)); I[np.arange(len(v)), di] = 1
        cnt = I.sum(0); has = cnt > 0
        dm = np.where(has, (v @ I) / np.maximum(cnt, 1), 0.0)
        pt = float(dm[has].mean()) if has.any() else float("nan")
        W = self.M * has
        bs = (W * dm).sum(1) / np.maximum(W.sum(1), 1e-12)
        return self._pack(pt, bs, int(has.sum()), int(len(v)))

    @staticmethod
    def _pack(pt, bs, nd, nl) -> dict:
        bs = bs[~np.isnan(bs)]
        lo, hi = (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))) if len(bs) else (np.nan, np.nan)
        p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))) if len(bs) else np.nan
        p = max(p, 1.0 / max(len(bs), 1))
        return dict(estimate=float(pt), lo=lo, hi=hi, p_boot=p, n_drugs=nd, n_links=nl)


def inv_simpson(x) -> float:
    s = pd.Series(x).value_counts(normalize=True)
    return float(1.0 / (s ** 2).sum()) if len(s) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/r6_specificity_v2/r6r_spec.json")
    ap.add_argument("--wiring-test", action="store_true", help="random arrays instead of the treated means; outputs to assets/scratch")
    a = ap.parse_args()
    global REP
    out_man = MAN
    if a.wiring_test:
        REP = Path(os.environ["R6F_ASSETS"]) / "scratch" / "wiring_repeat_audit"
        out_man = REP
    spec = json.loads(Path(a.spec).read_text())
    B, seed = int(spec["n_boot"]), int(spec["boot_seed"])
    halves = Path(os.environ["R6F_ASSETS"]) / "cache" / "halves_v1"
    wells = pd.read_csv(MAN / "usable_wells.tsv", sep="\t")
    pairs = pd.read_csv(MAN / "eligible_pair_links.tsv", sep="\t")
    anch = pd.read_csv(MAN / "eligible_anchor_links.tsv", sep="\t")
    same = pd.read_csv(MAN / "eligible_sameplate_wells.tsv", sep="\t")
    elig = json.loads((MAN / "eligibility_summary.json").read_text())
    tm = np.load(halves / "treated_halves.npy", mmap_mode="r")           # [3, n_wells, G]
    if a.wiring_test:                                                    # no response value is read
        tm = np.random.default_rng(1).normal(size=(3, int(wells.wid.max()) + 1, tm.shape[2])).astype(np.float32)
    G = tm.shape[2]
    REP.mkdir(parents=True, exist_ok=True)

    # ---- identifiability gate (metadata only) ------------------------------------------------------
    prim = pairs[pairs.n_disjoint_pairs > 0]
    ident = dict(n_unique_drugs=len(set(prim.drug_d) | set(prim.drug_e)), n_plate_pairs=int(prim.plate_pair.nunique()))
    identifiable = ident["n_unique_drugs"] >= 8 and ident["n_plate_pairs"] >= 2
    print("identifiability", ident, identifiable, flush=True)

    # ---- per (line, drug, plate) responses: cell-weighted over usable wells ------------------------
    X = {}                                                             # (l, d, P) -> (full, A, B)
    for (l, d, P), w in wells.groupby(["line", "drug", "plate_index"]):
        wid = w.wid.to_numpy()
        out = []
        for j, ncol in ((2, "n_full"), (0, "n_A"), (1, "n_B")):
            n = w[ncol].to_numpy().astype(np.float64)
            out.append((np.asarray(tm[j, wid], np.float64) * n[:, None]).sum(0) / n.sum())
        X[(int(l), int(d), int(P))] = tuple(out)
    pool_wells = {k: g for k, g in wells.groupby(["line", "plate_index"])}

    # ---- pair links -------------------------------------------------------------------------------
    strata = {k: sorted(set(g.drug_d) | set(g.drug_e)) for k, g in pairs.groupby(["line", "plate_P", "plate_Q"])}

    def con(l, d, e, P, j=0):
        return X[(l, d, P)][j] - X[(l, e, P)][j]

    rows = []
    for r in pairs.itertuples():
        l, P, Q, d, e = int(r.line), int(r.plate_P), int(r.plate_Q), int(r.drug_d), int(r.drug_e)
        cP, cQ = con(l, d, e, P), con(l, d, e, Q)
        rec = dict(line=l, plate_P=P, plate_Q=Q, plate_pair=r.plate_pair, drug_d=d, drug_e=e, n_common_drugs=r.n_common_drugs,
                   n_disjoint_pairs=r.n_disjoint_pairs, name_class_d=r.name_class_d, name_class_e=r.name_class_e,
                   norm_P=float(np.sqrt(np.mean(cP ** 2))), norm_Q=float(np.sqrt(np.mean(cQ ** 2))),
                   cos_matched=cos(cP, cQ), rmse_matched=rmse(cP, cQ),
                   rmse_zero=float(np.sqrt((np.mean(cP ** 2) + np.mean(cQ ** 2)) / 2)))
        # wrong match: disjoint pairs of the same stratum, both directions
        others = [x for x in strata[(l, P, Q)] if x not in (d, e)]
        wc, wr = [], []
        for d2, e2 in itertools.combinations(others, 2):
            for S, T in ((P, Q), (Q, P)):
                u, v = con(l, d, e, S), con(l, d2, e2, T)
                wc.append(cos(u, v))
                wr.append(float(np.sqrt(np.mean(u ** 2) + np.mean(v ** 2))))   # orientation-averaged squared error
        rec.update(cos_wrong=float(np.mean(wc)) if wc else np.nan, rmse_wrong=float(np.mean(wr)) if wr else np.nan)
        # technical split (same well halves) on each plate, and noise-matched cross-plate halves
        hs = [cos(con(l, d, e, S, 1), con(l, d, e, S, 2)) for S in (P, Q)]
        hr = [rmse(con(l, d, e, S, 1), con(l, d, e, S, 2)) for S in (P, Q)]
        xs = [cos(con(l, d, e, P, 1), con(l, d, e, Q, 2)), cos(con(l, d, e, Q, 1), con(l, d, e, P, 2))]
        xr = [rmse(con(l, d, e, P, 1), con(l, d, e, Q, 2)), rmse(con(l, d, e, Q, 1), con(l, d, e, P, 2))]
        rec.update(cos_half_samewell=float(np.mean(hs)), rmse_half_samewell=float(np.mean(hr)),
                   cos_half_crossplate=float(np.mean(xs)), rmse_half_crossplate=float(np.mean(xr)))
        rows.append(rec)
    L = pd.DataFrame(rows)
    L["d_cos"] = L.cos_matched - L.cos_wrong
    L["d_rmse"] = L.rmse_matched - L.rmse_wrong
    L["d_rmse_zero"] = L.rmse_matched - L.rmse_zero
    L["primary_set"] = L.n_disjoint_pairs > 0
    L["identical_or_salt"] = (L.name_class_d != "other") & (L.name_class_e != "other")
    L.to_csv(REP / "paired_plate_contrasts.tsv", sep="\t", index=False, float_format="%.6f")

    # ---- single-drug links (S1 common anchor; S2 pool centring) ------------------------------------
    srows = []
    for r in anch.itertuples():
        l, P, Q, d = int(r.line), int(r.plate_P), int(r.plate_Q), int(r.drug)
        common = strata[(l, P, Q)]
        oth = [x for x in common if x != d]
        aP = X[(l, d, P)][0] - np.mean([X[(l, x, P)][0] for x in oth], 0)
        aQ = X[(l, d, Q)][0] - np.mean([X[(l, x, Q)][0] for x in oth], 0)
        rec = dict(line=l, plate_P=P, plate_Q=Q, plate_pair=r.plate_pair, drug=d, name_class=r.name_class, n_anchor=len(oth),
                   cos_anchor=cos(aP, aQ), rmse_anchor=rmse(aP, aQ), rmse_anchor_zero=float(np.sqrt((np.mean(aP ** 2) + np.mean(aQ ** 2)) / 2)))
        for tag, loo in (("pool", False), ("poolloo", True)):
            cs = []
            for S in (P, Q):
                pw = pool_wells[(l, S)]
                if loo:
                    pw = pw[pw.drug != d]
                # v1 rule: equal weight per usable well of the pool (cells already averaged within a well)
                cen = np.asarray(tm[2, pw.wid.to_numpy()], np.float64).mean(0)
                cs.append(X[(l, d, S)][0] - cen)
            rec[f"cos_{tag}"] = cos(cs[0], cs[1]); rec[f"rmse_{tag}"] = rmse(cs[0], cs[1])
            rec[f"rmse_{tag}_zero"] = float(np.sqrt((np.mean(cs[0] ** 2) + np.mean(cs[1] ** 2)) / 2))
        srows.append(rec)
    S = pd.DataFrame(srows)
    S["d_rmse_anchor_zero"] = S.rmse_anchor - S.rmse_anchor_zero

    # ---- S6 same plate, different well -------------------------------------------------------------
    s6, s6p = [], []
    two = {}
    for r in same.itertuples():
        l, d, P = int(r.line), int(r.drug), int(r.plate_index)
        w = wells[(wells.line == l) & (wells.drug == d) & (wells.plate_index == P)].sort_values("sample")
        w1, w2 = w.iloc[0], w.iloc[1]
        pw = pool_wells[(l, P)]; pw = pw[pw.drug != d].sort_values("sample")
        # the two wells share one pool: centre them by DISJOINT halves of the other wells (alternating by
        # sample id) so that a shared centring vector does not create agreement by itself
        wid_o = pw.wid.to_numpy()
        cen1 = np.asarray(tm[2, wid_o[0::2]], np.float64).mean(0); cen2 = np.asarray(tm[2, wid_o[1::2]], np.float64).mean(0)
        x1, x2 = np.asarray(tm[2, int(w1.wid)], np.float64), np.asarray(tm[2, int(w2.wid)], np.float64)
        two[(l, d, P)] = (x1, x2)
        s6.append(dict(line=l, drug=d, plate=P, name_class=r.name_class, sample_1=w1["sample"], sample_2=w2["sample"],
                       compound_1=w1.compound_name, compound_2=w2.compound_name, cos_poolloo=cos(x1 - cen1, x2 - cen2), rmse=rmse(x1 - cen1, x2 - cen2),
                       rmse_zero=float(np.sqrt((np.mean((x1 - cen1) ** 2) + np.mean((x2 - cen2) ** 2)) / 2))))
    for (l, P), g in same.groupby(["line", "plate_index"]):
        ds = sorted(int(x) for x in g.drug)
        for d, e in itertools.combinations(ds, 2):
            c1 = two[(l, d, P)][0] - two[(l, e, P)][0]; c2 = two[(l, d, P)][1] - two[(l, e, P)][1]
            s6p.append(dict(line=int(l), plate=int(P), drug_d=d, drug_e=e, cos=cos(c1, c2), rmse=rmse(c1, c2),
                            rmse_zero=float(np.sqrt((np.mean(c1 ** 2) + np.mean(c2 ** 2)) / 2))))
    S6, S6p = pd.DataFrame(s6), pd.DataFrame(s6p)

    # ---- summaries -----------------------------------------------------------------------------------
    allpair_drugs = set(L.drug_d) | set(L.drug_e) | set(S.drug) | set(S6.drug) | (set(S6p.drug_d) | set(S6p.drug_e) if len(S6p) else set())
    bt = Boot(allpair_drugs, B, seed)
    out = []

    def emit(kind, name, setname, st, **kw):
        out.append(dict(row_type=kind, quantity=name, set=setname, **st, **kw))

    def pair_block(D_, setname, kind="secondary"):
        for q in ("cos_matched", "cos_wrong", "d_cos", "rmse_matched", "rmse_wrong", "d_rmse", "rmse_zero", "d_rmse_zero",
                  "cos_half_samewell", "cos_half_crossplate", "rmse_half_samewell", "rmse_half_crossplate"):
            if D_[q].notna().any():
                emit(kind, q, setname, bt.pair_stat(D_[q].to_numpy(), D_.drug_d.to_numpy(), D_.drug_e.to_numpy()),
                     n_plate_pairs=int(D_.plate_pair.nunique()), eff_n_plate_pairs=inv_simpson(D_.plate_pair),
                     eff_n_drugs=inv_simpson(pd.concat([D_.drug_d, D_.drug_e])), n_lines=int(D_.line.nunique()))

    P_ = L[L.primary_set]
    p1 = bt.pair_stat(P_.d_cos.to_numpy(), P_.drug_d.to_numpy(), P_.drug_e.to_numpy())
    p2 = bt.pair_stat(P_.d_rmse.to_numpy(), P_.drug_d.to_numpy(), P_.drug_e.to_numpy())
    wr = bt.pair_stat(P_.rmse_wrong.to_numpy(), P_.drug_d.to_numpy(), P_.drug_e.to_numpy())["estimate"]
    ps = sorted([("P1_cos", p1["p_boot"]), ("P2_rmse", p2["p_boot"])], key=lambda t: t[1])
    holm = {}
    for i, (k, p) in enumerate(ps):
        holm[k] = bool(p <= 0.05 / (2 - i)) and (i == 0 or holm[ps[0][0]])
    fav = {"P1_cos": p1["estimate"] > 0, "P2_rmse": p2["estimate"] < 0}
    emit("primary", "P1_cos: matched cos - wrong-match cos", "primary (disjoint wrong-match available)", p1,
         holm_reject=holm["P1_cos"], favourable=fav["P1_cos"], n_plate_pairs=int(P_.plate_pair.nunique()),
         eff_n_plate_pairs=inv_simpson(P_.plate_pair), eff_n_drugs=inv_simpson(pd.concat([P_.drug_d, P_.drug_e])), n_lines=int(P_.line.nunique()))
    emit("primary", "P2_rmse: matched RMSE - wrong-match RMSE", "primary (disjoint wrong-match available)", p2,
         holm_reject=holm["P2_rmse"], favourable=fav["P2_rmse"], rel_to_wrong=p2["estimate"] / wr,
         rel_lo=p2["lo"] / wr, rel_hi=p2["hi"] / wr, n_plate_pairs=int(P_.plate_pair.nunique()),
         eff_n_plate_pairs=inv_simpson(P_.plate_pair), eff_n_drugs=inv_simpson(pd.concat([P_.drug_d, P_.drug_e])), n_lines=int(P_.line.nunique()))
    pair_block(P_, "primary (disjoint wrong-match available)")
    pair_block(L, "all pair links")
    pair_block(P_[P_.identical_or_salt], "S5 primary, identical or salt-variant compounds only", "sensitivity")
    # S3 plate leave-one-out
    loo_sign = []
    for X_ in sorted(set(L.plate_P) | set(L.plate_Q)):
        D_ = P_[(P_.plate_P != X_) & (P_.plate_Q != X_)]
        if len(D_) == 0 or len(set(D_.drug_d) | set(D_.drug_e)) < 2:
            emit("sensitivity", "S3 plate LOO", f"drop plate {X_}", dict(estimate=np.nan, n_drugs=0, n_links=0), note="no primary link left")
            continue
        a1 = bt.pair_stat(D_.d_cos.to_numpy(), D_.drug_d.to_numpy(), D_.drug_e.to_numpy())
        a2 = bt.pair_stat(D_.d_rmse.to_numpy(), D_.drug_d.to_numpy(), D_.drug_e.to_numpy())
        emit("sensitivity", "S3 plate LOO: P1_cos", f"drop plate {X_}", a1, n_plate_pairs=int(D_.plate_pair.nunique()))
        emit("sensitivity", "S3 plate LOO: P2_rmse", f"drop plate {X_}", a2, n_plate_pairs=int(D_.plate_pair.nunique()))
        loo_sign.append((a1["estimate"] > 0) and (a2["estimate"] < 0))
    # S7 drug leave-one-out range
    lo_rng = {"P1": [], "P2": []}
    for g in sorted(set(P_.drug_d) | set(P_.drug_e)):
        D_ = P_[(P_.drug_d != g) & (P_.drug_e != g)]
        lo_rng["P1"].append(bt.pair_stat(D_.d_cos.to_numpy(), D_.drug_d.to_numpy(), D_.drug_e.to_numpy())["estimate"])
        lo_rng["P2"].append(bt.pair_stat(D_.d_rmse.to_numpy(), D_.drug_d.to_numpy(), D_.drug_e.to_numpy())["estimate"])
    for k, v in lo_rng.items():
        emit("sensitivity", f"S7 drug LOO range: {k}", "primary", dict(estimate=float(np.median(v)), lo=float(np.min(v)), hi=float(np.max(v)),
                                                                     n_drugs=len(v), n_links=int(len(P_))), note="lo/hi = min/max over leaving one drug out; estimate = median")
    # per plate pair
    for pp, D_ in L.groupby("plate_pair"):
        for q in ("cos_matched", "d_cos", "d_rmse", "d_rmse_zero", "cos_half_samewell", "cos_half_crossplate"):
            if D_[q].notna().any():
                emit("by_plate_pair", q, f"plate pair {pp}", bt.pair_stat(D_[q].to_numpy(), D_.drug_d.to_numpy(), D_.drug_e.to_numpy()), n_lines=int(D_.line.nunique()))
    # S1 / S2 single drug
    for q in ("cos_anchor", "d_rmse_anchor_zero", "rmse_anchor", "rmse_anchor_zero", "cos_pool", "cos_poolloo"):
        emit("sensitivity", f"S1/S2 {q}", "single-drug links (strata with >= 3 common drugs)", bt.single_stat(S[q].to_numpy(), S.drug.to_numpy()),
             n_plate_pairs=int(S.plate_pair.nunique()))
    Sid = S[S.name_class != "other"]
    emit("sensitivity", "S1 cos_anchor", "S5 identical or salt-variant only", bt.single_stat(Sid.cos_anchor.to_numpy(), Sid.drug.to_numpy()))
    # S6
    if len(S6):
        S6["d_rmse_zero"] = S6.rmse - S6.rmse_zero
        for q in ("cos_poolloo", "d_rmse_zero"):
            emit("sensitivity", f"S6 same plate, different well: {q}", "8 drugs with 2 wells on one plate", bt.single_stat(S6[q].to_numpy(), S6.drug.to_numpy()))
        Si = S6[S6.name_class != "other"]
        emit("sensitivity", "S6 cos_poolloo", "S6 identical or salt-variant only", bt.single_stat(Si.cos_poolloo.to_numpy(), Si.drug.to_numpy()))
    if len(S6p):
        S6p["d_rmse_zero"] = S6p.rmse - S6p.rmse_zero
        for q in ("cos", "d_rmse_zero"):
            emit("sensitivity", f"S6 same-plate pair contrast: {q}", "drug pairs with 2 wells each on one plate",
                 bt.pair_stat(S6p[q].to_numpy(), S6p.drug_d.to_numpy(), S6p.drug_e.to_numpy()), n_plates=int(S6p.plate.nunique()))

    # ---- verdict -------------------------------------------------------------------------------------
    sig_fav = {k: holm[k] and fav[k] for k in holm}
    practical = abs(p2["estimate"] / wr) >= 0.005
    if not identifiable:
        verdict = "unidentifiable"
    elif sig_fav["P1_cos"] and sig_fav["P2_rmse"] and practical and all(loo_sign):
        verdict = "repeatable"
    elif any(sig_fav.values()):
        verdict = "weak_or_uncertain"
    else:
        verdict = "not_established"
    emit("verdict", verdict, "primary", dict(estimate=np.nan), note=json.dumps(dict(sig_favourable=sig_fav, p2_practical=practical,
                                                                                    plate_loo_all_favourable=all(loo_sign), identifiable=identifiable,
                                                                                    r6e_entry_from_r6r=verdict in ("repeatable", "weak_or_uncertain"))))
    T = pd.DataFrame(out)
    T.insert(0, "version", "r6_specificity_v2"); T.insert(0, "track", "R6")
    T.to_csv(REP / "repeat_contrast_audit.tsv", sep="\t", index=False, float_format="%.6f")

    # ---- per drug ------------------------------------------------------------------------------------
    long = pd.concat([L.assign(drug=L.drug_d), L.assign(drug=L.drug_e)])
    per = long.groupby("drug").agg(n_pair_links=("line", "size"), plate_pairs=("plate_pair", lambda s: ",".join(sorted(set(s)))),
                                   cos_matched=("cos_matched", "mean"), cos_wrong=("cos_wrong", "mean"), d_cos=("d_cos", "mean"),
                                   d_rmse=("d_rmse", "mean"), cos_half_samewell=("cos_half_samewell", "mean"),
                                   cos_half_crossplate=("cos_half_crossplate", "mean")).reset_index()
    sd = S.groupby("drug").agg(cos_anchor=("cos_anchor", "mean"), cos_pool=("cos_pool", "mean"), cos_poolloo=("cos_poolloo", "mean"),
                               name_class=("name_class", "first")).reset_index()
    per = per.merge(sd, on="drug", how="outer")
    per.to_csv(REP / "repeat_contrast_by_drug.tsv", sep="\t", index=False, float_format="%.4f")
    if len(S6):
        S6.to_csv(REP / "sameplate_well_repeats.tsv", sep="\t", index=False, float_format="%.6f")

    rec = dict(spec=a.spec, spec_sha256=sha256(a.spec), commit=os.popen("git rev-parse HEAD").read().strip(),
               inputs={f: sha256(MAN / f) for f in ("usable_wells.tsv", "eligible_pair_links.tsv", "eligible_anchor_links.tsv",
                                                    "eligible_sameplate_wells.tsv", "eligibility_summary.json")},
               treated_halves=str(halves / "treated_halves.npy"), treated_halves_bytes=os.path.getsize(halves / "treated_halves.npy"),
               identifiability=ident, identifiable=identifiable, verdict=verdict, holm=holm, favourable=fav, p2_practical=practical,
               plate_loo_all_favourable=all(loo_sign), n_pair_links=int(len(L)), n_primary_links=int(len(P_)), n_single_links=int(len(S)),
               n_sameplate_single=int(len(S6)), n_sameplate_pairs=int(len(S6p)), eligibility_counts=elig["paired_plate"])
    (out_man / "repeat_audit_receipt.json").write_text(json.dumps(rec, indent=1, default=float))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 60)
    print(T.drop(columns=["track", "version"]).to_string())
    print(per.to_string())


if __name__ == "__main__":
    main()
