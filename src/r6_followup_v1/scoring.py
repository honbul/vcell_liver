"""One evaluation set, many arms: total RMSE, within-pool contrast, retrieval, paired bootstrap.

An evaluation set is a list of conditions (line, drug group) with their pool and (for source
out-of-fold sets) their fold. Within-pool groups are (fold, pool); cell-line groups are
(fold, line), so candidates and centring never mix predictions of different fold models.
Every arm is added as a list of prediction arrays (training seeds x thinning seeds of one keep
rate); metrics are computed per array and averaged per condition BEFORE the drug-level bootstrap.
"""
from __future__ import annotations

import numpy as np

from . import metrics as M

RET_VARIANTS = {
    "line_all": "candidates: measured responses of all evaluated drugs of the same cell line (and fold); both sides centred by their own cell-line mean (the original R6 retrieval)",
    "pool_linecentred": "candidates: evaluated drugs of the same cell line AND control pool; representation as in line_all",
    "pool_centred": "candidates: evaluated drugs of the same control pool; both sides centred by their own pool mean",
}
RET_KEYS = ["top1", "topk", "mrr", "chance_top1", "chance_topk", "chance_mrr", "legacy_rank_score", "legacy_unique_top1", "n_candidates"]
DEF_TOTAL = "per-condition RMSE over 2000 genes -> mean within drug group -> unweighted mean over drug groups"
DEF_WITHIN = ("within-pool contrast RMSE: truth and prediction each centred by their own mean over the evaluated drugs of the "
              "same (fold, control pool), per-condition RMSE -> mean within drug group -> unweighted mean over drug groups; "
              "groups with <2 drugs contribute nothing")


class Scorer:
    def __init__(self, T, L, Dg, pool, fold=None, masks: dict | None = None, n_boot: int = 2000, seed: int = 0):
        self.T = np.asarray(T, np.float32)
        self.n = len(L)
        self.L, self.Dg, self.pool = np.asarray(L), np.asarray(Dg), np.asarray(pool)
        self.fold = np.zeros(self.n, int) if fold is None else np.asarray(fold)
        self.du, self.dinv = np.unique(self.Dg, return_inverse=True)
        self.nd = len(self.du)
        self.g_pool = M.groups_of(self.fold, self.pool)
        self.g_line = M.groups_of(self.fold, self.L)
        self.Mb = M.multiplicities(self.nd, n_boot, seed)
        self.masks = {"all": np.ones(self.n, bool)}
        self.masks.update(masks or {})
        self.A = {k: M.drug_indicator(self.dinv, self.nd, m.astype(np.float32)) for k, m in self.masks.items()}
        self.Tc_pool = M.centre_by(self.T, self.g_pool)
        self.Tc_line = M.centre_by(self.T, self.g_line)
        self.in_pair_group = np.zeros(self.n, bool)
        for idx in self.g_pool:
            if len(idx) >= 2:
                self.in_pair_group[idx] = True
        self.truth_rms = np.sqrt((self.T.astype(np.float64) ** 2).mean(1))
        self.truth_c_rms = np.where(self.in_pair_group, np.sqrt((self.Tc_pool ** 2).mean(1)), np.nan)
        self.arms: dict = {}

    # ---- arms ------------------------------------------------------------------------------------
    def add(self, name: str, preds: list, retrieval: bool = True) -> dict:
        nv = len(preds)
        v = np.zeros(self.n); lw = np.zeros(self.n); cm = np.zeros(self.n); pm = np.zeros(self.n)
        gl, pe, pc = [], 0.0, 0.0
        ret = {k: {q: np.zeros(self.n) for q in RET_KEYS} for k in RET_VARIANTS} if retrieval else {}
        for P in preds:
            P = np.asarray(P, np.float32)
            E = P.astype(np.float64) - self.T
            v += np.sqrt((E ** 2).mean(1)) / nv
            pm += np.sqrt((P.astype(np.float64) ** 2).mean(1)) / nv
            gr = M.grams(E, self.g_pool)
            gl.append(gr)
            li = M.within_point(gr, self.g_pool, self.n)
            lw += li / nv
            big = [K for K in gr if K.shape[0] >= 2]
            pe += float(np.mean([np.mean(li[idx]) for idx in self.g_pool if len(idx) >= 2])) / nv
            pc += float(np.mean([M.pair_contrast_from_gram(K) for K in big])) / nv
            Pc_pool = M.centre_by(P, self.g_pool)
            cm += np.sqrt((Pc_pool ** 2).mean(1)) / nv
            if retrieval:
                Pc_line = M.centre_by(P, self.g_line)
                for k, (a, b, g) in dict(line_all=(Pc_line, self.Tc_line, self.g_line),
                                         pool_linecentred=(Pc_line, self.Tc_line, self.g_pool),
                                         pool_centred=(Pc_pool, self.Tc_pool, self.g_pool)).items():
                    sc = M.retrieval_scores(*M.retrieval_counts(a, b, g))
                    for q in RET_KEYS:
                        ret[k][q] += sc[q] / nv
        Lacc, V = M.within_boot(gl, self.g_pool, self.dinv, self.Mb)
        arm = dict(v=v, lw=lw, cmag=cm, pmag=pm, ret=ret, pool_equal_within=pe, pair_contrast=pc, n_views=nv, bs={})
        for mk, A in self.A.items():
            arm["bs"][mk] = dict(total=M.macro_boot_fixed(v, A, self.Mb), within=M.macro_boot_varying(Lacc, V, A, self.Mb))
        self.arms[name] = arm
        return arm

    # ---- summaries -------------------------------------------------------------------------------
    def summary(self, name: str, mask: str = "all") -> dict:
        a, A = self.arms[name], self.A[mask]
        tot, nd, nu = M.macro_point(a["v"], A)
        wit, ndw, nuw = M.macro_point(a["lw"], A)
        lo, hi = M.ci(a["bs"][mask]["total"]); wlo, whi = M.ci(a["bs"][mask]["within"])
        ratio = np.where(self.truth_c_rms > 0, a["cmag"] / np.maximum(self.truth_c_rms, 1e-12), np.nan)
        out = dict(subset=mask, n_drugs=nd, n_units=nu, total_rmse=tot, total_lo=lo, total_hi=hi,
                   n_drugs_within=ndw, n_units_within=nuw, within_rmse=wit, within_lo=wlo, within_hi=whi,
                   centred_magnitude_ratio=M.macro_point(ratio, A)[0],
                   magnitude_ratio=M.macro_point(a["pmag"] / np.maximum(self.truth_rms, 1e-12), A)[0])
        if mask == "all":
            out.update(within_rmse_pool_equal=a["pool_equal_within"], pair_contrast_rmse_pool_equal=a["pair_contrast"],
                       n_pool_groups=int(sum(len(i) >= 2 for i in self.g_pool)), n_seed_views=a["n_views"])
        return out

    def pair(self, cand: str, comp: str, mask: str = "all") -> dict:
        a, b, A = self.arms[cand], self.arms[comp], self.A[mask]
        out = dict(subset=mask, candidate=cand, comparator=comp, sign="candidate - comparator (negative = candidate better)")
        for key, col in (("total", "v"), ("within", "lw")):
            pa, nd, nu = M.macro_point(a[col], A); pb, _, _ = M.macro_point(b[col], A)
            lo, hi = M.ci(a["bs"][mask][key] - b["bs"][mask][key])
            diff = pa - pb
            out.update({f"{key}_candidate": pa, f"{key}_comparator": pb, f"{key}_diff": diff, f"{key}_lo": lo, f"{key}_hi": hi,
                        f"{key}_rel": diff / pb if pb else float("nan"),
                        f"{key}_category": M.decision_category(diff, lo, hi, pb), f"n_drugs_{key}": nd, f"n_units_{key}": nu})
        return out

    def retrieval_rows(self, name: str) -> list[dict]:
        a, A, rows = self.arms[name], self.A["all"], []
        for k, desc in RET_VARIANTS.items():
            r = dict(candidate_pool=k, candidate_pool_definition=desc)
            for q in RET_KEYS:
                val, nd, nu = M.macro_point(a["ret"][k][q], A)
                r[q] = val
                r["n_drugs"], r["n_units"] = nd, nu
            for q in ("top1", "mrr"):
                r[f"{q}_lo"], r[f"{q}_hi"] = M.ci(M.macro_boot_fixed(a["ret"][k][q], A, self.Mb))
            r["top1_minus_chance"] = r["top1"] - r["chance_top1"]
            r["mrr_minus_chance"] = r["mrr"] - r["chance_mrr"]
            rows.append(r)
        return rows

    def retrieval_pair(self, cand: str, comp: str) -> list[dict]:
        a, b, A, rows = self.arms[cand], self.arms[comp], self.A["all"], []
        for k in RET_VARIANTS:
            r = dict(candidate_pool=k, candidate=cand, comparator=comp, sign="candidate - comparator (positive = candidate better)")
            for q in ("top1", "mrr"):
                d = a["ret"][k][q] - b["ret"][k][q]
                r[f"{q}_diff"] = M.macro_point(d, A)[0]
                r[f"{q}_lo"], r[f"{q}_hi"] = M.ci(M.macro_boot_fixed(d, A, self.Mb))
            rows.append(r)
        return rows
