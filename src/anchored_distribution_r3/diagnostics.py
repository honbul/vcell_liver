"""R3-0: is there distributional signal beyond a mean shift, and is the 64-cell sample stable?

Runs on TRAIN-role drugs only. Every comparison uses the same number of real cells on both sides
and the same non-negative restoration. "Oracle" rows use the observed treated mean and are a
diagnostic, not a deployable method. See reports/anchored_distribution_r3/DECISIONS.md (R3-0) for
the comparisons a..f and the pre-specified gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D
from .metrics import (Scorer, drug_block_bootstrap, drug_table, fit_pca_kmeans, restore_nonneg,
                      shift_to_mean)


def mm(scorer, src, tgt):
    """Mean-match src to tgt's observed mean (oracle), restore, score against tgt."""
    mu = tgt.mean(0)
    return scorer.ed(restore_nonneg(shift_to_mean(src, mu), mu), tgt)


def profile(t, c, eps=0.02):
    """Per-gene shape-change profiles of treated vs control cells (log1p units)."""
    lv = torch.log((t.var(0) + eps) / (c.var(0) + eps))
    dz = (t <= 0).float().mean(0) - (c <= 0).float().mean(0)
    return lv, dz


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/anchored_distribution_r3/tables")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    d = D.load()
    W = D.eval_wells(d, "train")
    scale = D.fit_gene_scale(d, W)
    pm, pc, km = fit_pca_kmeans(d, W, scale["sd_within"])
    S = Scorer(scale["sd_within"], pm, pc, {k: torch.as_tensor(v, device=dev) for k, v in km.items()}, dev)
    print(f"wells={len(W)} dropped_small={W.attrs['n_dropped_small']} device={dev} "
          f"setup={time.time()-t0:.0f}s", flush=True)

    # partners: another well of the same condition; another train drug on the same (line, plate)
    W["rank"] = [D.bhash(f"r3-partner|{g}") for g in W.gid]
    same = {k: g.sort_values("rank").index.tolist() for k, g in W.groupby(["line", "drug"])}
    plate_grp = {k: g.sort_values("rank").index.tolist() for k, g in W.groupby(["line", "plate_index"])}

    rows = []
    ctrl_cache = {}
    for i, r in enumerate(W.itertuples()):
        n = int(r.n_eval)
        T = S.t(d.cells(r, n))
        key = (r.plate_index, r.line)
        if key not in ctrl_cache:
            if len(ctrl_cache) > 64:
                ctrl_cache.clear()
            ctrl_cache[key] = S.t(d.ctrl_cells(*key))
        C = ctrl_cache[key][:n]
        muT = T.mean(0)
        rec = dict(gid=r.gid, line=r.line, drug=r.drug, plate_index=r.plate_index, n=n,
                   lib_t=float(np.median(d.lib[r.start:r.start + n])))
        rec["a"] = S.ed(C, T)
        shifted = shift_to_mean(C, muT)
        rec["b_raw"] = S.ed(shifted, T)
        resto = restore_nonneg(shifted, muT)
        sc = S.score(resto, T, line=r.line)
        rec["b"] = sc["ed"]; rec["b_pca50"] = sc["ed_pca50"]; rec["b_shape"] = sc["ed_shape"]
        rec["b_var_ratio"] = sc["var_ratio"]; rec["b_zero_mae"] = sc["zero_frac_mae"]
        rec["b_cluster_tv"] = sc.get("cluster_tv", np.nan); rec["b_cov_relfro"] = sc["cov_relfro"]
        rec["a_pca50"] = S.score(C, T)["ed_pca50"]
        rec["shift_neg_frac"] = float((shifted < 0).float().mean())
        rec["shift_mean_err_after"] = float((resto.mean(0) - muT).abs().max())
        # split-half (sampling stability only) and the same-size shifted control
        T1, T2 = T[0::2], T[1::2]
        h = min(len(T1), len(T2)); T1, T2 = T1[:h], T2[:h]
        rec["c_raw"] = S.ed(T1, T2); rec["c_mm"] = mm(S, T1, T2)
        rec["c_pca50"] = S.score(T1, T2)["ed_pca50"]
        rec["b_half"] = mm(S, C[:h], T2)
        rec["b_half1"] = mm(S, C[:h], T1)
        # independent well of the same condition
        peers = [j for j in same[(r.line, r.drug)] if j != i]
        if peers:
            q = W.iloc[peers[0]]
            m = min(n, int(q.n_eval))
            T_o = S.t(d.cells(q, m))
            rec["d_raw"] = S.ed(T_o, T[:m]); rec["d_mm"] = mm(S, T_o, T[:m])
            rec["d_same_plate"] = int(q.plate_index == r.plate_index)
            rec["d_pca50_mm"] = S.score(restore_nonneg(shift_to_mean(T_o, T[:m].mean(0)), T[:m].mean(0)), T[:m])["ed_pca50"]
            lvo, dzo = profile(T_o, S.t(d.ctrl_cells(q.plate_index, q.line))[:m])
        # another train drug, same line and plate ("shuffled")
        cand = [j for j in plate_grp[(r.line, r.plate_index)] if W.drug.iat[j] != r.drug]
        lv, dz = profile(T, C)
        if cand:
            j = cand[int(D.bhash(f"r3-other|{r.gid}") * len(cand))]
            q2 = W.iloc[j]
            m = min(n, int(q2.n_eval))
            T_x = S.t(d.cells(q2, m))
            rec["e_raw"] = S.ed(T_x, T[:m]); rec["e_mm"] = mm(S, T_x, T[:m])
            rec["e_pca50_mm"] = S.score(restore_nonneg(shift_to_mean(T_x, T[:m].mean(0)), T[:m].mean(0)), T[:m])["ed_pca50"]
            rec["e_drug"] = int(q2.drug)
            lvx, dzx = profile(T_x, C[:m])
            rec["lv_corr_other"] = corr(lv, lvx); rec["dz_corr_other"] = corr(dz, dzx)
            rec["lib_x"] = float(np.median(d.lib[q2.start:q2.start + m]))
        if peers:
            rec["lv_corr_same"] = corr(lv, lvo); rec["dz_corr_same"] = corr(dz, dzo)
        rec["lib_c"] = float(np.median(np.concatenate([d.lib[s:e] for s, e in d.ctrl_slices[key]])))
        rec["tot_var_ratio_tc"] = float((T / S.sd).var(0).sum() / (C / S.sd).var(0).sum())
        rec["zero_t"] = float((T <= 0).float().mean()); rec["zero_c"] = float((C <= 0).float().mean())
        rows.append(rec)
        if i % 1000 == 0:
            print(i, f"{time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "d1_wells_train.tsv.gz", sep="\t", index=False, float_format="%.5g")

    # control well vs control well on the same plate and line
    frows = []
    for (p, l), sl in d.ctrl_slices.items():
        if len(sl) < 2:
            continue
        A = S.t(np.asarray(d.X[sl[0][0]:sl[0][1]])); B = S.t(np.asarray(d.X[sl[1][0]:sl[1][1]]))
        m = min(len(A), len(B), D.N_CAP)
        if m < 32:
            continue
        frows.append(dict(plate_index=p, line=l, n=m, f_raw=S.ed(A[:m], B[:m]), f_mm=mm(S, A[:m], B[:m]),
                          f_split=S.ed(A[:m][0::2][:m // 2], A[:m][1::2][:m // 2])))
    F = pd.DataFrame(frows)
    F.to_csv(out / "d1_ctrl_wells.tsv", sep="\t", index=False, float_format="%.5g")

    # ---------------- summaries with drug-block bootstrap
    def boot(df, expr, cols):
        dt = drug_table(df.dropna(subset=cols), cols)
        return drug_block_bootstrap(dt, lambda t: float(expr(t)), B=a.boot)

    summ = dict(n_wells=int(len(R)), n_conditions=int(R.groupby(["line", "drug"]).ngroups),
                n_drugs=int(R.drug.nunique()), n_lines=int(R.line.nunique()),
                n_dropped_small=W.attrs["n_dropped_small"], n_eval=R.n.describe().to_dict(),
                scale=dict(floor_q=scale["floor_q"], n_cells=scale["n_cells"]))
    for c in ["a", "b_raw", "b", "c_raw", "c_mm", "b_half", "e_raw", "e_mm", "a_pca50", "b_pca50", "c_pca50",
              "b_shape", "b_cluster_tv", "b_zero_mae", "b_var_ratio", "tot_var_ratio_tc"]:
        summ[f"all.{c}"] = boot(R, lambda t, c=c: t[c].mean(), [c])
    summ["all.M_e_raw_minus_c_raw"] = boot(R, lambda t: (t.e_raw - t.c_raw).mean(), ["e_raw", "c_raw"])
    summ["all.b_half_minus_c_mm"] = boot(R, lambda t: (t.b_half - t.c_mm).mean(), ["b_half", "c_mm"])
    summ["all.e_mm_minus_c_mm"] = boot(R, lambda t: (t.e_mm - t.c_mm).mean(), ["e_mm", "c_mm"])
    summ["all.frac_wells_e_raw_gt_c_raw"] = float((R.e_raw > R.c_raw).mean())
    Mw = R.dropna(subset=["d_mm"])
    summ["multi.n_wells"] = int(len(Mw)); summ["multi.n_drugs"] = int(Mw.drug.nunique())
    summ["multi.frac_same_plate"] = float(Mw.d_same_plate.mean())
    for c in ["a", "b", "d_raw", "d_mm", "e_raw", "e_mm", "c_mm", "b_pca50", "d_pca50_mm", "e_pca50_mm",
              "lv_corr_same", "lv_corr_other", "dz_corr_same", "dz_corr_other"]:
        summ[f"multi.{c}"] = boot(Mw, lambda t, c=c: t[c].mean(), [c])
    summ["multi.G1_H_b_minus_d_mm"] = boot(Mw, lambda t: (t.b - t.d_mm).mean(), ["b", "d_mm"])
    summ["multi.G1_ratio_H_over_a_minus_d_mm"] = boot(
        Mw, lambda t: (t.b - t.d_mm).mean() / (t.a - t.d_mm).mean(), ["a", "b", "d_mm"])
    summ["multi.G2_e_mm_minus_d_mm"] = boot(Mw, lambda t: (t.e_mm - t.d_mm).mean(), ["e_mm", "d_mm"])
    summ["multi.G2_pca50_e_mm_minus_d_mm"] = boot(Mw, lambda t: (t.e_pca50_mm - t.d_pca50_mm).mean(),
                                                  ["e_pca50_mm", "d_pca50_mm"])
    summ["multi.G1_pca50"] = boot(Mw, lambda t: (t.b_pca50 - t.d_pca50_mm).mean(), ["b_pca50", "d_pca50_mm"])
    summ["multi.lv_corr_same_minus_other"] = boot(Mw, lambda t: (t.lv_corr_same - t.lv_corr_other).mean(),
                                                  ["lv_corr_same", "lv_corr_other"])
    summ["multi.dz_corr_same_minus_other"] = boot(Mw, lambda t: (t.dz_corr_same - t.dz_corr_other).mean(),
                                                  ["dz_corr_same", "dz_corr_other"])
    for sp in [0, 1]:
        sub = Mw[Mw.d_same_plate == sp]
        if sub.drug.nunique() >= 5:
            summ[f"multi.same_plate{sp}.n_wells"] = int(len(sub))
            summ[f"multi.same_plate{sp}.H"] = boot(sub, lambda t: (t.b - t.d_mm).mean(), ["b", "d_mm"])
            summ[f"multi.same_plate{sp}.G2"] = boot(sub, lambda t: (t.e_mm - t.d_mm).mean(), ["e_mm", "d_mm"])
    summ["ctrl.f_raw_mean"] = float(F.f_raw.mean()); summ["ctrl.f_mm_mean"] = float(F.f_mm.mean())
    summ["ctrl.f_split_mean"] = float(F.f_split.mean()); summ["ctrl.n_pairs"] = int(len(F))
    # sample stability: N/2 vs N on the same wells, and noise of the aggregate
    cond = R.groupby(["line", "drug"])[["b", "b_half", "b_half1", "c_mm"]].mean()
    summ["stability.cond_sd_b64"] = float(cond.b.std())
    summ["stability.cond_sd_halfdiff"] = float((cond.b_half - cond.b_half1).std())
    summ["stability.cond_corr_half_vs_half"] = float(np.corrcoef(cond.b_half, cond.b_half1)[0, 1])
    summ["stability.cond_sd_c_mm_n32"] = float(cond.c_mm.std())
    bb = summ["all.b"]; summ["stability.boot_se_all_b"] = (bb["hi"] - bb["lo"]) / 3.92
    # library size / plate
    R["dlib"] = np.log(R.lib_t / R.lib_c)
    summ["lib.corr_b_absdlib"] = float(np.corrcoef(R.b, R.dlib.abs())[0, 1])
    summ["lib.corr_a_absdlib"] = float(np.corrcoef(R.a, R.dlib.abs())[0, 1])
    summ["lib.corr_totvar_dlib"] = float(np.corrcoef(np.log(R.tot_var_ratio_tc), R.dlib)[0, 1])
    summ["lib.dlib_quantiles"] = R.dlib.quantile([.05, .25, .5, .75, .95]).to_dict()
    summ["plate.mean_by_plate"] = R.groupby("plate_index")[["a", "b", "c_mm", "dlib"]].mean().to_dict()
    summ["dev_lines"] = [int(x) for x in d.dev_lines]
    summ["wall_s"] = time.time() - t0
    D.dump_json(summ, out / "d1_summary.json")
    print(json.dumps(summ, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
