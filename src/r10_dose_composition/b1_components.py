"""B1: K selection, replicate-well component diagnostics, mean-change decomposition and
synthetic positive controls of the evaluator. TRAIN drugs (and vehicle wells) only.

Outputs in --out (a run directory under assets/runs/):
  K_selection.tsv, atlas_K.json         per-line K from the pre-fixed rule (config atlas.K_rule)
  replicate_components_pairs.tsv.gz     one row per well pair x subsample size
  decomposition.tsv.gz                  per train condition: mean change vs control split into
                                        composition and within-state terms (control and symmetric reference)
  synthetic_controls.tsv                positive controls (composition-only / within-only / both)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .core import LINES, Cloud, LineData, Query, cloud_stat, energy_distance, score, state_moments
import torch

DOSES = (10.0, 100.0, 1000.0, 10000.0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def k_selection(cfg, split, out):
    rows, chosen = [], {}
    nmin = cfg["min_cells_well"]
    train = set(split[split.role == "train"].drug)
    for line in LINES:
        info = json.load(open(f"{cfg['cache_dir']}/atlas_{line}/fit_info.json"))
        res = {}
        for K in sorted(set(cfg["atlas"]["K_candidates"]) | set(cfg["atlas"]["K_sensitivity"])):
            D = LineData(cfg, line, K)
            P1, P2 = [], []
            for drug in train:
                for dose in DOSES:
                    a, b = D.cells(drug, dose, "rep1"), D.cells(drug, dose, "rep2")
                    if a is None or b is None or len(a) < nmin or len(b) < nmin:
                        continue
                    P1.append(D.S[a].mean(0)); P2.append(D.S[b].mean(0))
            P1, P2 = np.array(P1), np.array(P2)
            occ = np.array(info["gmm"][str(K)]["fit_occupancy"])
            r = np.array([np.corrcoef(P1[:, k], P2[:, k])[0, 1] for k in range(K)])
            sc = float((occ * r).sum() / occ.sum())
            rep_tv = float(0.5 * np.abs(P1 - P2).sum(1).mean())
            res[K] = sc
            rows.append(dict(line=line, K=K, candidate=K in cfg["atlas"]["K_candidates"], n_conditions=len(P1),
                             score_weighted_rep_corr=round(sc, 4), per_state_rep_corr=";".join(f"{x:.3f}" for x in r),
                             fit_occupancy=";".join(f"{x:.3f}" for x in occ), min_occupancy=float(occ.min()),
                             mean_rep_tv=round(rep_tv, 4),
                             mean_max_membership=info["gmm"][str(K)]["mean_max_membership"]))
        cand = [K for K in cfg["atlas"]["K_candidates"]
                if min(json.load(open(f"{cfg['cache_dir']}/atlas_{line}/fit_info.json"))["gmm"][str(K)]["fit_occupancy"]) >= 0.02]
        cand = sorted(cand)
        best = max(cand, key=lambda k: res[k])
        if best != cand[0] and res[best] - res[cand[0]] < 0.02:
            best = cand[0]
        chosen[line] = int(best)
    pd.DataFrame(rows).to_csv(f"{out}/K_selection.tsv", sep="\t", index=False)
    json.dump(chosen, open(f"{out}/atlas_K.json", "w"))
    log("K chosen", chosen)
    return chosen


def sub(idx, n, rng):
    return idx if n is None or len(idx) <= n else rng.choice(idx, n, replace=False)


def pair_rows(D: LineData, line, A, B, kind, meta, rng, sizes=(None,)):
    rows = []
    for n in sizes:
        if n is not None and (len(A) < n or len(B) < n):
            continue
        a, b = sub(A, n, rng), sub(B, n, rng)
        P = Cloud.uniform(D.Z[a])
        Q = Query("", D.Z[b], D.S[b], D.G[0] * np.nan)
        s = score(D, P, Q)
        r = dict(line=line, K=D.K, pair_type=kind, n_sub="full" if n is None else n, n_a=len(a), n_b=len(b), **meta)
        r.update({k: s[k] for k in ("ed", "mmd", "comp_tv", "comp_js", "mean_rmse_latent", "within_mean_rmse",
                                    "within_ed", "within_n_states", "within_mass_scored")})
        rows.append(r)
    return rows


def replicate_components(cfg, split, K_by_line, out, seed=0):
    nmin = cfg["min_cells_well"]
    train = sorted(split[split.role == "train"].drug)
    rows = []
    for line in LINES:
        D = LineData(cfg, line, K_by_line[line])
        rng = np.random.default_rng(seed)
        ok = {}
        for drug in train:
            for dose in DOSES:
                for rep in ("rep1", "rep2"):
                    c = D.cells(drug, dose, rep)
                    if c is not None and len(c) >= nmin:
                        ok[(drug, dose, rep)] = c
        for drug in train:
            for dose in DOSES:
                if (drug, dose, "rep1") in ok and (drug, dose, "rep2") in ok:
                    rows += pair_rows(D, line, ok[(drug, dose, "rep1")], ok[(drug, dose, "rep2")], "replicate_same_condition",
                                      dict(drug=drug, dose_a=dose, dose_b=dose, rep_a="rep1", rep_b="rep2"), rng,
                                      sizes=(None, 32, 64, 128))
                for rep in ("rep1", "rep2"):
                    i = DOSES.index(dose)
                    if i + 1 < len(DOSES) and (drug, dose, rep) in ok and (drug, DOSES[i + 1], rep) in ok:
                        rows += pair_rows(D, line, ok[(drug, dose, rep)], ok[(drug, DOSES[i + 1], rep)],
                                          "adjacent_dose_same_rep",
                                          dict(drug=drug, dose_a=dose, dose_b=DOSES[i + 1], rep_a=rep, rep_b=rep), rng)
                    if (drug, dose, rep) in ok:
                        others = [d2 for d2 in train if d2 != drug and (d2, dose, rep) in ok]
                        if others:
                            d2 = others[int(rng.integers(len(others)))]
                            rows += pair_rows(D, line, ok[(drug, dose, rep)], ok[(d2, dose, rep)],
                                              "other_drug_same_dose_same_rep",
                                              dict(drug=drug, drug_b=d2, dose_a=dose, dose_b=dose, rep_a=rep, rep_b=rep),
                                              rng)
            # response strength: condition vs pooled vehicle of the same replicate
            for dose in DOSES:
                for rep in ("rep1", "rep2"):
                    if (drug, dose, rep) in ok:
                        rows += pair_rows(D, line, ok[(drug, dose, rep)], D.ctrl[rep], "condition_vs_control",
                                          dict(drug=drug, dose_a=dose, dose_b=0.0, rep_a=rep, rep_b=rep), rng)
        w = D.wells
        veh = {rep: list(w[(w.dose == 0) & (w.replicate == rep)].Combo) for rep in ("rep1", "rep2")}
        plate = D.obs.drop_duplicates("Combo").set_index("Combo").culture_plate
        for rep in ("rep1", "rep2"):
            v = veh[rep]
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    rows += pair_rows(D, line, D.idx[v[i]], D.idx[v[j]],
                                      "vehicle_same_rep_same_plate" if plate[v[i]] == plate[v[j]] else "vehicle_same_rep_other_plate",
                                      dict(drug="vehicle", dose_a=0.0, dose_b=0.0, rep_a=rep, rep_b=rep), rng,
                                      sizes=(None, 32, 64, 128))
        for a, b in zip(veh["rep1"], veh["rep2"]):
            rows += pair_rows(D, line, D.idx[a], D.idx[b], "vehicle_cross_rep",
                              dict(drug="vehicle", dose_a=0.0, dose_b=0.0, rep_a="rep1", rep_b="rep2"), rng,
                              sizes=(None, 32, 64, 128))
        log(line, "pairs", len(rows))
    t = pd.DataFrame(rows)
    t.to_csv(f"{out}/replicate_components_pairs.tsv.gz", sep="\t", index=False)
    return t


def decomposition(cfg, split, K_by_line, out):
    """mu_Q - mu_C = sum_k (pi_Qk - pi_Ck) m_Ck + sum_k pi_Qk (m_Qk - m_Ck)  (control reference)
    and the symmetric version with averaged reference. Norms are RMS over the d PCA axes."""
    rows = []
    sh = cfg["shrink"]
    train = sorted(split[split.role == "train"].drug)
    for line in LINES:
        D = LineData(cfg, line, K_by_line[line])
        for rep in ("rep1", "rep2"):
            C = state_moments(D.Z[D.ctrl[rep]], D.S[D.ctrl[rep]], D.atlas, sh["kappa_mean"], sh["kappa_cov"])
            for drug in train:
                for dose in DOSES:
                    q = D.cells(drug, dose, rep)
                    if q is None or len(q) < cfg["min_cells_well"]:
                        continue
                    Qm = state_moments(D.Z[q], D.S[q], D.atlas, sh["kappa_mean"], sh["kappa_cov"])
                    mC, mQ = C["m_shrunk"], Qm["m_shrunk"]
                    dpi = Qm["pi"] - C["pi"]
                    comp = dpi @ mC
                    within = Qm["pi"] @ (mQ - mC)
                    comp_s = dpi @ (0.5 * (mC + mQ))
                    within_s = (0.5 * (Qm["pi"] + C["pi"])) @ (mQ - mC)
                    total = D.Z[q].mean(0) - D.Z[D.ctrl[rep]].mean(0)
                    rms = lambda v: float(np.sqrt((v ** 2).mean()))
                    rows.append(dict(line=line, drug=drug, dose=dose, rep=rep, n=len(q), K=D.K,
                                     dmean_rms=rms(total), comp_term_rms=rms(comp), within_term_rms=rms(within),
                                     comp_term_sym_rms=rms(comp_s), within_term_sym_rms=rms(within_s),
                                     resid_rms=rms(total - comp - within),
                                     comp_share_sym=float((comp_s @ total) / max(total @ total, 1e-12)),
                                     comp_tv_vs_control=float(0.5 * np.abs(dpi).sum())))
    t = pd.DataFrame(rows)
    t.to_csv(f"{out}/decomposition.tsv.gz", sep="\t", index=False)
    return t


def synthetic_controls(cfg, K_by_line, out, seed=0):
    """Vehicle rep1 cells = 'prediction'; vehicle rep2 cells, transformed, = 'truth'.
    composition-only: draw 400 truth cells with exact hard-state counts after moving a mass delta
    from the largest to the second-largest state (realisable without replacement); within-only: shift every truth cell
    by sum_k s_k(z) * delta * sd_k * u_k with fixed random unit directions u_k; both: the two together.
    The oracle-repair rows (oracle_pi uses the realised re-read truth composition) apply the true change to the prediction through the same assembly rules
    as the 2x2 heads (reweighting by pi ratios / shifting by state residuals)."""
    rows = []
    rng = np.random.default_rng(seed)
    for line in LINES:
        D = LineData(cfg, line, K_by_line[line])
        A, B = D.ctrl["rep1"], D.ctrl["rep2"]
        ZA, SA = D.Z[A], D.S[A]
        # R3-estimator equivalence check (uniform single group)
        s1 = cloud_stat(Cloud.uniform(ZA[:300]), Cloud.uniform(D.Z[B][:300]), "ed")
        s2 = energy_distance(torch.as_tensor(ZA[:300]), torch.as_tensor(D.Z[B][:300]))
        rows.append(dict(line=line, case="check_ed_equals_R3_estimator", level=0, value_a=s1, value_b=s2,
                         abs_diff=abs(s1 - s2)))
        rr = float(np.abs(D.atlas.read(ZA) - SA).max())
        rows.append(dict(line=line, case="check_numpy_reread_equals_sklearn_posterior", level=0, value_a=rr,
                         value_b=0.0, abs_diff=rr))
        piB = D.S[B].mean(0)
        K, d = D.K, D.Z.shape[1]
        u = rng.normal(size=(K, d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        sdk = np.sqrt(np.trace(D.atlas.cov, axis1=1, axis2=2) / d)
        hard = D.S[B].argmax(1)
        occ = np.bincount(hard, minlength=K)
        big, second = np.argsort(-occ)[:2]
        m = 400
        for case in ("null", "comp_only", "within_only", "both"):
            for lev in ((0,) if case == "null" else (1, 2)):
                dc = {0: 0.0, 1: 0.10, 2: 0.25}[lev] if case in ("comp_only", "both") else 0.0
                dw = {0: 0.0, 1: 0.5, 2: 1.0}[lev] if case in ("within_only", "both") else 0.0
                # exact hard-state counts: move dc of the mass from the largest to the 2nd-largest state
                tgt = occ / occ.sum()
                tgt = tgt + dc * (np.eye(K)[second] - np.eye(K)[big])
                cnt = np.floor(tgt * m).astype(int)
                cnt[big] += m - cnt.sum()
                assert (cnt <= occ).all(), "synthetic target not realisable"
                pick = np.concatenate([rng.choice(np.where(hard == k)[0], cnt[k], replace=False)
                                       for k in range(K) if cnt[k] > 0])
                ZQ = D.Z[B][pick] + D.S[B][pick] @ (dw * sdk[:, None] * u)
                SQ = D.atlas.read(ZQ)
                Q = Query("", ZQ, SQ, np.zeros(D.G.shape[1]))
                piT = SQ.mean(0)                      # realised truth composition (re-read)
                for repair in ("none", "oracle_pi", "oracle_shift", "oracle_both"):
                    w = np.full(len(ZA), 1 / len(ZA))
                    X = ZA.copy()
                    if repair in ("oracle_pi", "oracle_both") and dc > 0:
                        w = w * (SA @ (piT / SA.mean(0)))
                        w /= w.sum()
                    if repair in ("oracle_shift", "oracle_both") and dw > 0:
                        X = X + SA @ (dw * sdk[:, None] * u)
                    s = score(D, Cloud(X, w, np.zeros(len(X), int)), Q)
                    rows.append(dict(line=line, case=case, level=lev, delta_comp=dc, delta_within_sd=dw,
                                     repair=repair, n_pred=len(ZA), n_truth=m,
                                     true_tv_shift=float(0.5 * np.abs(piT - piB).sum()),
                                     **{k: s[k] for k in ("comp_tv", "within_mean_rmse", "within_ed", "ed",
                                                          "mean_rmse_latent")}))
        log(line, "synthetic done")
    t = pd.DataFrame(rows)
    t.to_csv(f"{out}/synthetic_controls.tsv", sep="\t", index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    assert not os.path.exists(f"{args.out}/DONE"), "immutable run id: output exists"
    os.makedirs(args.out, exist_ok=True)
    split = pd.read_csv(args.split, sep="\t")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    K = k_selection(cfg, split, args.out)
    synthetic_controls(cfg, K, args.out)
    decomposition(cfg, split, K, args.out)
    replicate_components(cfg, split, K, args.out)
    open(f"{args.out}/DONE", "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))


if __name__ == "__main__":
    main()
