"""Positive-control calibration of the within-pool metrics (plan 5.4, R6-A4). Nothing is trained.

Question: can the within-pool contrast RMSE and the within-pool retrieval detect a prediction that
really carries drug-specific signal? Material: barcode-disjoint halves of the treated wells and of
the reference controls (build_halves.py), TRAIN drugs only. Every TRAIN drug is scored as a member
of its own source fold (groups = (fold, pool)), i.e. exactly the units of the out-of-fold tables.

    target_B            response from treated half B minus reference half B  (the scored "truth")
    oracle_A_disjoint   response from treated half A minus reference half A  (no cell shared)
    oracle_A_sharedctrl treated half A minus the FULL reference, scored against treated half B minus
                        the FULL reference (the control term is shared: shows how much a shared
                        control inflates agreement; not a valid specificity estimate)
    mismatched_A        oracle_A_disjoint of ANOTHER drug of the same (fold, pool) (cyclic shift):
                        carries the pool-shared part but the wrong drug
    poolmean_A_oof      shrunk pool mean of half-A responses of the fold's fitting drugs (kappa of
                        Phase A): a pool-constant prediction
    replicate wells     conditions measured on two plates: full-well response on one plate against
                        the other (the only replicate-like units; a different plate is also a
                        different control pool, so this mixes replicate noise and plate effects)

A split half is a technical split of one well, not an independent culture replicate: it bounds
what measurement noise alone allows, not biological reproducibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import metrics as M
from .baselines import MeanBaseline
from .scoring import DEF_TOTAL, DEF_WITHIN, Scorer

MIN_HALF = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/r6_followup_v1/positive_control_calibration.tsv")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    d = C.load(eval_views=False)
    root = C.assets() / "cache" / "halves_v1"
    w = pd.read_csv(root / "wells.tsv", sep="\t")
    tm = np.load(root / "treated_halves.npy").astype(np.float64)      # [A, B, full] x wells x G
    rf = np.load(root / "ref_halves.npy").astype(np.float64)          # [A, B, full] x pools x G
    kappa = json.loads(Path("manifests/r6_followup_v1/kappa_selection.json").read_text())["chosen_kappa"]
    w = w[w.valid].copy() if "valid" in w else w
    w["pool"] = d.pool_id[w.plate_index.to_numpy(), w.line.to_numpy()]
    w = w[w.pool >= 0]
    nL, nD, G = d.Y.shape
    resp = {k: np.zeros((nL, nD, G), np.float32) for k in ("A", "B", "A_shared", "B_shared", "full")}
    ok = np.zeros((nL, nD), bool)
    n_dropped = 0
    for (l, g), ww in w.groupby(["line", "drug"]):
        if not d.valid[l, g]:
            continue
        if (ww.n_A < MIN_HALF).any() or (ww.n_B < MIN_HALF).any():
            n_dropped += 1
            continue
        wi, pi = ww.wid.to_numpy(), ww.pool.to_numpy()
        resp["A"][l, g] = (tm[0, wi] - rf[0, pi]).mean(0)
        resp["B"][l, g] = (tm[1, wi] - rf[1, pi]).mean(0)
        resp["A_shared"][l, g] = (tm[0, wi] - rf[2, pi]).mean(0)
        resp["B_shared"][l, g] = (tm[1, wi] - rf[2, pi]).mean(0)
        resp["full"][l, g] = (tm[2, wi] - rf[2, pi]).mean(0)
        ok[l, g] = True
    tr = np.asarray(C.train_drugs(d))
    L, Dg = np.nonzero(ok)
    keep = np.isin(Dg, tr)
    L, Dg = L[keep], Dg[keep]
    pool, fold = C.pool_of(d, L, Dg), d.fold[Dg]
    chk = float(np.abs(resp["full"][L, Dg] - d.Y[L, Dg]).max())
    chk_med = float(np.median(np.abs(resp["full"][L, Dg] - d.Y[L, Dg]).max(1)))
    print(f"conditions {len(L)}, drugs {len(np.unique(Dg))}, dropped for < {MIN_HALF} cells per half: {n_dropped}; "
          f"full-well response vs R6 truth: max abs diff {chk:.2e}, median of per-condition max {chk_med:.2e}", flush=True)

    # pool-constant prediction from half-A responses of each fold's fitting drugs
    pm = np.zeros((len(L), G), np.float32)
    for k in C.FOLDS:
        q = fold == k
        f = ~q
        mb = MeanBaseline(resp["A"], L[f], Dg[f], pool[f], nL, len(d.pools))
        pm[q] = mb.predict(L[q], Dg[q], pool[q], kappa)[0]
    groups = M.groups_of(fold, pool)
    mis = np.zeros((len(L), G), np.float32); mis_ok = np.zeros(len(L), bool)
    A = resp["A"][L, Dg]
    for idx in groups:
        if len(idx) >= 2:
            o = idx[np.argsort(Dg[idx])]
            mis[o] = A[np.roll(o, 1)]; mis_ok[o] = True

    rows = []

    def emit(sc: Scorer, target: str, arms: list[str], pairs: list[tuple[str, str]]):
        for name in arms:
            s = sc.summary(name)
            ret = {r["candidate_pool"]: r for r in sc.retrieval_rows(name)}
            rows.append(dict(C.std_fields(split_role="source_heldfold_units_positive_control"), row_type="arm", target=target, arm=name, comparator="",
                             n_drugs=s["n_drugs"], n_units=s["n_units"], total_rmse=s["total_rmse"], total_lo=s["total_lo"], total_hi=s["total_hi"],
                             within_rmse=s["within_rmse"], within_lo=s["within_lo"], within_hi=s["within_hi"],
                             centred_magnitude_ratio=s["centred_magnitude_ratio"],
                             pool_top1=ret["pool_centred"]["top1"], pool_top1_lo=ret["pool_centred"]["top1_lo"], pool_top1_hi=ret["pool_centred"]["top1_hi"],
                             pool_mrr=ret["pool_centred"]["mrr"], pool_chance_top1=ret["pool_centred"]["chance_top1"],
                             pool_chance_mrr=ret["pool_centred"]["chance_mrr"], pool_n_candidates=ret["pool_centred"]["n_candidates"],
                             line_top1=ret["line_all"]["top1"], line_chance_top1=ret["line_all"]["chance_top1"]))
        for c, b in pairs:
            p = sc.pair(c, b)
            rp = {r["candidate_pool"]: r for r in sc.retrieval_pair(c, b)}["pool_centred"]
            rows.append(dict(C.std_fields(split_role="source_heldfold_units_positive_control"), row_type="pair", target=target, arm=c, comparator=b,
                             n_drugs=p["n_drugs_total"], n_units=p["n_units_total"], total_rmse=p["total_diff"], total_lo=p["total_lo"], total_hi=p["total_hi"],
                             within_rmse=p["within_diff"], within_lo=p["within_lo"], within_hi=p["within_hi"],
                             total_rel=p["total_rel"], within_rel=p["within_rel"], total_category=p["total_category"], within_category=p["within_category"],
                             pool_top1=rp["top1_diff"], pool_top1_lo=rp["top1_lo"], pool_top1_hi=rp["top1_hi"], pool_mrr=rp["mrr_diff"]))

    sc = Scorer(resp["B"][L, Dg], L, Dg, pool, fold, n_boot=a.n_boot, seed=0)
    sc.add("oracle_A_disjoint", [A]); sc.add("poolmean_A_oof", [pm]); sc.add("mismatched_A", [np.where(mis_ok[:, None], mis, pm)])
    sc.add("full_truth_R6", [d.Y[L, Dg]])
    emit(sc, "half_B_disjoint_controls", ["poolmean_A_oof", "oracle_A_disjoint", "mismatched_A", "full_truth_R6"],
         [("oracle_A_disjoint", "poolmean_A_oof"), ("mismatched_A", "poolmean_A_oof"), ("oracle_A_disjoint", "mismatched_A")])
    within_oracle = sc.summary("oracle_A_disjoint")["within_rmse"]; within_const = sc.summary("poolmean_A_oof")["within_rmse"]

    sc2 = Scorer(resp["B_shared"][L, Dg], L, Dg, pool, fold, n_boot=a.n_boot, seed=0)
    sc2.add("oracle_A_sharedctrl", [resp["A_shared"][L, Dg]]); sc2.add("poolmean_A_oof", [pm])
    emit(sc2, "half_B_shared_full_control", ["poolmean_A_oof", "oracle_A_sharedctrl"], [("oracle_A_sharedctrl", "poolmean_A_oof")])

    # the full-well truth scored with a pool-constant prediction, same units: the level every R6 model sat at
    sc3 = Scorer(d.Y[L, Dg], L, Dg, pool, fold, n_boot=a.n_boot, seed=0)
    pmf = np.zeros((len(L), G), np.float32)
    for k in C.FOLDS:
        q = fold == k
        mb = MeanBaseline(d.Y, L[~q], Dg[~q], pool[~q], nL, len(d.pools))
        pmf[q] = mb.predict(L[q], Dg[q], pool[q], kappa)[0]
    sc3.add("poolmean_full_oof", [pmf]); sc3.add("oracle_halfA_disjoint", [A]); sc3.add("oracle_halfB_disjoint", [resp["B"][L, Dg]])
    emit(sc3, "full_well_truth_R6", ["poolmean_full_oof", "oracle_halfA_disjoint", "oracle_halfB_disjoint"], [("oracle_halfA_disjoint", "poolmean_full_oof")])
    within_full = sc3.summary("poolmean_full_oof")["within_rmse"]

    # derived noise estimates (assume the two halves' errors are independent and equal-sized; an estimate, not a measurement)
    s_half = within_oracle / np.sqrt(2.0)
    s_full = within_oracle / 2.0
    sig = float(np.sqrt(max(within_full ** 2 - s_full ** 2, 0.0)))
    for k, v, note in (("est_half_noise_rms_within", s_half, "oracle within-pool RMSE / sqrt(2)"),
                       ("est_full_well_noise_rms_within", s_full, "oracle within-pool RMSE / 2"),
                       ("est_signal_rms_within", sig, "sqrt(within RMS of the full truth^2 - full-well noise^2)"),
                       ("est_best_within_rmse_vs_full_truth", s_full, "within-pool RMSE of a noise-free prediction against the full-well truth"),
                       ("est_max_rel_gain_within", s_full / within_full - 1.0, "relative change of the within-pool RMSE a noise-free prediction would give (negative = better)"),
                       ("pool_constant_within_vs_halfB", within_const, "reference level for the oracle rows")):
        rows.append(dict(C.std_fields(split_role="source_heldfold_units_positive_control"), row_type="derived_estimate", target="noise_model", arm=k, comparator="",
                         within_rmse=v, note=note))

    # ---- replicate-like wells: the same (line, drug) on two plates ----------------------------------
    wr = w[w.drug.isin(tr) & (w.n_full >= 2 * MIN_HALF)].copy()
    Rw = (tm[2, wr.wid.to_numpy()] - rf[2, wr.pool.to_numpy()])
    RA = (tm[0, wr.wid.to_numpy()] - rf[0, wr.pool.to_numpy()]); RB = (tm[1, wr.wid.to_numpy()] - rf[1, wr.pool.to_numpy()])
    wr["row"] = np.arange(len(wr))
    cen = {p_: Rw[g_.row.to_numpy()].mean(0) for p_, g_ in wr.groupby("pool") if len(g_) >= 5}
    recs = []
    for (l, g), ww in wr.groupby(["line", "drug"]):
        ww = ww.drop_duplicates("plate_index").sort_values("plate_index")
        if len(ww) < 2 or not all(p_ in cen for p_ in ww.pool):
            continue
        r1, r2 = ww.iloc[0], ww.iloc[1]
        x1, x2 = Rw[r1.row], Rw[r2.row]
        c1, c2 = x1 - cen[r1.pool], x2 - cen[r2.pool]
        ha, hb = RA[r1.row] - cen[r1.pool], RB[r1.row] - cen[r1.pool]
        cos = lambda u, v: float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum()))
        recs.append(dict(line=l, drug=g, rep_total=np.sqrt(((x1 - x2) ** 2).mean()), rep_centred=np.sqrt(((c1 - c2) ** 2).mean()),
                         zero_centred=np.sqrt((c1 ** 2).mean()), half_centred_same_well=np.sqrt(((ha - hb) ** 2).mean()),
                         rep_centred_cosine=cos(c1, c2), half_centred_cosine=cos(ha, hb), rep_raw_cosine=cos(x1, x2),
                         centred_rms_plate1=np.sqrt((c1 ** 2).mean()), centred_rms_plate2=np.sqrt((c2 ** 2).mean()),
                         n_cells_plate1=r1.n_full, n_cells_plate2=r2.n_full))
    rp = pd.DataFrame(recs)
    if len(rp):
        per = rp.groupby("drug").mean(numeric_only=True).drop(columns=["line"])
        per.insert(0, "n_lines", rp.groupby("drug").size())
        per = per.reset_index().rename(columns={"drug": "drug_group"})
        for k_, v_ in C.std_fields(split_role="source_train_replicate_wells").items():
            per.insert(0, k_, v_)
        per.to_csv(Path(a.out).with_name("positive_control_replicate_wells.tsv"), sep="\t", index=False, float_format="%.4f")
        mac = per.drop(columns=["track", "version", "split_role", "drug_group", "n_lines", "n_cells_plate1", "n_cells_plate2"]).mean()
        for k, v in mac.items():
            rows.append(dict(C.std_fields(split_role="source_train_replicate_wells"), row_type="replicate_wells", target="other_plate_full_well", arm=k, comparator="",
                             n_drugs=int(rp.drug.nunique()), n_units=int(len(rp)), within_rmse=v,
                             note="drug-macro mean; centred = minus the mean full-well response of all TRAIN-drug wells of that (plate, line) pool; "
                                  "rep = plate 1 vs plate 2 full wells; half = halves A vs B of the plate-1 well; cosine rows are cosines, not RMSE"))
    out = pd.DataFrame(rows)
    out["metric_definition"] = f"total: {DEF_TOTAL} | within: {DEF_WITHIN} | retrieval: tie-corrected, candidates = drugs of the same (fold, pool), pool-centred"
    out["sign"] = np.where(out.row_type == "pair", "candidate - comparator", "")
    out.to_csv(a.out, sep="\t", index=False, float_format="%.6f")
    C.dump_json(dict(n_conditions=int(len(L)), n_drugs=int(len(np.unique(Dg))), n_dropped_small_halves=n_dropped, min_cells_per_half=MIN_HALF, kappa=kappa,
                     full_response_vs_truth_max_abs=chk, full_response_vs_truth_median_of_max=chk_med, n_replicate_conditions=int(len(rp)),
                     n_replicate_drugs=int(rp.drug.nunique()) if len(rp) else 0, commit=C.git_commit(),
                     unit_note="split halves are technical splits of one well; replicate wells are on different plates"),
                "manifests/r6_followup_v1/positive_control_receipt.json")
    pd.set_option("display.width", 250)
    print(out.drop(columns=["metric_definition", "track", "version", "split_role", "sign"]).to_string())


if __name__ == "__main__":
    main()
