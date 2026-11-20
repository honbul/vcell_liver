"""Phase A (no retraining): pool baselines, within-pool contrast, pool retrieval on EXISTING predictions.

Three evaluation sets, never mixed:
    source_oof_baselines            the three source folds; only the response-mean baselines (no network
                                    was trained per fold yet). kappa of the shrunk pool mean is chosen here.
    source_dev_fold2_existing       original R6 development run (fit folds 0+1, evaluate fold 2, seed 42);
                                    its settings were selected on this very fold (selection data, not a test)
    existing_test_diagnostic        the 70 already-scored TEST drugs with the 24 final checkpoints'
                                    stored predictions; cause diagnosis only, nothing is selected here

Writes reports/r6_followup_v1/{pool_contract,pool_baselines,single_vs_multiplate,within_pool_contrasts,
retrieval_pool_vs_global,teacher_depth_interaction,phase_a_snu423}.tsv and per-condition tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import metrics as M
from .baselines import KAPPAS, MeanBaseline
from .scoring import DEF_TOTAL, DEF_WITHIN, Scorer

PS = (1.0, 0.5, 0.25)
SNU423 = "CVCL_0366"


def kname(k: float) -> str:
    return "linemean" if np.isinf(k) else ("poolmean" if k == 0 else f"shrunk_k{int(k)}")


def baseline_preds(d, fit, Lq, Dq) -> dict:
    Lf, Df = d.conditions(fit)
    mb = MeanBaseline(d.Y, Lf, Df, C.pool_of(d, Lf, Df), d.Y.shape[0], len(d.pools))
    pq = C.pool_of(d, Lq, Dq)
    out = {kname(k): mb.predict(Lq, Dq, pq, k)[0] for k in KAPPAS}
    out["globalmean"] = mb.global_predict(len(Lq))
    out["nochange"] = np.zeros((len(Lq), d.Y.shape[2]), np.float32)
    _, level, npl = mb.predict(Lq, Dq, pq, 0.0)
    return out, level, npl


def masks_for(d, L, Dg, multi) -> dict:
    m = multi[L, Dg]
    snu = np.flatnonzero(d.lines.cell_line_id.to_numpy() == SNU423)
    return dict(single_plate=~m, multi_plate=m, snu423=(L == snu[0]) if len(snu) else np.zeros(len(L), bool))


def arm_rows(sc: Scorer, arms: dict, role: str, masks=("all",), **extra) -> list[dict]:
    rows = []
    for (arm, p), key in arms.items():
        for mk in masks:
            rows.append(dict(C.std_fields(split_role=role), row_type="arm", arm=arm, p=p, **sc.summary(key, mk), **extra))
    return rows


def pair_rows(sc: Scorer, arms: dict, pairs, role: str, masks=("all",), **extra) -> list[dict]:
    rows = []
    for a, b in pairs:
        for p in PS:
            if (a, p) in arms and (b, p) in arms:
                for mk in masks:
                    rows.append(dict(C.std_fields(split_role=role), row_type="pair", p=p, **sc.pair(arms[(a, p)], arms[(b, p)], mk), **extra))
    return rows


def depth_interaction(sc: Scorer, arms: dict, base: str, cand: str, role: str) -> list[dict]:
    """I_depth = {L_base(0.25) - L_cand(0.25)} - {L_base(1) - L_cand(1)}; > 0: cand helps more at low depth."""
    rows = []
    for key, col in (("total", "v"), ("within", "lw")):
        A = sc.A["all"]
        pt = lambda arm, p: M.macro_point(sc.arms[arms[(arm, p)]][col], A)[0]
        bs = lambda arm, p: sc.arms[arms[(arm, p)]]["bs"]["all"][key]
        g25, g1 = pt(base, 0.25) - pt(cand, 0.25), pt(base, 1.0) - pt(cand, 1.0)
        b = (bs(base, 0.25) - bs(cand, 0.25)) - (bs(base, 1.0) - bs(cand, 1.0))
        lo, hi = M.ci(b)
        l25, h25 = M.ci(bs(base, 0.25) - bs(cand, 0.25)); l1, h1 = M.ci(bs(base, 1.0) - bs(cand, 1.0))
        rows.append(dict(C.std_fields(split_role=role), metric=key, base=base, candidate=cand,
                         gain_p025=g25, gain_p025_lo=l25, gain_p025_hi=h25, gain_p1=g1, gain_p1_lo=l1, gain_p1_hi=h1,
                         I_depth=g25 - g1, I_lo=lo, I_hi=hi, base_p025=pt(base, 0.25), I_rel_to_base=(g25 - g1) / pt(base, 0.25),
                         n_drugs=sc.nd, definition="I_depth = {L_base(p=0.25) - L_cand(p=0.25)} - {L_base(p=1) - L_cand(p=1)}; gain > 0 means the candidate is better"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/r6_followup_v1/phase_a.json")
    ap.add_argument("--out", default="reports/r6_followup_v1")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pdir = Path("predictions/r6_followup_v1"); pdir.mkdir(parents=True, exist_ok=True)
    mdir = Path("manifests/r6_followup_v1"); mdir.mkdir(parents=True, exist_ok=True)
    d = C.load(eval_views=False)
    pw = C.plate_weights()
    multi = (pw > 0).sum(-1) > 1
    tr, te = C.train_drugs(d), C.test_drugs(d)
    plates = pd.read_csv("manifests/chemcpa_moe_v2/plates.tsv", sep="\t").sort_values("plate_index").plate.to_numpy()
    cid = d.lines.cell_line_id.to_numpy()

    # ---- pool contract ---------------------------------------------------------------------------
    Lv, Dv = np.nonzero(d.valid)
    pv = C.pool_of(d, Lv, Dv)
    role = d.splits.split.to_numpy()[Dv]
    rows = []
    for r in d.pools.itertuples():
        m = pv == r.pool_id
        pl = sorted({plates[j] for l, g in zip(Lv[m], Dv[m]) for j in np.flatnonzero(pw[l, g] > 0)})
        row = dict(C.std_fields(), pool_id=int(r.pool_id), control_reference_id=f"{r.plate}|{r.cell_line_id}|reference",
                   line_id=r.cell_line_id, primary_plate=r.plate, all_treated_plates=",".join(pl),
                   n_control_cells=int(r.n_total), n_eval_input=int(r.n_eval_input), n_fit_input=int(r.n_fit_input), n_reference=int(r.n_reference),
                   train_drug_count=int((m & (role == "train")).sum()), val_drug_count_unused=int((m & (role == "val")).sum()),
                   test_drug_count=int((m & (role == "test")).sum()))
        for k in C.FOLDS:
            row[f"train_drug_count_fold{k}"] = int((m & (role == "train") & (d.fold[Dv] == k)).sum())
        row["train_multi_plate_conditions"] = int((m & (role == "train") & multi[Lv, Dv]).sum())
        row["test_multi_plate_conditions"] = int((m & (role == "test") & multi[Lv, Dv]).sum())
        row["sha256_reference_barcodes"] = r.sha256_reference_barcodes
        rows.append(row)
    pc = pd.DataFrame(rows)
    pc.to_csv(out / "pool_contract.tsv", sep="\t", index=False)
    print("pools", len(pc), "train drugs/pool", pc.train_drug_count.describe().to_dict(), "test drugs/pool", pc.test_drug_count.describe().to_dict(), flush=True)

    base_rows, svm_rows, win_rows, ret_rows, depth_rows, snu_rows = [], [], [], [], [], []
    base_arms = [kname(k) for k in KAPPAS] + ["globalmean", "nochange"]

    # ---- source OOF: response-mean baselines, kappa choice -----------------------------------------
    Ls, Ds, Fs, Ps_, lev, npl = [], [], [], {k: [] for k in base_arms}, [], []
    for k in C.FOLDS:
        fit, held = C.source_split(d, k)
        L, Dg = d.conditions(held)
        pr, level, n_ = baseline_preds(d, fit, L, Dg)
        Ls.append(L); Ds.append(Dg); Fs.append(np.full(len(L), k)); lev.append(level); npl.append(n_)
        for q in base_arms:
            Ps_[q].append(pr[q])
    L, Dg, F = np.concatenate(Ls), np.concatenate(Ds), np.concatenate(Fs)
    T = d.Y[L, Dg]
    sc = Scorer(T, L, Dg, C.pool_of(d, L, Dg), F, masks_for(d, L, Dg, multi), a.n_boot)
    arms = {}
    for q in base_arms:
        sc.add(q, [np.concatenate(Ps_[q])])
        for p in PS:
            arms[(q, p)] = q
    role_s = "source_oof_baselines"
    one_p = {k: v for k, v in arms.items() if k[1] == 1.0}
    kap = {kname(k): sc.summary(kname(k))["total_rmse"] for k in KAPPAS}
    best = min(KAPPAS, key=lambda k: kap[kname(k)])
    sel = dict(rule="smallest source out-of-fold total RMSE (drug-macro) over the three source folds; no TEST value used",
               grid=[("inf" if np.isinf(k) else k) for k in KAPPAS], oof_total_rmse=kap, chosen_kappa=("inf" if np.isinf(best) else best),
               chosen_arm=kname(best), at_grid_edge=bool(best in (KAPPAS[0], KAPPAS[-1])),
               oof_fallback=dict(pd.Series(np.concatenate(lev)).value_counts()), n_oof_conditions=int(len(L)), n_oof_drugs=int(sc.nd),
               fit_drugs_per_pool=pd.Series(np.concatenate(npl)).describe().to_dict())
    C.dump_json(sel, mdir / "kappa_selection.json")
    print(json.dumps(sel, indent=1, default=float), flush=True)
    shr = kname(best)
    note = "p-independent (no control-cell input); kappa selected on these rows"
    base_rows += arm_rows(sc, one_p, role_s, note=note)
    base_pairs = [(kname(k), "linemean") for k in KAPPAS[:-1]] + [("linemean", "globalmean"), ("poolmean", shr)]
    base_rows += [r for r in pair_rows(sc, arms, base_pairs, role_s) if r["p"] == 1.0]
    svm_rows += arm_rows(sc, one_p, role_s, masks=("all", "single_plate", "multi_plate"))
    svm_rows += [r for r in pair_rows(sc, arms, base_pairs, role_s, masks=("all", "single_plate", "multi_plate")) if r["p"] == 1.0]
    win_rows += arm_rows(sc, one_p, role_s)
    for q in ("poolmean", shr, "linemean"):
        for r in sc.retrieval_rows(q):
            ret_rows.append(dict(C.std_fields(split_role=role_s), row_type="arm", arm=q, p=1.0, **r))
    pd.DataFrame(dict(split_role=role_s, fold=F, cell_line_id=cid[L], drug_group=Dg, pool_id=sc.pool, multi_plate=multi[L, Dg],
                      truth_rms=sc.truth_rms, truth_centred_rms=sc.truth_c_rms,
                      **{f"{q}_total": sc.arms[q]["v"] for q in ("linemean", "poolmean", shr)},
                      **{f"{q}_within": sc.arms[q]["lw"] for q in ("linemean", "poolmean", shr)})).to_csv(
        pdir / "phase_a_source_oof_baselines_per_condition.tsv.gz", sep="\t", index=False, float_format="%.5f", compression="gzip")
    del sc

    # ---- existing predictions: dev fold 2 and the already-seen test set ------------------------------
    for tag, role_e, fit, held, fold in (("dev", "source_dev_fold2_existing", *C.source_split(d, 2), 2),
                                         ("final", "existing_test_diagnostic", tr, te, -1)):
        L, Dg = d.conditions(held)
        T = d.Y[L, Dg]
        sc = Scorer(T, L, Dg, C.pool_of(d, L, Dg), None, masks_for(d, L, Dg, multi), a.n_boot)
        bp, level, n_ = baseline_preds(d, fit, L, Dg)
        arms = {}
        for q in ("nochange", "globalmean", "linemean", "poolmean", shr):
            sc.add(q, [bp[q]])
            for p in PS:
                arms[(q, p)] = q
        root = C.orig_assets() / "runs" / tag
        files = {}
        for arm, paths in spec[tag].items():
            zs = [C.load_pred(root / rel, L, Dg) for rel in paths]
            files[arm] = [dict(path=rel, sha256=C.D.sha256_file(root / rel)) for rel in paths]
            if arm in ("nochange", "linemean"):                       # cross-check of the recomputed references
                assert np.allclose(zs[0]["p1.0_s0"], bp[arm], atol=2e-3), arm
                continue
            for p in PS:
                sc.add(f"{arm}@{p}", [z[k] for z in zs for k in C.PKEYS[p]])
                arms[(arm, p)] = f"{arm}@{p}"
            print(tag, arm, {p: round(sc.summary(arms[(arm, p)])["total_rmse"], 6) for p in PS}, flush=True)
        pairs = [tuple(x) for x in spec["pairs"]]
        pairs = [(shr if x == "SHRUNK" else x, shr if y == "SHRUNK" else y) for x, y in pairs]
        ms = ("all", "single_plate", "multi_plate")
        keep = lambda r: r["arm"] in ("nochange", "globalmean", "linemean", "poolmean", shr, "Z0", "B00", "B01", "NOSTATE", "TEACH0")
        base_rows += [r for r in arm_rows(sc, arms, role_e) if keep(r)]
        base_rows += [r for r in pair_rows(sc, arms, pairs, role_e) if {r["candidate"].split("@")[0], r["comparator"].split("@")[0]} & {"poolmean", shr, "linemean", "Z0"}]
        svm_rows += [r for r in arm_rows(sc, arms, role_e, masks=ms) if keep(r)]
        svm_rows += pair_rows(sc, arms, pairs, role_e, masks=ms)
        win_rows += arm_rows(sc, arms, role_e)
        win_rows += pair_rows(sc, arms, pairs, role_e)
        snu_rows += arm_rows(sc, arms, role_e, masks=("snu423",)) + pair_rows(sc, arms, pairs, role_e, masks=("snu423",))
        for (arm, p), key in arms.items():
            if arm in ("nochange", "globalmean") or (key == arm and p != 1.0):
                continue
            for r in sc.retrieval_rows(key):
                ret_rows.append(dict(C.std_fields(split_role=role_e), row_type="arm", arm=arm, p=p, **r))
        for x, y in pairs:
            for p in PS:
                if (x, p) in arms and (y, p) in arms and arms[(x, p)] != x:
                    for r in sc.retrieval_pair(arms[(x, p)], arms[(y, p)]):
                        ret_rows.append(dict(C.std_fields(split_role=role_e), row_type="pair", p=p, **r))
        for base, cand in (("B00", "B01"), ("B00", "B11"), ("B10", "B11")):
            if (base, 1.0) in arms and (cand, 1.0) in arms:
                depth_rows += depth_interaction(sc, arms, base, cand, role_e)
        cols = dict(split_role=role_e, fold=fold, cell_line_id=cid[L], drug_group=Dg, pool_id=sc.pool, multi_plate=multi[L, Dg],
                    fallback_level=level, n_fit_drugs_in_pool=n_, truth_rms=sc.truth_rms, truth_centred_rms=sc.truth_c_rms)
        for (arm, p), key in arms.items():
            if key == arm and p != 1.0:
                continue
            sfx = arm if key == arm else f"{arm}_p{p}"
            cols[f"{sfx}_total"] = sc.arms[key]["v"]; cols[f"{sfx}_within"] = sc.arms[key]["lw"]
        pd.DataFrame(cols).to_csv(pdir / f"phase_a_{role_e}_per_condition.tsv.gz", sep="\t", index=False, float_format="%.5f", compression="gzip")
        C.dump_json(dict(split_role=role_e, n_conditions=int(len(L)), n_drugs=int(sc.nd), files=files, commit=C.git_commit(),
                         fallback=dict(pd.Series(level).value_counts()), fit_drugs_per_pool=pd.Series(n_).describe().to_dict()),
                    mdir / f"phase_a_inputs_{tag}.json")
        del sc

    def clean(rows):
        df = pd.DataFrame(rows)
        for c in ("arm", "candidate", "comparator"):
            if c in df:
                df[c] = df[c].astype(object).where(df[c].isna(), df[c].astype(str).str.replace(r"@.*$", "", regex=True))
        df["metric_definition_total"] = DEF_TOTAL; df["metric_definition_within"] = DEF_WITHIN
        df["bootstrap"] = f"paired, drug structure groups (all cell lines together), {a.n_boot} replicates, pools re-centred per replicate"
        return df

    clean(base_rows).to_csv(out / "pool_baselines.tsv", sep="\t", index=False, float_format="%.6f")
    clean(svm_rows).to_csv(out / "single_vs_multiplate.tsv", sep="\t", index=False, float_format="%.6f")
    clean(win_rows).to_csv(out / "within_pool_contrasts.tsv", sep="\t", index=False, float_format="%.6f")
    clean(snu_rows).to_csv(out / "phase_a_snu423.tsv", sep="\t", index=False, float_format="%.6f")
    rr = pd.DataFrame(ret_rows)
    for c in ("arm", "candidate", "comparator"):
        rr[c] = rr[c].astype(object).where(rr[c].isna(), rr[c].astype(str).str.replace(r"@.*$", "", regex=True))
    rr["metric_definition"] = "tie-corrected top1/top5/MRR (appendix B) by Pearson correlation of centred prediction with centred measured candidates; drug-macro; legacy_* = original R6 definitions"
    rr.to_csv(out / "retrieval_pool_vs_global.tsv", sep="\t", index=False, float_format="%.6f")
    pd.DataFrame(depth_rows).to_csv(out / "teacher_depth_interaction.tsv", sep="\t", index=False, float_format="%.7f")
    w = clean(win_rows)
    print(w[(w.row_type == "arm") & (w.subset == "all")][["split_role", "arm", "p", "total_rmse", "within_rmse", "within_lo", "within_hi", "centred_magnitude_ratio"]].to_string(index=False))
    print(w[(w.row_type == "pair") & (w.subset == "all")][["split_role", "candidate", "comparator", "p", "total_diff", "total_rel", "total_category", "within_diff", "within_lo", "within_hi", "within_rel", "within_category"]].to_string(index=False))
    print(rr[rr.row_type == "arm"][["split_role", "arm", "p", "candidate_pool", "top1", "chance_top1", "mrr", "chance_mrr", "legacy_rank_score", "legacy_unique_top1", "n_candidates"]].to_string(index=False))
    print(pd.DataFrame(depth_rows).to_string(index=False))


if __name__ == "__main__":
    main()
