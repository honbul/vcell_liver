#!/usr/bin/env python
"""G4: F3 metrics (raw-data truth, F3 code) for State ST-HVG zeroshot, PIE (official + seed retrains)
and the F3 / PIE-split baselines, on shared targets x genes. Read-only use of the F3 env.

Part A (--state, all 4 folds; condition "unseen context, seen target"):
  targets  T3 = State-predicted ∩ F3 condition-A targets of L ∩ PIE unseen_ctx perturbations
           T2 = State-predicted ∩ F3 condition-A targets (State vs F3 baselines only)
  genes    G  = F3 fold panel (source-defined) ∩ State 2,000 HVGs ∩ PIE gene axis (by symbol)
  predictors: State_final (primary; no selection), State_best (val_loss on L's test cells -> reference),
           State_final_pool43 (basal-pool resample), State_authors_best (authors' own eval predictions,
           their filtered cells), PIE_official (G3 GPU repro, delta head), PIE-split same_target_cal,
           F3 calibrated_same_target / same_target_mean / no_change / global_mean / ridge_pca.
  metric variants: "excl" = F3 definition (target gene excluded from every metric incl. retrieval);
           "incl" = target gene kept (only matters when it is in G).
  paired bootstrap (targets, 2,000): State_final vs F3_calibrated_same_target, vs PIE_official,
           vs State_best; PIE_official vs F3_calibrated_same_target.
Part B (--pie-seeds, HepG2): unseen_ctx and unseen_ctx_pert, genes = F3 panel ∩ PIE axis, targets in
  F3 (as G3 §4): PIE_seed<S>, PIE_official, PIE-split baselines, F3 calibrated (ctx only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_loco import per_target_metrics, retrieval_rank  # noqa: E402
from g3_pie_f3metrics import boot_diff  # noqa: E402

LINES = {"hepg2": "HepG2", "k562": "K562", "rpe1": "RPE1", "jurkat": "Jurkat"}
KEYS = ["rmse", "pearson", "pearson_pertcentered", "dir_top20", "deg_auprc", "retrieval_rank"]


class Truth:
    def __init__(self, f3, L):
        f3 = Path(f3)
        self.M = np.load(f3 / "matrices" / f"{L}_delta.npz", allow_pickle=True)
        pb = np.load(f3 / "pseudobulk" / f"{L}.npz", allow_pickle=True)
        sym = dict(zip(pb["var_ids"].tolist(), pb["var_names"].tolist()))
        self.genes = self.M["genes"]
        self.sym = np.array([sym.get(g, g) for g in self.genes])
        pid = np.load(f3 / "loco" / f"panel_{L}.npy", allow_pickle=True)
        gpos = {g: i for i, g in enumerate(self.genes.tolist())}
        self.panel = np.array([gpos[g] for g in pid.tolist()])
        src = [l for l in LINES.values() if l != L]
        self.m_src_full = np.mean([np.load(f3 / "matrices" / f"{s}_delta.npz")["delta"].mean(0) for s in src], 0)
        self.tidx = {t: i for i, t in enumerate(self.M["targets"].tolist())}
        fa = sorted((f3 / "loco").glob(f"pred_{L}_A_*.npz"))[0]
        self.FA = np.load(fa, allow_pickle=True)
        self.fa_tidx = {t: i for i, t in enumerate(self.FA["targets"].tolist())}
        self.panel_pos = {j: k for k, j in enumerate(self.panel.tolist())}

    def select(self, targets, fj):
        ti = np.array([self.tidx[t] for t in targets])
        Y = self.M["delta"][ti][:, fj].astype(np.float64)
        deg, tst = self.M["deg"][ti][:, fj], self.M["tstat"][ti][:, fj]
        pos = {g: k for k, g in enumerate(fj.tolist())}
        rows = np.array([pos.get(int(self.M["tgene"][t]), -1) if self.M["tgene"][t] >= 0 else -1 for t in ti])
        return Y, deg, tst, rows

    def f3_pred(self, name, targets, fj):
        cols = np.array([self.panel_pos[j] for j in fj])
        return self.FA[name][[self.fa_tidx[t] for t in targets]][:, cols].astype(np.float64)


def score_set(preds, Y, deg, tst, rows, m_src, variant):
    r = rows if variant == "excl" else np.full_like(rows, -1)
    gmask = np.ones(Y.shape[1], bool)
    if variant == "excl":
        gmask[rows[rows >= 0]] = False
    per = {}
    for name, X in preds.items():
        ok = ~np.isnan(X).any(1)  # a predictor may miss targets (authors' anchor): score its own subset
        df = pd.DataFrame(np.nan, index=range(len(Y)), columns=KEYS + ["n_deg"])
        if ok.sum() > 2:
            m = pd.DataFrame(per_target_metrics(X[ok], Y[ok], deg[ok], tst[ok], r[ok], m_src))
            m["retrieval_rank"] = retrieval_rank(X[ok], Y[ok], m_src, gmask)
            df.loc[np.where(ok)[0], m.columns] = m.to_numpy()
        per[name] = df
    return per


def summarize(per, tag, out_rows):
    for name, df in per.items():
        out_rows.append(dict(**tag, predictor=name, n_targets=int(df["rmse"].notna().sum()), **df[KEYS].mean().to_dict()))


def boots_for(per, pairs, tag, nboot, rng, out):
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        for k in KEYS:
            d, lo, hi, n = boot_diff(per[a][k].to_numpy(float), per[b][k].to_numpy(float), nboot, rng)
            out.append(dict(**tag, a=a, b=b, metric=k, diff=d, lo=lo, hi=hi, n=n))


def part_state(a, rng):
    summ, boots, meta, perrows = [], [], {}, []
    for fold, L in LINES.items():
        sf = Path(a.state_pred) / f"{L}.npz"
        if not sf.is_file():
            print(f"[g4f] no State predictions for {L}")
            continue
        T = Truth(a.f3, L)
        S = np.load(sf, allow_pickle=True)
        s_t = S["targets"].astype(str).tolist()
        s_g = {g: i for i, g in enumerate(S["genes"].astype(str).tolist())}
        P = np.load(Path(a.g3) / f"pred_{fold}_unseen_ctx.npz", allow_pickle=True)
        p_g = {g: i for i, g in enumerate(P["genes"].tolist())}
        p_t = {t: i for i, t in enumerate(P["perts"].tolist())}
        fj = np.array([j for j in T.panel if T.sym[j] in s_g and T.sym[j] in p_g])
        sj = np.array([s_g[T.sym[j]] for j in fj]); pj = np.array([p_g[T.sym[j]] for j in fj])
        st_idx = {t: i for i, t in enumerate(s_t)}
        aut = {t: i for i, t in enumerate(S["authors_targets"].astype(str).tolist())} if "authors_targets" in S.files else {}
        T2 = [t for t in s_t if t in T.fa_tidx and t in T.tidx]
        T3 = [t for t in T2 if t in p_t]
        meta[fold] = dict(n_genes=int(len(fj)), panel=int(len(T.panel)), n_state_targets=len(s_t), T2=len(T2), T3=len(T3))
        for tset_name, tset in (("T3_state_pie_f3", T3), ("T2_state_f3", T2)):
            Y, deg, tst, rows = T.select(tset, fj)
            m_src = T.m_src_full[fj]
            si = [st_idx[t] for t in tset]
            preds = {"State_final": S["delta_final_p42"][si][:, sj].astype(np.float64),
                     "State_best": S["delta_best_p42"][si][:, sj].astype(np.float64)}
            if "delta_final_p43" in S.files:
                preds["State_final_pool43"] = S["delta_final_p43"][si][:, sj].astype(np.float64)
            for nm in ("calibrated_same_target", "same_target_mean", "no_change", "global_mean", "ridge_pca"):
                preds[f"F3_{nm}"] = T.f3_pred(nm, tset, fj)
            if tset_name.startswith("T3"):
                pi = [p_t[t] for t in tset]
                preds["PIE_official"] = P["PIE_repro__delta"][pi][:, pj].astype(np.float64)
                preds["PIEsplit_same_target_cal"] = P["same_target_cal__delta"][pi][:, pj].astype(np.float64)
            # authors' own eval predictions (their cells) where available, as an anchor
            if aut:
                for nm, key in (("State_authors_best", "authors_delta_pred"), ("State_authors_truth_as_pred", "authors_delta_real")):
                    X = np.full((len(tset), len(sj)), np.nan)
                    have = np.array([t in aut for t in tset])
                    X[have] = S[key][[aut[t] for t in tset if t in aut]][:, sj]
                    preds[nm] = X
                meta[fold][f"{tset_name}_authors_n"] = int(have.sum())
            meta[fold][f"{tset_name}_frac_target_gene_in_G"] = float((rows >= 0).mean())
            for variant in ("excl", "incl"):
                per = score_set(preds, Y, deg, tst, rows, m_src, variant)
                tag = dict(fold=fold, target_set=tset_name, variant=variant)
                summarize(per, tag, summ)
                pairs = [("State_final", "F3_calibrated_same_target"), ("State_final", "PIE_official"),
                         ("State_final", "State_best"), ("PIE_official", "F3_calibrated_same_target"),
                         ("State_best", "F3_calibrated_same_target"), ("State_final", "F3_same_target_mean"),
                         ("State_final", "F3_no_change")]
                boots_for(per, pairs, tag, a.nboot, rng, boots)
                if variant == "excl" and tset_name.startswith("T3"):
                    for nm, df in per.items():
                        perrows.append(df.assign(fold=fold, predictor=nm, target=tset))
        print(f"[g4f] {fold}: {meta[fold]}", flush=True)
    out = Path(a.out)
    pd.DataFrame(summ).to_csv(out / "state_f3metrics_summary.tsv", sep="\t", index=False, float_format="%.5f")
    pd.DataFrame(boots).to_csv(out / "state_f3metrics_paired_bootstrap.tsv", sep="\t", index=False, float_format="%.5f")
    (out / "state_f3metrics_meta.json").write_text(json.dumps(meta, indent=1))
    if perrows:
        pd.concat(perrows).to_csv(Path(a.state_pred) / "state_f3metrics_per_target_T3.tsv.gz", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 500):
        d = pd.DataFrame(summ)
        print(d[d.variant == "excl"].to_string())


def part_pie_seeds(a, rng):
    fold, L = "hepg2", "HepG2"
    T = Truth(a.f3, L)
    summ, boots, meta = [], [], {}
    for setting in ("unseen_ctx", "unseen_ctx_pert"):
        P = np.load(Path(a.g3) / f"pred_{fold}_{setting}.npz", allow_pickle=True)
        perts = P["perts"].tolist()
        p_g = {g: i for i, g in enumerate(P["genes"].tolist())}
        fj = np.array([j for j in T.panel if T.sym[j] in p_g]); pj = np.array([p_g[T.sym[j]] for j in fj])
        keep = [i for i, p in enumerate(perts) if p in T.tidx]
        tset = [perts[i] for i in keep]
        Y, deg, tst, rows = T.select(tset, fj)
        m_src = T.m_src_full[fj]
        preds = {"PIE_official": P["PIE_repro__delta"][keep][:, pj].astype(np.float64)}
        base = ["same_target_cal", "ridge_stacked_kd", "global_mean_kd"] if setting == "unseen_ctx" else ["ridge_stacked_kd", "global_mean_kd"]
        for b in base:
            preds[f"PIEsplit_{b}"] = P[f"{b}__delta"][keep][:, pj].astype(np.float64)
        if setting == "unseen_ctx":
            ok = [t in T.fa_tidx for t in tset]
            if all(ok):
                preds["F3_calibrated_same_target"] = T.f3_pred("calibrated_same_target", tset, fj)
        seeds = []
        for f in sorted(Path(a.pie_seeds).glob(f"pred_{fold}_{setting}_seed*.npz")):
            S = np.load(f, allow_pickle=True)
            assert S["perts"].tolist() == perts and S["genes"].tolist() == P["genes"].tolist()
            nm = "PIE_" + f.stem.split("_")[-1]
            preds[nm] = S["delta"][keep][:, pj].astype(np.float64)
            seeds.append(nm)
        meta[setting] = dict(n_targets=len(tset), n_genes=int(len(fj)), seeds=seeds)
        for variant in ("excl", "incl"):
            per = score_set(preds, Y, deg, tst, rows, m_src, variant)
            tag = dict(fold=fold, setting=setting, variant=variant)
            summarize(per, tag, summ)
            ref = "PIEsplit_same_target_cal" if setting == "unseen_ctx" else "PIEsplit_ridge_stacked_kd"
            pairs = [(s, ref) for s in seeds] + [(s, "PIE_official") for s in seeds] + [("PIE_official", ref)]
            if setting == "unseen_ctx":
                pairs += [(s, "F3_calibrated_same_target") for s in seeds]
            boots_for(per, pairs, tag, a.nboot, rng, boots)
    out = Path(a.out)
    pd.DataFrame(summ).to_csv(out / "pie_seed_f3metrics_summary_hepg2.tsv", sep="\t", index=False, float_format="%.5f")
    pd.DataFrame(boots).to_csv(out / "pie_seed_f3metrics_paired_bootstrap_hepg2.tsv", sep="\t", index=False, float_format="%.5f")
    (out / "pie_seed_f3metrics_meta_hepg2.json").write_text(json.dumps(meta, indent=1))
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        d = pd.DataFrame(summ)
        print(d[d.variant == "excl"].to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f3", default="assets/f3")
    ap.add_argument("--g3", default="assets/g3_pie/baselines")
    ap.add_argument("--state-pred", default="assets/g4/state/pred")
    ap.add_argument("--pie-seeds", default="assets/g4/pie_seeds")
    ap.add_argument("--out", default="reports/feedback/G4")
    ap.add_argument("--part", choices=["state", "pie_seeds", "both"], default="both")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261110)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    if a.part in ("state", "both"):
        part_state(a, rng)
    if a.part in ("pie_seeds", "both"):
        part_pie_seeds(a, rng)


if __name__ == "__main__":
    main()
