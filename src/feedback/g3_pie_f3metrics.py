#!/usr/bin/env python
"""G3 / G1-b part 2: score PIE's held-out-line predictions (and the PIE-split baselines) with F3's
metrics against F3's independently processed truth (raw GEO/figshare h5ad -> CP10k log1p,
gem_group-matched control, Welch DEG), on overlapping targets and genes.

Preprocessing differences (stated in the report): PIE truth = PIE `delta_p` (pseudobulk log1p of
knockdown-filtered perturbed cells minus the line's pooled control mean; DE = Mann-Whitney/BH on
CP10k with >=5 CPM filter); F3 truth = mean log1p(CP10k by UMI_count) of all cells of the target
minus the gem_group-composition-matched control mean; DEG = Welch t + BH over 6,642 genes.
Cell counts differ (HepG2: PIE 90,826 cells vs F3 145,473 cells in the raw file).

Inputs: assets/g3_pie/baselines/pred_<fold>_<setting>.npz (from g3_pie_baselines.py, incl.
PIE_repro if the eval ran), assets/f3/matrices/<L>_delta.npz, assets/f3/loco/panel_<L>.npy,
assets/f3/pseudobulk/<L>.npz (var_ids/var_names), assets/f3/loco/pred_<L>_A_*.npz (F3 baselines,
condition A, only for targets seen in F3's source lines).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_loco import per_target_metrics, retrieval_rank  # noqa: E402

LINES = {"hepg2": "HepG2", "k562": "K562", "rpe1": "RPE1", "jurkat": "Jurkat"}
KEYS = ["rmse", "pearson", "pearson_pertcentered", "dir_top20", "deg_auprc", "retrieval_rank"]
LFC_HEADS = {"PIE_repro", "same_target_raw", "same_target_cal", "nearest_line_same_target", "global_mean_kd",
             "ridge_stacked_kd"}


def boot_diff(a, b, nboot, rng):
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = a[ok] - b[ok]
    if len(d) < 3:
        return np.nan, np.nan, np.nan, int(len(d))
    bs = rng.choice(d, (nboot, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="hepg2")
    ap.add_argument("--g3", default="assets/g3_pie/baselines")
    ap.add_argument("--f3", default="assets/f3")
    ap.add_argument("--out", default="reports/feedback/G3")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    L = LINES[args.fold]
    f3 = Path(args.f3)
    M = np.load(f3 / "matrices" / f"{L}_delta.npz", allow_pickle=True)
    pb = np.load(f3 / "pseudobulk" / f"{L}.npz", allow_pickle=True)
    sym = dict(zip(pb["var_ids"].tolist(), pb["var_names"].tolist()))
    f3_genes = M["genes"]  # Ensembl, 6,642 common genes
    f3_sym = np.array([sym.get(g, g) for g in f3_genes])
    panel_ids = np.load(f3 / "loco" / f"panel_{L}.npy", allow_pickle=True)  # Ensembl ids (source-defined)
    gpos_f3 = {g: i for i, g in enumerate(f3_genes.tolist())}
    panel = np.array([gpos_f3[g] for g in panel_ids.tolist()])  # indices into f3_genes, panel order
    # source mean delta on the full common axis (Systema-style centring in per_target_metrics)
    src = [l for l in LINES.values() if l != L]
    m_src_full = np.mean([np.load(f3 / "matrices" / f"{s}_delta.npz")["delta"].mean(0) for s in src], 0)
    tidx = {t: i for i, t in enumerate(M["targets"].tolist())}
    rows_out, boots, truth_agree, meta = [], [], [], {}
    for setting in ("unseen_ctx", "unseen_ctx_pert"):
        f = Path(args.g3) / f"pred_{args.fold}_{setting}.npz"
        if not f.is_file():
            print(f"[g3f3] missing {f}")
            continue
        P = np.load(f, allow_pickle=True)
        pie_genes = P["genes"].tolist()
        pg = {g: i for i, g in enumerate(pie_genes)}
        # genes: F3 panel ∩ PIE axis (by symbol)
        gsel = [(j, pg[f3_sym[j]]) for j in panel if f3_sym[j] in pg]
        fj = np.array([a for a, _ in gsel]); pj = np.array([b for _, b in gsel])
        perts = P["perts"].tolist()
        keep = [(i, tidx[p]) for i, p in enumerate(perts) if p in tidx]
        pi = np.array([a for a, _ in keep]); ti = np.array([b for _, b in keep])
        Y = M["delta"][ti][:, fj].astype(np.float64)
        deg = M["deg"][ti][:, fj]
        tst = M["tstat"][ti][:, fj]
        pos = {g: k for k, g in enumerate(fj.tolist())}
        rows = np.array([pos.get(int(M["tgene"][t]), -1) if M["tgene"][t] >= 0 else -1 for t in ti])
        m_src = m_src_full[fj]
        gene_mask = np.ones(len(fj), bool)
        tg = rows[rows >= 0]
        gene_mask[tg] = False
        meta[setting] = dict(n_test_rows=len(perts), n_overlap_targets=int(len(ti)),
                             n_genes=int(len(fj)), panel_size=int(len(panel)))
        # truth agreement PIE delta_p vs F3 delta
        T = P["truth_delta"][pi][:, pj].astype(np.float64)
        for k in range(len(ti)):
            msk = np.ones(len(fj), bool)
            if rows[k] >= 0:
                msk[rows[k]] = False
            a, b = T[k, msk], Y[k, msk]
            truth_agree.append(dict(setting=setting, perturbation=perts[pi[k]],
                                    pearson=float(np.corrcoef(a, b)[0, 1]),
                                    rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                                    norm_ratio=float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-12))))
        preds = {n[: -len("__delta")]: P[n][pi][:, pj].astype(np.float64)
                 for n in P.files if n.endswith("__delta")}
        preds["PIE_truth_as_prediction"] = T  # preprocessing ceiling: PIE's own label vs F3 truth
        # LFC heads (log2 FC scale): only scale-free F3 metrics are meaningful (RMSE is not reported)
        for n in P.files:
            if n.endswith("__lfc") and n[: -len("__lfc")] in LFC_HEADS:
                preds[n[: -len("__lfc")] + "[lfc]"] = P[n][pi][:, pj].astype(np.float64)
        if setting == "unseen_ctx":
            fa = list((f3 / "loco").glob(f"pred_{L}_A_*.npz"))
            if fa:
                FA = np.load(fa[0], allow_pickle=True)
                ftidx = {t: i for i, t in enumerate(FA["targets"].tolist())}
                ppos = {g: k for k, g in enumerate(panel.tolist())}
                cols = np.array([ppos[j] for j in fj])
                have = np.array([perts[i] in ftidx for i in pi])
                meta[setting]["f3_A_targets_present"] = int(have.sum())
                for n in FA.files:
                    if n == "targets":
                        continue
                    arr = np.full((len(pi), len(fj)), np.nan)
                    arr[have] = FA[n][[ftidx[perts[i]] for i in pi[have]]][:, cols]
                    preds[f"F3_{n}"] = arr
        per = {}
        for name, X in preds.items():
            ok = ~np.isnan(X).any(1)
            m = per_target_metrics(np.nan_to_num(X), Y, deg, tst, rows, m_src)
            df = pd.DataFrame(m)
            rr = np.full(len(ti), np.nan)
            if ok.sum() > 2:
                rr[ok] = retrieval_rank(X[ok], Y[ok], m_src, gene_mask)
            df["retrieval_rank"] = rr
            df.loc[~ok, :] = np.nan
            if name.endswith("[lfc]"):
                df["rmse"] = np.nan
            df["perturbation"] = [perts[i] for i in pi]
            per[name] = df
            s = df[KEYS].mean().to_dict()
            rows_out.append(dict(setting=setting, predictor=name, n_targets=int(ok.sum()), **s))
        for ref in ("PIE_repro", "PIE_repro[lfc]"):
            if ref not in per:
                continue
            for name, df in per.items():
                if name == ref or name.startswith("PIE_repro"):
                    continue
                for k in KEYS:
                    d, lo, hi, n = boot_diff(per[ref][k].to_numpy(float), df[k].to_numpy(float), args.nboot, rng)
                    boots.append(dict(setting=setting, a=ref, b=name, metric=k, diff=d, lo=lo, hi=hi, n=n))
        pd.concat([d.assign(predictor=n) for n, d in per.items()]).to_csv(
            Path(args.g3) / f"f3metrics_per_target_{args.fold}_{setting}.tsv.gz", sep="\t", index=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out / f"f3metrics_summary_{args.fold}.tsv", sep="\t", index=False, float_format="%.5f")
    pd.DataFrame(boots).to_csv(out / f"f3metrics_paired_bootstrap_{args.fold}.tsv", sep="\t", index=False, float_format="%.5f")
    ta = pd.DataFrame(truth_agree)
    ta.to_csv(Path(args.g3) / f"truth_agreement_{args.fold}.tsv.gz", sep="\t", index=False)
    meta["truth_agreement_median"] = ta.groupby("setting")[["pearson", "rmse", "norm_ratio"]].median().to_dict() if len(ta) else {}
    (out / f"f3metrics_meta_{args.fold}.json").write_text(json.dumps(meta, indent=1))
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame(rows_out).to_string())
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
