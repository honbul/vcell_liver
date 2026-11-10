"""F4: effect-strength / noise / replicate analysis from per-well pseudobulks (f4_tahoe_pseudobulk.py).

For each liver line (SNU-423, HepG2/C3A):
  * plate-matched DMSO reference = sum of the two DMSO wells of that plate
  * per treated well: log2(CPM+1) delta vs plate DMSO on genes expressed in DMSO (mean CPM >= 10)
    metrics: rms_lfc, n genes |lfc|>0.5, n genes |lfc|>1
  * nulls: (a) size-matched random DMSO pseudo-wells vs rest of plate DMSO,
           (b) DMSO well vs the other DMSO well of the same plate (includes well effects),
           (c) DMSO pool of plate i vs plate j (plate noise)
  * replicate concordance (same drug x dose on 2+ wells/plates), dose concordance, SNU-423 vs HepG2 same-well
  * Tahoe official pseudobulk DESeq2 summary (all lines) to place SNU-423 among 50 lines.
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

LINES = {"snu423": "CVCL_0366", "hepg2c3a": "CVCL_1098"}


def load(asset_dir: Path, tag: str):
    z = np.load(asset_dir / f"{tag}_pseudobulk.npz", allow_pickle=True)
    gt = pd.read_json(io.StringIO(str(z["groups"])), orient="records")
    return z["sums"].astype(np.float64), z["genes"].astype(str), gt


def lcpm(x: np.ndarray) -> np.ndarray:
    tot = x.sum(axis=-1, keepdims=True)
    return np.log2(x / np.maximum(tot, 1) * 1e6 + 1)


def metrics(lfc: np.ndarray) -> dict:
    return dict(rms_lfc=float(np.sqrt(np.mean(lfc ** 2))), n_lfc05=int((np.abs(lfc) > 0.5).sum()),
                n_lfc1=int((np.abs(lfc) > 1).sum()))


def analyse_line(asset_dir: Path, out_dir: Path, tag: str, summ: pd.DataFrame):
    S, genes, gt = load(asset_dir, tag)
    smp = gt[gt.kind == "sample"].copy()
    smp = smp.merge(summ[["sample", "drug", "conc_uM", "cell_ratio_vs_plate_dmso", "moa-fine"]].rename(
        columns={"sample": "group", "drug": "drug_meta"}), on="group", how="left")
    is_dmso = smp.drug.str.startswith("DMSO")
    plates = sorted(smp.plate.unique())
    pool = {p: S[smp[(smp.plate == p) & is_dmso].index].sum(axis=0) for p in plates}
    pool_l = {p: lcpm(v) for p, v in pool.items()}
    cpm_pool = np.vstack([v / v.sum() * 1e6 for v in pool.values()])
    expr = (cpm_pool.mean(axis=0) >= 10) & ~pd.Series(genes).str.startswith("unmapped").to_numpy()
    lp = {p: v[expr] for p, v in pool_l.items()}

    rows = []
    lfc_store = {}
    for i, r in smp.iterrows():
        if r.n_cells == 0:
            continue
        lf = lcpm(S[i])[expr] - lp[r.plate]
        m = metrics(lf)
        rows.append(dict(sample=r.group, plate=r.plate, drug=r.drug_meta, conc_uM=r.conc_uM, n_cells=int(r.n_cells),
                         total_umi=float(r.total_umi), cell_ratio_vs_plate_dmso=r.cell_ratio_vs_plate_dmso,
                         moa_fine=r["moa-fine"], is_dmso=bool(str(r.drug).startswith("DMSO")), **m))
        lfc_store[r.group] = lf
    eff = pd.DataFrame(rows)

    # Sampling-noise model: mean squared log2(CPM+1) delta between two pseudobulks with total UMI U1, U2
    # follows a power law in h = 1/U1 + 1/U2 (fitted on random DMSO pseudo-wells = pure sampling noise).
    # Effect size R = observed rms^2 / sampling-only expectation. Real DMSO well pairs (sampling + well
    # effects) give the empirical null of R; a well is "detectable" if R exceeds every DMSO well pair's R.
    pool_umi = {p: float(v.sum()) for p, v in pool.items()}
    nb = []
    for i, r in gt[gt.kind == "dmso_bucket"].iterrows():
        rest = pool[r.plate] - S[i]
        lf = lcpm(S[i])[expr] - lcpm(rest)[expr]
        nb.append(dict(B=r.B, plate=r.plate, n_cells=int(r.n_cells), h=1 / S[i].sum() + 1 / rest.sum(), **metrics(lf)))
    nb = pd.DataFrame(nb)
    beta, alpha = np.polyfit(np.log(nb.h), np.log(nb.rms_lfc ** 2), 1)
    pred = lambda h: np.exp(alpha + beta * np.log(h))
    nb["R"] = nb.rms_lfc ** 2 / pred(nb.h)
    nbB = nb.groupby("B").agg(mean_cells=("n_cells", "mean"), rms_median=("rms_lfc", "median"),
                              R_median=("R", "median"), R_max=("R", "max")).reset_index()
    wb = []
    for p in plates:
        w = smp[(smp.plate == p) & is_dmso]
        if len(w) == 2:
            a, b = w.index
            lf = lcpm(S[a])[expr] - lcpm(S[b])[expr]
            h = 1 / S[a].sum() + 1 / S[b].sum()
            wb.append(dict(plate=p, n_cells_a=int(w.n_cells.iloc[0]), n_cells_b=int(w.n_cells.iloc[1]), h=h, **metrics(lf)))
    wb = pd.DataFrame(wb)
    wb["R"] = wb.rms_lfc ** 2 / pred(wb.h)
    pc = []
    for p, q in itertools.combinations(plates, 2):
        pc.append(dict(p=p, q=q, **metrics(lp[p] - lp[q])))
    pc = pd.DataFrame(pc)

    eff["h"] = 1 / eff.total_umi + eff.plate.map(lambda p: 1 / pool_umi[p])
    eff["R_vs_sampling"] = eff.rms_lfc ** 2 / pred(eff.h)
    eff["R_vs_dmso_well_median"] = eff.R_vs_sampling / float(wb.R.median())
    eff["excess_var"] = eff.rms_lfc ** 2 - pred(eff.h) * float(wb.R.median())
    thr = float(wb.R.max())
    eff["resp_strict"] = eff.R_vs_sampling > thr
    eff["resp_strong"] = eff.R_vs_sampling > 2 * thr
    eff["resp_lenient"] = eff.resp_strict
    eff["resp_sizematched"] = eff.resp_strict
    noise_model = {"log(rms2) = alpha + beta*log(h), h=1/U1+1/U2": [float(alpha), float(beta)],
                   "bucket_R_by_B": nbB.round(4).to_dict(orient="records"),
                   "dmso_well_pair_R": wb[["plate", "n_cells_a", "n_cells_b", "rms_lfc", "R"]].round(4).to_dict(orient="records"),
                   "dmso_well_pair_R_median": float(wb.R.median()), "threshold_detectable_R": thr, "threshold_strong_R": 2 * thr}
    eff["low_cells_lt50"] = eff.n_cells < 50

    trt = eff[~eff.is_dmso]
    # replicate concordance
    rep = []
    for (d, c), g in trt.groupby(["drug", "conc_uM"]):
        if len(g) < 2:
            continue
        for a, b in itertools.combinations(g["sample"], 2):
            ra = trt.set_index("sample").loc[[a, b]]
            rep.append(dict(drug=d, conc_uM=c, a=a, b=b, plate_a=ra.plate.iloc[0], plate_b=ra.plate.iloc[1],
                            r=float(np.corrcoef(lfc_store[a], lfc_store[b])[0, 1]),
                            both_resp=bool(ra.resp_strict.all()), min_cells=int(ra.n_cells.min())))
    rep = pd.DataFrame(rep)
    rng = np.random.default_rng(0)
    smps = trt["sample"].to_numpy()
    tr_i = trt.set_index("sample")
    null_r = []
    while len(null_r) < 2000:
        a, b = rng.choice(smps, 2, replace=False)
        if tr_i.at[a, "drug"] == tr_i.at[b, "drug"] or tr_i.at[a, "plate"] == tr_i.at[b, "plate"]:
            continue
        null_r.append(dict(r=float(np.corrcoef(lfc_store[a], lfc_store[b])[0, 1]),
                           both_resp=bool(tr_i.at[a, "resp_strict"] and tr_i.at[b, "resp_strict"])))
    null_r = pd.DataFrame(null_r)
    # dose concordance 0.5 vs 5 uM (first well each)
    dose = []
    for d, g in trt.groupby("drug"):
        gg = g.sort_values("n_cells", ascending=False).drop_duplicates("conc_uM").set_index("conc_uM")
        if 0.5 in gg.index and 5.0 in gg.index:
            a, b = gg.at[0.5, "sample"], gg.at[5.0, "sample"]
            dose.append(dict(drug=d, r=float(np.corrcoef(lfc_store[a], lfc_store[b])[0, 1]),
                             same_plate=bool(gg.at[0.5, "plate"] == gg.at[5.0, "plate"]),
                             resp5=bool(gg.at[5.0, "resp_strict"]),
                             rms_increasing=bool(gg.sort_index().excess_var.is_monotonic_increasing) if len(gg) == 3 else None))
    dose = pd.DataFrame(dose)

    eff.to_csv(out_dir / f"{tag}_effect_strength.tsv", sep="\t", index=False, float_format="%.4g")
    q = lambda s: {k: float(v) for k, v in s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).items()}
    summary = {
        "n_expressed_genes_cpm10": int(expr.sum()),
        "n_treated_wells": int(len(trt)),
        "n_treated_drug_dose": int(trt.groupby(["drug", "conc_uM"]).ngroups),
        "treated_rms_lfc_quantiles": q(trt.rms_lfc),
        "treated_n_lfc05_quantiles": q(trt.n_lfc05),
                "noise_model": noise_model,
        "treated_R_vs_sampling_quantiles": q(trt.R_vs_sampling),
        "treated_R_vs_dmso_well_median_quantiles": q(trt.R_vs_dmso_well_median),
        "null_dmso_well_vs_well": {"rms_lfc_max": float(wb.rms_lfc.max()), "rms_lfc_median": float(wb.rms_lfc.median()),
                                   "n_lfc05_max": int(wb.n_lfc05.max()), "n_lfc05_median": float(wb.n_lfc05.median())},
        "null_plate_vs_plate_dmso_pool": {"rms_lfc_median": float(pc.rms_lfc.median()), "rms_lfc_max": float(pc.rms_lfc.max()),
                                          "n_lfc05_median": float(pc.n_lfc05.median()), "n_lfc05_max": int(pc.n_lfc05.max())},
        "frac_wells_resp_strong": float(trt.resp_strong.mean()),
        "frac_wells_resp_strict": float(trt.resp_strict.mean()),
        "frac_drugs_resp_any_dose_strong": float(trt.groupby("drug").resp_strong.any().mean()),
        "n_drug_dose_resp_strong_any_well": int(trt.groupby(["drug", "conc_uM"]).resp_strong.any().sum()),
        "n_drug_dose_resp_strict_any_well": int(trt.groupby(["drug", "conc_uM"]).resp_strict.any().sum()),
        "n_drugs_resp_strict_any_dose": int(trt.groupby("drug").resp_strict.any().sum()),
        "frac_drugs_resp_any_dose_strict": float(trt.groupby("drug").resp_strict.any().mean()),
        "frac_resp_strong_by_dose": trt.groupby("conc_uM").resp_strong.mean().round(4).to_dict(),
        "frac_resp_strict_by_dose": trt.groupby("conc_uM").resp_strict.mean().round(4).to_dict(),
        "wells_lt50_cells": int(trt.low_cells_lt50.sum()),
        "replicate_pairs": int(len(rep)),
        "replicate_r_median_all": float(rep.r.median()) if len(rep) else None,
        "replicate_r_median_both_resp": float(rep[rep.both_resp].r.median()) if len(rep) and rep.both_resp.any() else None,
        "replicate_pairs_both_resp": int(rep.both_resp.sum()) if len(rep) else 0,
        "null_r_diffdrug_diffplate_median": float(null_r.r.median()),
        "null_r_diffdrug_diffplate_q95": float(null_r.r.quantile(0.95)),
        "null_r_both_resp_median": float(null_r[null_r.both_resp].r.median()) if null_r.both_resp.any() else None,
        "dose_pairs_0p5_vs_5": int(len(dose)),
        "dose_r_median_all": float(dose.r.median()) if len(dose) else None,
        "dose_r_median_resp5": float(dose[dose.resp5].r.median()) if len(dose) and dose.resp5.any() else None,
        "dose_pairs_same_plate": int(dose.same_plate.sum()) if len(dose) else 0,
        "frac_drugs_rms_monotone_in_dose": float(dose.rms_increasing.dropna().mean()) if len(dose) else None,
        "spearman_excess_var_vs_cell_ratio": float(trt[["excess_var", "cell_ratio_vs_plate_dmso"]].corr("spearman").iloc[0, 1]),
        "top_moa_by_median_excess_var": trt.groupby("moa_fine").excess_var.median().sort_values(ascending=False).head(8).round(4).to_dict(),
    }
    rep.to_csv(out_dir / f"{tag}_replicate_pairs.tsv", sep="\t", index=False, float_format="%.4g")
    return eff, summary, lfc_store, genes[expr]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    A, R = Path(args.asset_dir), Path(args.out_dir)
    res, stores = {}, {}
    for tag in LINES:
        if not (A / f"{tag}_pseudobulk.npz").exists():
            continue
        summ = pd.read_csv(R / f"{tag}_drug_dose_plate_summary.tsv", sep="\t")
        eff, s, st, g = analyse_line(A, R, tag, summ)
        res[tag] = s
        stores[tag] = (eff, st, g)

    if len(stores) == 2:  # SNU-423 vs HepG2/C3A from the same wells
        (e1, s1, g1), (e2, s2, g2) = stores["snu423"], stores["hepg2c3a"]
        common = np.intersect1d(g1, g2)
        i1 = pd.Index(g1).get_indexer(common); i2 = pd.Index(g2).get_indexer(common)
        m = e1[~e1.is_dmso].merge(e2[~e2.is_dmso], on=["sample"], suffixes=("_s", "_h"))
        m["r_same_well"] = [np.corrcoef(s1[x][i1], s2[x][i2])[0, 1] for x in m["sample"]]
        rng = np.random.default_rng(1)
        nul = []
        for x in m["sample"]:
            pl = m.loc[m["sample"] == x, "plate_s"].iloc[0]
            cand = m[(m.plate_s == pl) & (m["sample"] != x) & (m.drug_s != m.loc[m["sample"] == x, "drug_s"].iloc[0])]["sample"].to_numpy()
            if len(cand):
                y = rng.choice(cand)
                nul.append(np.corrcoef(s1[x][i1], s2[y][i2])[0, 1])
        both = m.resp_strict_s & m.resp_strict_h
        res["snu423_vs_hepg2c3a"] = {
            "n_common_expressed_genes": int(len(common)), "n_wells": int(len(m)),
            "r_same_well_median_all": float(m.r_same_well.median()),
            "r_same_well_median_both_resp": float(m[both].r_same_well.median()) if both.any() else None,
            "n_wells_both_resp": int(both.sum()),
            "null_r_same_plate_diff_drug_median": float(np.median(nul)), "null_r_q95": float(np.quantile(nul, 0.95)),
            "spearman_excess_var_snu_vs_hepg2": float(m[["excess_var_s", "excess_var_h"]].corr("spearman").iloc[0, 1]),
            "resp_strict_concordance_snu_rows_hepg2_cols": pd.crosstab(m.resp_strict_s, m.resp_strict_h).to_dict(),
        }
        m[["sample", "drug_s", "conc_uM_s", "plate_s", "rms_lfc_s", "rms_lfc_h", "excess_var_s", "excess_var_h",
           "resp_strict_s", "resp_strict_h", "r_same_well"]].to_csv(R / "snu423_vs_hepg2c3a_same_well.tsv", sep="\t",
                                                          index=False, float_format="%.4g")

    pb = R / "tahoe_pbde_condition_summary.tsv.gz"
    if pb.exists():
        d = pd.read_csv(pb, sep="\t")
        d = d[d.drug.astype(str).str.strip() != "DMSO_TF"]
        d["concentration"] = d.concentration.astype(float).round(4)
        for k in [10, 100]:
            d[f"resp{k}"] = d.n_sig05_lfc05 >= k
        # Tahoe DESeq2 calls are not null-calibrated (hundreds of genes even for inert drugs), so use a
        # within-line relative criterion: >= 3x the line's median n_sig05_lfc05
        d["strong3x"] = d.n_sig05_lfc05 >= 3 * d.groupby("Cell_ID_Cellosaur").n_sig05_lfc05.transform("median")
        line = d.groupby("Cell_ID_Cellosaur").agg(n_cond=("drug", "size"), n_drugs=("drug", "nunique"),
                                                  median_sig05=("n_sig05", "median"),
                                                  median_sig05_lfc05=("n_sig05_lfc05", "median"),
                                                  frac_cond_ge10=("resp10", "mean"), frac_cond_ge100=("resp100", "mean"),
                                                  frac_cond_strong3x=("strong3x", "mean"),
                                                  median_cells_trt=("n_cells_trt", "median"),
                                                  median_cells_ctrl=("n_cells_ctrl", "median")).sort_values("median_sig05", ascending=False)
        line.to_csv(R / "tahoe_pbde_line_summary.tsv", sep="\t", float_format="%.4g")
        snu = d[d.Cell_ID_Cellosaur == "CVCL_0366"].copy()
        snu["drug"] = snu.drug.str.strip()
        other = d[d.Cell_ID_Cellosaur != "CVCL_0366"].copy(); other["drug"] = other.drug.str.strip()
        n_other_resp = other.groupby(["drug", "concentration"]).strong3x.sum().rename("n_other_lines_strong3x")
        snu = snu.join(n_other_resp, on=["drug", "concentration"])
        res["tahoe_pbde"] = {
            "n_lines": int(d.Cell_ID_Cellosaur.nunique()), "n_conditions": int(len(d)),
            "snu423_rank_by_median_sig05_of_lines": int(list(line.index).index("CVCL_0366") + 1) if "CVCL_0366" in line.index else None,
            "hepg2c3a_rank_by_median_sig05_of_lines": int(list(line.index).index("CVCL_1098") + 1) if "CVCL_1098" in line.index else None,
            "snu423": line.loc["CVCL_0366"].round(4).to_dict() if "CVCL_0366" in line.index else None,
            "hepg2c3a": line.loc["CVCL_1098"].round(4).to_dict() if "CVCL_1098" in line.index else None,
            "all_lines_median_frac_cond_ge10": float(line.frac_cond_ge10.median()),
            "snu423_conditions_strong3x": int(snu.strong3x.sum()),
            "snu423_strong3x_and_strong_in_ge5_other_lines": int(((snu.strong3x) & (snu.n_other_lines_strong3x >= 5)).sum()),
            "snu423_strong3x_and_strong_in_ge10_other_lines": int(((snu.strong3x) & (snu.n_other_lines_strong3x >= 10)).sum()),
            "snu423_drug_dose_resp100": int(snu.resp100.sum()),
        }
        if "snu423" in stores:
            e = stores["snu423"][0]
            e = e[~e.is_dmso].copy()
            e["plate_num"] = e.plate.str.replace("plate", "").astype(str)
            e["conc_uM"] = e.conc_uM.astype(float).round(4)
            snu["plate_num"] = snu.plate.astype(str).str.replace("plate", "")
            mm = e.merge(snu.drop(columns=["plate"]), left_on=["drug", "conc_uM", "plate_num"], right_on=["drug", "concentration", "plate_num"])
            res["tahoe_pbde"]["n_matched_to_own_pseudobulk"] = int(len(mm))
            res["tahoe_pbde"]["own_resp_strict_and_strong_in_ge5_other_lines"] = int((mm.resp_strict & (mm.n_other_lines_strong3x >= 5)).sum())
            res["tahoe_pbde"]["own_resp_strict_and_strong_in_ge10_other_lines"] = int((mm.resp_strict & (mm.n_other_lines_strong3x >= 10)).sum())
            res["tahoe_pbde"]["own_resp_strict_vs_tahoe_strong3x"] = pd.crosstab(mm.resp_strict, mm.strong3x).to_dict()
            res["tahoe_pbde"]["spearman_own_excess_var_vs_tahoe_nsig05"] = float(mm[["excess_var", "n_sig05"]].corr("spearman").iloc[0, 1])
            res["tahoe_pbde"]["own_resp_strict_vs_tahoe_nsig05_lfc05_ge100"] = pd.crosstab(mm.resp_strict, mm.n_sig05_lfc05 >= 100).to_dict()
            mm[["sample", "drug", "conc_uM", "plate", "n_cells", "rms_lfc", "excess_var", "R_vs_sampling", "resp_strict", "resp_strong",
                "n_sig05", "n_sig05_lfc05", "strong3x", "n_cells_trt", "n_cells_ctrl", "n_other_lines_strong3x"]].to_csv(
                R / "snu423_effect_vs_tahoe_pbde.tsv", sep="\t", index=False, float_format="%.4g")
    (R / "f4_effect_summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
