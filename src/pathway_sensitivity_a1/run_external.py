#!/usr/bin/env python3
"""Stage A4: LCCL external evaluation with frozen settings.

predict : train on LIMORE only and predict LCCL models from LCCL RNA.
          - new lineage: LCCL family absent from LIMORE -> model trained on all eligible LIMORE models
          - cross-study leave-family-out: LCCL family present in LIMORE -> that family is removed from
            LIMORE training and from alpha selection before predicting
evaluate: rank agreement per matched drug between predicted AA and LCCL sensitivity (-AUC).
          AA and AUC/GI50 values are never pooled or put on a common scale.

Usage: run_external.py predict  --run-dir RUN
       run_external.py evaluate --run-dir RUN --out-dir reports/pathway_sensitivity_a1/tables
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
import metrics as M  # noqa: E402
import nested  # noqa: E402
from run_internal import features  # noqa: E402


def setup(cfg):
    reg = core.load_registry()
    dm = pd.read_csv(os.path.join(core.MANIFESTS, "drug_match_lccl.tsv"), sep="\t").fillna("")
    dm = dm[dm.match_status == "matched"]
    src = reg[(reg.study == "LIMORE") & reg.included].sort_values("model_id")
    tgt = reg[(reg.study == "LCCL") & reg.included].sort_values("model_id").copy()
    tgt["eval_type"] = np.where(tgt.family_id.isin(set(src.family_id)), "cross_study_leave_family_out", "new_lineage")
    return reg, dm, src, tgt


def cmd_predict(args):
    cfg = core.load_config()
    reg, dm, src, tgt = setup(cfg)
    limore, lccl = core.load_study(cfg, "LIMORE"), core.load_study(cfg, "LCCL")
    assert limore.sets == lccl.sets
    long = core.load_limore_response(cfg)
    drugs = sorted(dm.limore_name)
    src_models = list(src.model_id)
    fam = src.family_id.values
    Y = core.response_matrix(long, src_models, drugs, "y")
    row_l = {m: i for i, m in enumerate(limore.models)}
    row_c = {m: i for i, m in enumerate(lccl.models)}
    out = os.path.join(args.run_dir, "external", "predictions.tsv.gz")
    if os.path.exists(out):
        raise FileExistsError(f"refusing to overwrite {out}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames = []
    for kind in cfg["external"]["models"]:
        Xs = features(cfg, limore, kind, None)[[row_l[m] for m in src_models]]
        Xt_all = features(cfg, lccl, kind, None)
        # one training set per excluded family; "" = nothing excluded (new lineages)
        for fam_x, t in tgt.assign(key=np.where(tgt.eval_type == "new_lineage", "", tgt.family_id)).groupby("key"):
            rows = fam != fam_x
            if fam_x:
                assert (~rows).sum() >= 1 and fam_x not in set(fam[rows])
            else:
                assert rows.all() and not set(t.family_id) & set(fam)
            Xt = Xt_all[[row_c[m] for m in t.model_id]]
            pred, info = nested.fit_predict_missing(
                Xs[rows], Y[rows], fam[rows], Xt, kind=kind, alphas=cfg["model"]["alpha_grid"],
                n_inner=cfg["split"]["inner_folds"], inner_seed=cfg["external"]["inner_seed"], K=len(limore.sets),
                const_sd=cfg["endpoint"]["constant_response_sd"])
            df = pd.DataFrame(pred, index=list(t.model_id), columns=drugs).rename_axis("model_id").reset_index()
            df = df.melt(id_vars="model_id", var_name="drug_id", value_name="pred")
            df = df.merge(pd.DataFrame(dict(drug_id=drugs, alpha=info["alpha"], train_mean=info["mu"],
                                            s_train=info["sd"])), on="drug_id")
            df["kind"] = kind
            df["excluded_limore_family"] = fam_x
            df["n_train_models"] = int(rows.sum())
            df["pca_dim"] = info["pca_dim"][0] if kind == "A2" else np.nan
            frames.append(df)
    res = pd.concat(frames, ignore_index=True)
    res = res.merge(tgt[["model_id", "family_id", "eval_type"]], on="model_id", validate="many_to_one")
    res = res.merge(dm[["limore_name", "lccl_name"]].rename(columns={"limore_name": "drug_id"}), on="drug_id",
                    validate="many_to_one")
    assert not res.duplicated(["kind", "model_id", "drug_id"]).any()
    assert len(res) == len(cfg["external"]["models"]) * len(tgt) * len(drugs)
    # leave-family-out really left the family out
    x = res[res.eval_type == "cross_study_leave_family_out"]
    assert (x.excluded_limore_family == x.family_id).all()
    res.to_csv(out + ".tmp.gz", sep="\t", index=False, float_format="%.8g", compression="gzip")
    os.replace(out + ".tmp.gz", out)
    print(res.groupby(["kind", "eval_type"]).model_id.nunique())


def spearman_table(d: pd.DataFrame, value: str, min_n: int) -> pd.DataFrame:
    """d: family-level rows with columns drug_id, family_id, pred, <value> (higher = more sensitive)."""
    rows = []
    for drug, t in d.groupby("drug_id"):
        t = t.dropna(subset=["pred", value])
        rho = M.safe_spearman(t.pred, t[value]) if len(t) >= min_n else float("nan")
        rows.append(dict(drug_id=drug, n_families=len(t), spearman=rho))
    return pd.DataFrame(rows)


def boot_mean_spearman(d: pd.DataFrame, value: str, min_n: int, n_boot: int, seed: int) -> tuple:
    fams = np.array(sorted(d.family_id.unique()))
    by = {f: t for f, t in d.groupby("family_id")}
    rs = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        pick = rs.choice(fams, size=len(fams), replace=True)
        b = pd.concat([by[f] for f in pick], ignore_index=True)
        out.append(spearman_table(b, value, min_n).spearman.mean())
    return M.ci(np.array(out))


def cmd_evaluate(args):
    cfg = core.load_config()
    reg, dm, src, tgt = setup(cfg)
    ext = cfg["external"]
    min_n = ext["min_families_for_spearman"]
    pred = pd.read_csv(os.path.join(args.run_dir, "external", "predictions.tsv.gz"), sep="\t")
    obs = pd.read_csv(os.path.join(core.cache_dir(cfg), "lccl__response_long.tsv"), sep="\t")
    n = len(pred)
    df = pred.merge(obs, on=["model_id", "lccl_name"], how="left", validate="many_to_one")
    assert len(df) == n
    gi_ceiling = obs.GI50.max()
    df["sens_auc"] = -df.AUC                       # direction only; no rescaling
    df["sens_gi50"] = -df.GI50
    os.makedirs(args.out_dir, exist_ok=True)

    # study descriptors per matched drug
    cen = obs[obs.lccl_name.isin(set(dm.lccl_name)) & obs.model_id.isin(set(tgt.model_id))]
    desc = cen.groupby("lccl_name").agg(n_models=("AUC", "count"), auc_min=("AUC", "min"), auc_max=("AUC", "max"),
                                        gi50_at_ceiling_fraction=("GI50", lambda v: float((v >= gi_ceiling).mean()))
                                        ).reset_index()

    per_drug, summary = [], []
    for (kind, et), t in df.groupby(["kind", "eval_type"]):
        fam = t.groupby(["drug_id", "family_id"])[["pred", "sens_auc", "sens_gi50"]].mean().reset_index()
        for value in ("sens_auc", "sens_gi50"):
            tab = spearman_table(fam, value, min_n)
            tab.insert(0, "endpoint", {"sens_auc": "AUC", "sens_gi50": "GI50"}[value])
            tab.insert(0, "eval_type", et)
            tab.insert(0, "model", kind)
            per_drug.append(tab)
            lo, hi = boot_mean_spearman(fam, value, min_n, cfg["bootstrap"]["n"], ext["bootstrap_seed"])
            summary.append(dict(model=kind, eval_type=et, endpoint=tab.endpoint[0],
                                n_families=int(fam.family_id.nunique()), n_models=int(t.model_id.nunique()),
                                n_drugs_scored=int(tab.spearman.notna().sum()), mean_spearman=tab.spearman.mean(),
                                median_spearman=tab.spearman.median(), ci_low=lo, ci_high=hi,
                                n_drugs_positive=int((tab.spearman > 0).sum())))
    per_drug = pd.concat(per_drug, ignore_index=True)
    summary = pd.DataFrame(summary)

    # measured-vs-measured agreement for shared families (no model): LIMORE AA vs LCCL -AUC
    long = core.load_limore_response(cfg)
    fam_of = dict(zip(reg.model_id, reg.family_id))
    la = long[long.model_id.isin(set(src.model_id)) & long.drug_id.isin(set(dm.limore_name))].copy()
    la["family_id"] = la.model_id.map(fam_of)
    la = la.groupby(["drug_id", "family_id"]).y.mean().rename("pred").reset_index()       # AA plays the 'prediction'
    shared = df[(df.eval_type == "cross_study_leave_family_out") & (df.kind == ext["models"][0])]
    lo_ = shared.groupby(["drug_id", "family_id"])[["sens_auc", "sens_gi50"]].mean().reset_index()
    mm = la.merge(lo_, on=["drug_id", "family_id"], validate="one_to_one")
    conc = spearman_table(mm, "sens_auc", min_n).rename(columns={"spearman": "measured_AA_vs_AUC_spearman"})
    lo, hi = boot_mean_spearman(mm, "sens_auc", min_n, cfg["bootstrap"]["n"], ext["bootstrap_seed"])
    concordance = dict(n_families=int(mm.family_id.nunique()), mean_spearman=float(conc.measured_AA_vs_AUC_spearman.mean()),
                       median_spearman=float(conc.measured_AA_vs_AUC_spearman.median()), ci_low=lo, ci_high=hi)

    wide = per_drug[per_drug.endpoint == "AUC"].pivot(index="drug_id", columns=["eval_type", "model"], values="spearman")
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index().merge(conc[["drug_id", "measured_AA_vs_AUC_spearman"]], on="drug_id", how="left")
    wide = wide.merge(dm[["limore_name", "lccl_name"]].rename(columns={"limore_name": "drug_id"}), on="drug_id")
    wide = wide.merge(desc, on="lccl_name", how="left")
    internal = os.path.join(args.out_dir, "internal_per_drug.tsv")
    if os.path.exists(internal):
        it = pd.read_csv(internal, sep="\t")[["drug_id", "spearman_A1", "spearman_A1r", "spearman_A2", "spearman_A3"]]
        wide = wide.merge(it.rename(columns={c: "internal_oof_" + c for c in it.columns if c != "drug_id"}),
                          on="drug_id", how="left")
    wide.to_csv(os.path.join(args.out_dir, "external_per_drug.tsv"), sep="\t", index=False, float_format="%.4g")
    per_drug.to_csv(os.path.join(args.out_dir, "external_per_drug_long.tsv"), sep="\t", index=False, float_format="%.4g")
    summary.to_csv(os.path.join(args.out_dir, "external_summary.tsv"), sep="\t", index=False, float_format="%.4g")

    # individual points for the new lineages (too few for a summary statistic to carry weight)
    pts = df[df.eval_type == "new_lineage"][["kind", "model_id", "family_id", "drug_id", "lccl_name", "pred"]]
    pts.to_csv(os.path.join(args.out_dir, "external_new_lineage_predictions.tsv"), sep="\t", index=False,
               float_format="%.5g")
    counts = tgt.groupby("eval_type").agg(n_models=("model_id", "nunique"), n_families=("family_id", "nunique"))
    status = "external_validation_limited"
    res = dict(run_dir=os.path.basename(os.path.normpath(args.run_dir)), status=status,
               status_reason="new-lineage evaluation has %d models in %d families; cross-study leave-family-out has %d models in %d families"
               % (counts.loc["new_lineage", "n_models"], counts.loc["new_lineage", "n_families"],
                  counts.loc["cross_study_leave_family_out", "n_models"],
                  counts.loc["cross_study_leave_family_out", "n_families"]),
               n_matched_drugs=int(len(dm)), measured_concordance_shared_families=concordance,
               summary=summary.to_dict(orient="records"))
    with open(os.path.join(args.out_dir, "external_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(summary.to_string())
    print(json.dumps(concordance, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "evaluate"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    {"predict": cmd_predict, "evaluate": cmd_evaluate}[a.cmd](a)
