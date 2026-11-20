"""R2-0 / D1: source-only baselines on fixed episodes and the lambda selection of the closed-form residual baselines.

Meta-train dev episodes (leave-family-out f0) pick lambda; meta-validation episodes are only reported.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from .common import CFG, public, runs, dump, new_run_dir, run_record
from .data import Source, Features
from . import methods as M
from .evalcore import score, family_mean, paired_family


def episodes_for(src, role, seeds, cap=None):
    eps = []
    for ci in src.eval_cells(role):
        for s in seeds:
            eps.append(src.eval_episode(ci, s, query_cap=cap))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="closed_form_v1")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    t0 = time.time()
    out = new_run_dir("closed_form", a.run_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = Source(dev)
    fmap = Features.fit(src)
    fpath = os.path.join(os.environ["R2_PROC"], "feature_basis.npz")
    if not os.path.exists(fpath):
        fmap.save(fpath)
    lams = cfg["baselines"]["lambda_grid"]
    ep_cfg = cfg["episodes"]
    sets = dict(meta_train_dev=episodes_for(src, "train", ep_cfg["meta_train_dev_support_seeds"], ep_cfg["dev_query_cap"]),
                meta_val=episodes_for(src, "val", ep_cfg["meta_val_support_seeds"]))
    rows = []
    with torch.no_grad():
        for sname, eps in sets.items():
            for ep in eps:
                for K in ep_cfg["k_values"]:
                    if K > len(ep.g_s):
                        continue
                    inp = ep.inputs(K)
                    ft = fmap.build(inp, src.FP)
                    preds = dict(no_change=M.no_change(inp), source_same_drug_mean=M.same_drug_mean(inp), mean_offset=M.mean_offset(inp))
                    for name, f in M.CLOSED_FORM.items():
                        for lam in lams:
                            preds[f"{name}|{lam:g}"] = f(inp, ft, float(lam))
                    for name, p in preds.items():
                        base, _, lam = name.partition("|")
                        r, _ = score(ep, K, p, dict(set=sname, method=base, lam=float(lam) if lam else np.nan, n_query=len(ep.g_q),
                                                    n_f0_cells_median=float(ep.n0_q.median().item())))
                        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "closed_form_long.tsv"), sep="\t", index=False)
    # lambda curves (family-equal) per set, K, method
    df["method_lam"] = df.method + np.where(df.lam.notna(), "|" + df.lam.map(lambda v: f"{v:g}"), "")
    curves = []
    for sname, g in df.groupby("set"):
        agg, _ = family_mean(g, by=("K", "method", "lam"))
        agg["set"] = sname
        curves.append(agg)
    curves = pd.concat(curves)
    curves.to_csv(public("manifests", "episodic_residual_r2", "d1_lambda_curves.tsv"), sep="\t", index=False)
    # selection on meta-train dev episodes only
    sel = {}
    dv = curves[(curves.set == "meta_train_dev") & curves.lam.notna()]
    for (K, m), g in dv.groupby(["K", "method"]):
        best = g.sort_values(["delta_rmse", "lam"], ascending=[True, False]).iloc[0]      # ties -> stronger shrinkage
        sel[f"{m}|{K}"] = dict(lam=float(best.lam), dev_delta_rmse=float(best.delta_rmse), at_grid_edge=bool(best.lam in (lams[0], lams[-1])))
    # report table with the selected lambda, both sets
    keep = df.lam.isna()
    for key, e in sel.items():
        m, K = key.split("|")
        keep |= (df.method == m) & (df.K == int(K)) & np.isclose(df.lam, e["lam"])
    rep = df[keep].copy()
    rep.drop(columns=["method_lam"]).to_csv(public("manifests", "episodic_residual_r2", "d1_closed_form_episodes.tsv"), sep="\t", index=False, float_format="%.6g")
    summ, contrasts = [], []
    metrics = ["delta_rmse", "centred_rmse", "centred_pearson", "delta_pearson", "retrieval_mrr_centred", "retrieval_mrr", "delta_rmse_shared_plate", "delta_rmse_disjoint_plate"]
    for sname, g in rep.groupby("set"):
        parts = None
        for v in metrics:
            agg, fam = family_mean(g, value=v)
            parts = agg if parts is None else parts.merge(agg.drop(columns="n_families"), on=["K", "method"])
            if v == "delta_rmse":
                for m in sorted(set(g.method) - {"mean_offset"}):
                    c = paired_family(fam, m, "mean_offset")
                    c["set"] = sname
                    contrasts.append(c)
        parts["set"] = sname
        summ.append(parts)
    summ = pd.concat(summ)
    summ.to_csv(public("manifests", "episodic_residual_r2", "d1_closed_form_summary.tsv"), sep="\t", index=False, float_format="%.6g")
    contrasts = pd.concat(contrasts)
    contrasts.to_csv(public("manifests", "episodic_residual_r2", "d1_contrasts_vs_mean_offset.tsv"), sep="\t", index=False, float_format="%.6g")
    res = run_record(a.run_id, "closed_form", t0, n_dev_episodes=len(sets["meta_train_dev"]), n_val_episodes=len(sets["meta_val"]),
                     dev_families=int(df[df.set == "meta_train_dev"].family.nunique()), val_families=int(df[df.set == "meta_val"].family.nunique()),
                     selected=sel, feature_basis=fpath)
    dump(res, os.path.join(out, "result.json"))
    dump(dict(selected=sel, rule=cfg["baselines"]["lambda_selection"], run_id=a.run_id, commit=res["commit"]),
         public("manifests", "episodic_residual_r2", "baseline_lambda_selection.json"))
    pd.set_option("display.width", 250)
    print(summ.to_string())
    print(contrasts.to_string())
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
