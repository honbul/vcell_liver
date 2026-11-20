"""D3 and the source-side selection of the TranSiGen fine-tuning settings (meta-validation virtual targets only).

Grid: learning rate x BatchNorm handling x number of updates, per K; search seed 42, support seed 1.
The BatchNorm contrast is a local check of one optimisation choice, not a causal analysis.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from .common import CFG, public, dump, new_run_dir, run_record
from .data import Source
from .evalcore import score, family_mean, paired_family
from . import tsg as G
from src.transigen_adaptation_b1.data import Data
from src.transigen_adaptation_b1 import tsg_adapter as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="tsg_source_v1")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    tc = cfg["baselines"]["transigen"]["full_ft"]
    t0 = time.time()
    out = new_run_dir("tsg_source", a.run_id)
    dev = torch.device("cuda")
    torch.set_num_threads(1)
    src = Source(dev)
    B = Data()
    seed = cfg["proposed"]["search_seed"]
    ck = G.checkpoints()[seed]
    base, _ = T.load_checkpoint(ck["checkpoint"], dev, seed=seed)
    ks = cfg["episodes"]["k_values"]
    rows, n_div = [], 0
    for ci in src.eval_cells("val"):
        ep = src.eval_episode(ci, 1)
        ra = G.source_row_access(B, src, ep.name)
        as_is = G.predict_units(base, ra, ep.g_q, dev)
        for K in ks:
            r, _ = score(ep, K, as_is, dict(method="transigen_as_is", lr=np.nan, bn="", updates=0, status="ok"))
            rows.append(r)
            ps = G.predict_units(base, ra, ep.g_s[:K], dev)
            r, _ = score(ep, K, as_is + (ep.y_s[:K] - ps).mean(0, keepdim=True), dict(method="transigen_offset", lr=np.nan, bn="", updates=0, status="ok"))
            rows.append(r)
            for lr in tc["lr_grid"]:
                for bn in tc["batchnorm"]:
                    preds, status, info = G.fine_tune(base, ep, K, ra, lr, bn, tc["update_grid"], seed, dev)
                    n_div += status != "ok"
                    for u in tc["update_grid"]:
                        if u in preds:
                            r, _ = score(ep, K, preds[u], dict(method="transigen_full_ft", lr=lr, bn=bn, updates=u, status=status))
                        else:
                            r = dict(cell=ep.name, family=ep.family, support_seed=ep.seed, K=K, method="transigen_full_ft", lr=lr, bn=bn, updates=u,
                                     status=status, delta_rmse=np.inf)
                        rows.append(r)
            print(ep.name, K, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "tsg_source_long.tsv"), sep="\t", index=False, float_format="%.6g")
    ft = df[df.method == "transigen_full_ft"]
    grid, fam = family_mean(ft, by=("K", "lr", "bn", "updates"))
    div = ft.groupby(["K", "lr", "bn", "updates"]).status.agg(lambda s: int((s != "ok").sum())).reset_index().rename(columns={"status": "n_diverged_episodes"})
    grid = grid.merge(div, on=["K", "lr", "bn", "updates"])
    ref, _ = family_mean(df[df.method != "transigen_full_ft"], by=("K", "method"))
    grid.to_csv(public("manifests", "episodic_residual_r2", "d3_tsg_ft_grid.tsv"), sep="\t", index=False, float_format="%.6g")
    ref.to_csv(public("manifests", "episodic_residual_r2", "d3_tsg_reference.tsv"), sep="\t", index=False, float_format="%.6g")
    sel = {}
    for K, g in grid.groupby("K"):
        b = g.sort_values(["delta_rmse", "lr", "updates"]).iloc[0]
        sel[str(int(K))] = dict(lr=float(b.lr), bn=b.bn, updates=int(b.updates), source_delta_rmse=float(b.delta_rmse),
                                as_is=float(ref[(ref.K == K) & (ref.method == "transigen_as_is")].delta_rmse.iloc[0]),
                                offset=float(ref[(ref.K == K) & (ref.method == "transigen_offset")].delta_rmse.iloc[0]))
    # D3: BatchNorm contrast, paired by family, for every (K, lr, updates)
    fam["method"] = fam.bn
    d3 = []
    for (lr, u), g in fam.groupby(["lr", "updates"]):
        g = g[np.isfinite(g.delta_rmse)]
        c = paired_family(g, "frozen_running_stats", "train_mode")
        c["lr"], c["updates"] = lr, u
        d3.append(c)
    d3 = pd.concat(d3)
    d3.to_csv(public("manifests", "episodic_residual_r2", "d3_batchnorm_contrast.tsv"), sep="\t", index=False, float_format="%.6g")
    res = run_record(a.run_id, "tsg_source_selection", t0, checkpoint=ck["run_id"], seed=seed, support_seed=1, n_cells=len(src.eval_cells("val")),
                     n_ft_runs=int(len(ft) / len(tc["update_grid"])), n_ft_runs_with_divergence=int(n_div), selected=sel)
    dump(res, os.path.join(out, "result.json"))
    dump(dict(selected=sel, rule=tc["selection"], run_id=a.run_id, commit=res["commit"], n_ft_runs=res["n_ft_runs"], n_ft_runs_with_divergence=res["n_ft_runs_with_divergence"]),
         public("manifests", "episodic_residual_r2", "tsg_ft_selection.json"))
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(grid.to_string())
    print(ref.to_string())
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
