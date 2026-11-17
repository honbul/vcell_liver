"""Post-hoc source diagnostic (added after R2-2): how much do two scalar shrinkages explain?

  f0_shrunk(p)  = f0(p) * n_p / (n_p + kappa_f)            n_p = number of source cells behind f0
  offset_shrunk = b_S * K / (K + kappa_b)                  b_S = mean support residual against f0_shrunk
kappa_f and kappa_b are picked per K on meta-train dev episodes; meta-validation is reported. No learned model, no HCC value.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from .common import CFG, public, dump, new_run_dir, run_record
from .data import Source
from .evalcore import family_mean
from .closed_form import episodes_for

KF = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
KB = [0.0, 4.0, 8.0, 16.0, 32.0, 64.0, float("inf")]


def shrunk_prediction(inp, kf, kb):
    wq = (inp["n0_q"] / (inp["n0_q"] + kf))[:, None]
    ws = (inp["n0_s"] / (inp["n0_s"] + kf))[:, None]
    base = inp["f0_q"] * wq
    if np.isinf(kb):
        return base
    K = len(inp["y_s"])
    b = (inp["y_s"] - inp["f0_s"] * ws).mean(0, keepdim=True)
    return base + b * (K / (K + kb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="diag_shrink_v1")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    t0 = time.time()
    out = new_run_dir("diag", a.run_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = Source(dev)
    ec = cfg["episodes"]
    sets = dict(meta_train_dev=episodes_for(src, "train", ec["meta_train_dev_support_seeds"], ec["dev_query_cap"]),
                meta_val=episodes_for(src, "val", ec["meta_val_support_seeds"]))
    rows = []
    with torch.no_grad():
        for sname, eps in sets.items():
            for ep in eps:
                for K in ec["k_values"]:
                    if K > len(ep.g_s):
                        continue
                    inp = ep.inputs(K)
                    for kf in KF:
                        for kb in KB:
                            p = shrunk_prediction(inp, kf, kb)
                            rows.append(dict(set=sname, cell=ep.name, family=ep.family, support_seed=ep.seed, K=K, kappa_f=kf, kappa_b=kb,
                                             delta_rmse=torch.sqrt(((p - ep.y_q) ** 2).mean(1)).mean().item()))
    df = pd.DataFrame(rows)
    grids = []
    for sname, g in df.groupby("set"):
        agg, _ = family_mean(g, by=("K", "kappa_f", "kappa_b"))
        agg["set"] = sname
        grids.append(agg)
    grid = pd.concat(grids)
    grid.to_csv(public("manifests", "episodic_residual_r2", "diag_shrink_grid.tsv"), sep="\t", index=False, float_format="%.6g")
    dv, va = grid[grid.set == "meta_train_dev"], grid[grid.set == "meta_val"]
    look = lambda t, K, kf, kb: float(t[(t.K == K) & (t.kappa_f == kf) & (t.kappa_b == kb)].delta_rmse.iloc[0])
    sel = {}
    for K in ec["k_values"]:
        d = dv[dv.K == K]
        b_f = d[np.isinf(d.kappa_b)].sort_values(["delta_rmse", "kappa_f"]).iloc[0]            # no support label used
        b_fb = d.sort_values(["delta_rmse", "kappa_f", "kappa_b"]).iloc[0]
        sel[str(K)] = dict(shrunk_same_drug_mean=dict(kappa_f=float(b_f.kappa_f), dev=float(b_f.delta_rmse), meta_val=look(va, K, b_f.kappa_f, float("inf"))),
                           shrunk_mean_offset=dict(kappa_f=float(b_fb.kappa_f), kappa_b=(float(b_fb.kappa_b) if np.isfinite(b_fb.kappa_b) else "inf"), dev=float(b_fb.delta_rmse),
                                                   meta_val=look(va, K, b_fb.kappa_f, b_fb.kappa_b)),
                           reference_meta_val=dict(source_same_drug_mean=look(va, K, 0.0, float("inf")), mean_offset=look(va, K, 0.0, 0.0)))
    dump(dict(selected=sel, rule="per K, lowest family-equal mean query delta RMSE on meta-train dev episodes", grids=dict(kappa_f=KF, kappa_b=[str(k) for k in KB]),
              run_id=a.run_id, commit=os.environ.get("R2_COMMIT", "")), public("manifests", "episodic_residual_r2", "shrink_selection.json"))
    dump(run_record(a.run_id, "diag_shrink", t0, selected=sel), os.path.join(out, "result.json"))
    print(json.dumps(sel, indent=1))
    pd.set_option("display.width", 250, "display.max_rows", 400)
    print(va[va.K == 16].pivot(index="kappa_f", columns="kappa_b", values="delta_rmse").to_string())


if __name__ == "__main__":
    main()
