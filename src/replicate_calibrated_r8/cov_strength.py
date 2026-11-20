"""Revision R-2 (diagnostic stage, closed form only): can ANY strength or structure of the replicate / shared-plate observation
model beat equal weights for the common correction on source-train dev episodes?

Variants: the signal-variance term v_r multiplied by 1 (pre-registered model), 4, 16, 64, 256 (larger -> closer to equal weights),
crossed with the structure: full / diag_only (no shared-plate block) / block_only (equal diagonal). kappa_f stays at the selected
value; kappa_b is re-selected per variant and K on the same dev episodes (this favours the variants). No confirmation family.
"""
import json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from .common import CFG, public, dump, man_file
from .data import Store
from . import methods as M
from .contrasts import family_contrasts
from src.episodic_residual_r2.evalcore import family_mean

SCALES = [1.0, 4.0, 16.0, 64.0, 256.0]
STRUCT = ["full", "diag_only", "block_only"]


def main():
    cfg = yaml.safe_load(open(CFG))
    ec = cfg["episodes"]
    KB = [float(x) for x in cfg["anchor"]["kappa_b_grid"]]
    sel = json.load(open(man_file("anchor_selection.json")))
    cov = sel["covariance_model"]
    t0 = time.time()
    st = Store("cpu")
    rm = lambda p, ep: torch.sqrt(((p - ep.y_q) ** 2).mean(1)).mean().item()
    rows = []
    with torch.no_grad():
        for ep in st.episodes("train", ec["dev_support_seeds"], ec["dev_query_cap"]):
            for K in ec["k_values"]:
                if K > len(ep.g_s):
                    continue
                inp = ep.inputs(K)
                b = dict(cell=ep.name, family=ep.family, support_seed=ep.seed, K=K)
                e = sel["anchor"]["iid"][str(K)]
                rows.append(dict(variant="iid", kappa_b=e["kappa_b"], delta_rmse=rm(M.c0(M.anchor(inp, e["kappa_f"], e["kappa_b"], None)), ep), k_eff=float(K), **b))
                for sc in SCALES:
                    for stt in STRUCT:
                        anc = M.anchor(inp, e["kappa_f"], 0.0, dict(cov, v_r_scale=sc, structure=stt))
                        for kb in KB:
                            sh = 0.0 if np.isinf(kb) else anc["keff"] / (anc["keff"] + kb)
                            rows.append(dict(variant=f"{stt}|x{sc:g}", kappa_b=kb, delta_rmse=rm(anc["fq"] + anc["ebar"] * sh, ep), k_eff=anc["keff"], **b))
    df = pd.DataFrame(rows)
    agg, _ = family_mean(df, by=("K", "variant", "kappa_b"))
    best = agg.sort_values(["delta_rmse", "kappa_b"]).groupby(["K", "variant"]).head(1)
    keep = df.merge(best[["K", "variant", "kappa_b"]], on=["K", "variant", "kappa_b"])
    keep = keep.rename(columns={"variant": "method"})
    _, fam = family_mean(keep)
    names = [v for v in keep.method.unique() if v != "iid"]
    cons = family_contrasts(fam, "delta_rmse", contrasts={f"{v}-iid": {v: 1.0, "iid": -1.0} for v in names})
    tab = best.rename(columns={"variant": "method"}).merge(cons.assign(method=cons.contrast.str[:-4])[["K", "method", "diff", "lo", "hi", "n_families_negative"]], on=["K", "method"], how="left")
    ke = keep.groupby(["K", "method"]).k_eff.mean().rename("k_eff_mean").reset_index()
    tab = tab.merge(ke, on=["K", "method"]).sort_values(["K", "delta_rmse"])
    tab.to_csv(public("manifests", "replicate_calibrated_r8", "cov_strength_dev.tsv"), sep="\t", index=False, float_format="%.6g")
    k16 = tab[(tab.K == 16) & (tab.method != "iid")].iloc[0]
    adopt = bool(k16.hi < 0)
    dump(dict(rule="adopt the best K=16 variant only if its paired family interval against equal weights lies below 0 on source-train dev episodes",
              best_k16=dict(variant=k16.method, delta_rmse=float(k16.delta_rmse), diff_vs_iid=float(k16["diff"]), lo=float(k16.lo), hi=float(k16.hi), kappa_b=float(k16.kappa_b)),
              adopt=adopt, seconds=round(time.time() - t0, 1), commit=os.environ.get("R8_COMMIT", "")), public("manifests", "replicate_calibrated_r8", "cov_strength_decision.json"))
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(tab.to_string())
    print("adopt:", adopt)


if __name__ == "__main__":
    main()
