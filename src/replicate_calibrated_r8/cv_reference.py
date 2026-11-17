"""Closed-form references (C0, C1, simple baselines) on exactly the fold episodes the cross-validation curves are scored on.
Lets a cross-validation score be read against the common correction without touching a confirmation family."""
import time
import pandas as pd
import torch
import yaml

from .common import CFG, public
from .data import Store, Features
from .arms import Context
from . import methods as M
from src.episodic_residual_r2.evalcore import family_mean


def main():
    cfg = yaml.safe_load(open(CFG))
    ec = cfg["episodes"]
    rows = []
    t0 = time.time()
    for fold in range(cfg["inner_cv"]["folds"] if "inner_cv" in cfg and "folds" in cfg["inner_cv"] else 3):
        st = Store("cpu", fold=fold)
        ctx = Context(Features.fit(st), st.FP)
        rm = lambda p, ep: torch.sqrt(((p - ep.y_q) ** 2).mean(1)).mean().item()
        with torch.no_grad():
            for ep in st.episodes("val", ec["dev_support_seeds"], ec["dev_query_cap"]):
                for K in ec["k_values"]:
                    inp = ep.inputs(K)
                    preds = dict(no_change=M.no_change(inp), same_drug_mean=M.same_drug_mean(inp), mean_offset=M.mean_offset(inp), C0_iid=ctx.c0(inp, "iid"),
                                 C1_iid=ctx.c1(inp, "iid"), C0_cov=ctx.c0(inp, "cov"), C1_cov=ctx.c1(inp, "cov"))
                    for m, p in preds.items():
                        rows.append(dict(fold=fold, cell=ep.name, family=ep.family, support_seed=ep.seed, K=K, method=m, delta_rmse=rm(p, ep)))
        print("fold", fold, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    agg, _ = family_mean(df, by=("fold", "K", "method"))
    mean = agg.groupby(["K", "method"]).delta_rmse.mean().reset_index()
    mean.insert(0, "fold", "mean")
    out = pd.concat([agg.drop(columns="n_families").astype({"fold": str}), mean])
    out.to_csv(public("manifests", "replicate_calibrated_r8", "cv_reference.tsv"), sep="\t", index=False, float_format="%.6g")
    print(mean.pivot(index="method", columns="K", values="delta_rmse").to_string())


if __name__ == "__main__":
    main()
