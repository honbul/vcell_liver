"""Source-side selection of the TranSiGen fine-tuning settings (frozen BatchNorm statistics; learning rate x number of updates per K)
on confirmation-family virtual targets, search seed 42, support seed 1 - the R2 rule on the retrained checkpoints. No HCC value."""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch

from .common import public, dump, new_run_dir, run_record, runs, sha256
from .data import Store
from src.episodic_residual_r2.evalcore import family_mean
from src.episodic_residual_r2 import tsg as G
from src.transigen_adaptation_b1.data import Data
from src.transigen_adaptation_b1 import tsg_adapter as T

LR = [1e-5, 1e-4, 1e-3]
UPDATES = [25, 50, 100, 200, 400, 800]
BN = "frozen_running_stats"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="tsg_select_v1")
    ap.add_argument("--source-runs", required=True, help="comma-separated tsg_source run ids (seeds 42,43,44)")
    a = ap.parse_args()
    t0 = time.time()
    out = new_run_dir("tsg_select", a.run_id)
    dev = torch.device("cuda")
    torch.set_num_threads(1)
    cks = {}
    for rid in a.source_runs.split(","):
        r = json.load(open(runs("tsg_source", rid, "result.json")))
        assert sha256(r["checkpoint"]) == r["checkpoint_sha256"] and r["stopped"] != "non_finite_loss"
        cks[str(r["seed"])] = dict(run_id=rid, checkpoint=r["checkpoint"], checkpoint_sha256=r["checkpoint_sha256"], best_epoch=r["best_epoch"],
                                   val_delta_rmse_best=r["val_delta_rmse_best"], lr=r["lr"])
    assert sorted(cks) == ["42", "43", "44"]
    st = Store(dev)
    B = Data()
    base, _ = T.load_checkpoint(cks["42"]["checkpoint"], dev, seed=42)
    rm = lambda p, ep: torch.sqrt(((p - ep.y_q) ** 2).mean(1)).mean().item()
    rows, n_div = [], 0
    for ci in st.eval_cells("val"):
        ep = st.eval_episode(ci, 1)
        ra = G.source_row_access(B, st, ep.name)
        as_is = G.predict_units(base, ra, ep.g_q, dev)
        for K in (8, 16, 32):
            b = dict(cell=ep.name, family=ep.family, support_seed=1, K=K)
            rows.append(dict(method="transigen_as_is", lr=np.nan, updates=0, delta_rmse=rm(as_is, ep), **b))
            ps = G.predict_units(base, ra, ep.g_s[:K], dev)
            rows.append(dict(method="transigen_offset", lr=np.nan, updates=0, delta_rmse=rm(as_is + (ep.y_s[:K] - ps).mean(0, keepdim=True), ep), **b))
            for lr in LR:
                preds, status, _ = G.fine_tune(base, ep, K, ra, lr, BN, UPDATES, 42, dev)
                n_div += status != "ok"
                for u in UPDATES:
                    rows.append(dict(method="transigen_full_ft", lr=lr, updates=u, delta_rmse=rm(preds[u], ep) if u in preds else np.inf, **b))
        print(ep.name, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    grid, _ = family_mean(df[df.method == "transigen_full_ft"], by=("K", "lr", "updates"))
    ref, _ = family_mean(df[df.method != "transigen_full_ft"], by=("K", "method"))
    grid.to_csv(public("manifests", "replicate_calibrated_r8", "tsg_ft_grid.tsv"), sep="\t", index=False, float_format="%.6g")
    sel = {}
    for K, g in grid.groupby("K"):
        b = g.sort_values(["delta_rmse", "lr", "updates"]).iloc[0]
        sel[str(int(K))] = dict(lr=float(b.lr), bn=BN, updates=int(b.updates), source_delta_rmse=float(b.delta_rmse),
                                as_is=float(ref[(ref.K == K) & (ref.method == "transigen_as_is")].delta_rmse.iloc[0]),
                                offset=float(ref[(ref.K == K) & (ref.method == "transigen_offset")].delta_rmse.iloc[0]))
    dump(dict(checkpoints=cks, ft=sel, rule="frozen BatchNorm statistics; per K the lowest confirmation-family query delta RMSE over lr x updates (seed 42, support seed 1)",
              run_id=a.run_id, n_ft_runs_with_divergence=int(n_div), commit=os.environ.get("R8_COMMIT", "")), public("manifests", "replicate_calibrated_r8", "tsg_selection.json"))
    dump(run_record(a.run_id, "tsg_select", t0, selected=sel, n_ft_runs_with_divergence=int(n_div)), os.path.join(out, "result.json"))
    print(grid.to_string())
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
