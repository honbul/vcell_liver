"""All compared methods on the fixed confirmation-family episodes (3 support seeds, K = 8/16/32, 3 model seeds).

Only source-side selections are used; the same Suite object is later used unchanged for the HCC targets.
Writes <tag>_episodes.tsv, <tag>_summary.tsv, <tag>_contrasts.tsv, <tag>_by_model_seed.tsv and global_state_ablation.tsv.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from .common import CFG, public, dump, new_run_dir, run_record
from .data import Store
from .suite import Suite, current_selection
from .contrasts import family_contrasts, ADAPTERS
from src.episodic_residual_r2.evalcore import score, family_mean
from src.episodic_residual_r2 import tsg as G
from src.transigen_adaptation_b1.data import Data

VALUES = ["delta_rmse", "centred_rmse", "centred_pearson", "delta_pearson", "retrieval_mrr_centred", "retrieval_mrr", "top20_direction", "magnitude_ratio",
          "delta_rmse_shared_plate", "delta_rmse_disjoint_plate"]
CONTRAST_VALUES = ["delta_rmse", "centred_rmse", "centred_pearson", "delta_rmse_shared_plate", "delta_rmse_disjoint_plate"]
PUB = lambda n: public("manifests", "replicate_calibrated_r8", n)


def summarise(df):
    """Model seeds are averaged per episode first; then the family-equal mean; paired family contrasts."""
    ep = df.groupby(["cell", "family", "support_seed", "K", "method"])[VALUES].mean().reset_index()
    parts, cons = None, []
    for v in VALUES:
        agg, fam = family_mean(ep, value=v)
        parts = agg if parts is None else parts.merge(agg.drop(columns="n_families"), on=["K", "method"])
        if v in CONTRAST_VALUES:
            cons.append(family_contrasts(fam, v))
    return parts, pd.concat(cons, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tag", default="source_eval")
    ap.add_argument("--no-tsg", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    t0 = time.time()
    out = new_run_dir("source_eval", a.run_id)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = Store(dev)
    sel = current_selection(with_tsg=not a.no_tsg)
    suite = Suite(sel, st, dev, with_tsg=not a.no_tsg, shuffle_modes=ADAPTERS)
    B = Data(os.environ["TB_PROC"]) if not a.no_tsg else None
    rows, failures = [], []
    for ci in st.eval_cells("val"):
        ra = G.source_row_access(B, st, st.cell_name[ci]) if B is not None else None
        for s in cfg["episodes"]["confirmation_support_seeds"]:
            ep = st.eval_episode(ci, s)
            for K in cfg["episodes"]["k_values"]:
                for method, seed, pred in suite.run(ep, K, ra):
                    if isinstance(pred, str):
                        failures.append(dict(cell=ep.name, support_seed=s, K=K, method=method, model_seed=seed, status=pred))
                        continue
                    r, _ = score(ep, K, pred, dict(method=method, model_seed=seed))
                    rows.append(r)
            print(ep.name, s, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(PUB(f"{a.tag}_episodes.tsv.gz"), sep="\t", index=False, float_format="%.6g")
    summ, cons = summarise(df)
    summ.to_csv(PUB(f"{a.tag}_summary.tsv"), sep="\t", index=False, float_format="%.6g")
    cons.to_csv(PUB(f"{a.tag}_contrasts.tsv"), sep="\t", index=False, float_format="%.6g")
    seedtab, _ = family_mean(df[df.model_seed >= 0], by=("K", "method", "model_seed"))
    seedtab.to_csv(PUB(f"{a.tag}_by_model_seed.tsv"), sep="\t", index=False, float_format="%.6g")
    if a.tag == "source_eval":
        g = cons[cons.contrast.isin(["B01-B00", "G_concat-B00", "G_dummy-B00", "B01-G_dummy", "B01-G_concat", "B11-B10"]) & cons.metric.isin(["delta_rmse", "centred_rmse"])].copy()
        g.insert(0, "set", "confirmation_families")
        g.to_csv(PUB("global_state_ablation.tsv"), sep="\t", index=False, float_format="%.6g")
    res = run_record(a.run_id, "source_eval", t0, n_rows=len(df), failures=failures, learned={k: {s: e.get("run_id", "") for s, e in v.items()} for k, v in sel["learned"].items()},
                     tsg_ft=sel["tsg"]["ft"] if not a.no_tsg else None)
    dump(res, os.path.join(out, "result.json"))
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(summ[["K", "method", "delta_rmse", "centred_rmse", "centred_pearson", "retrieval_mrr_centred", "n_families"]].to_string())
    print(cons[(cons.metric == "delta_rmse")].to_string())


if __name__ == "__main__":
    main()
