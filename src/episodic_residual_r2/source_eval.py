"""All compared methods on the fixed meta-validation episodes (3 support seeds, K = 8/16/32, 3 model seeds).

Uses only source-side selections. The same Suite object is later used unchanged for the HCC targets.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from .common import CFG, public, dump, new_run_dir, run_record
from .data import Source
from .evalcore import score, family_mean, paired_family
from .suite import Suite, current_selection
from . import tsg as G
from src.transigen_adaptation_b1.data import Data

VALUES = ["delta_rmse", "centred_rmse", "centred_pearson", "delta_pearson", "retrieval_mrr_centred", "retrieval_mrr", "top20_direction", "magnitude_ratio",
          "delta_rmse_shared_plate", "delta_rmse_disjoint_plate"]


def summarise_source(df, refs=("mean_offset", "source_same_drug_mean", "allfeat_krr")):
    """Model seeds are averaged per episode first; then the family-equal mean; paired family contrasts against the references."""
    ep = df.groupby(["cell", "family", "support_seed", "K", "method"])[VALUES].mean().reset_index()
    parts, contrasts = None, []
    for v in VALUES:
        agg, fam = family_mean(ep, value=v)
        parts = agg if parts is None else parts.merge(agg.drop(columns="n_families"), on=["K", "method"])
        if v in ("delta_rmse", "centred_rmse", "centred_pearson", "delta_rmse_shared_plate", "delta_rmse_disjoint_plate"):
            for ref in refs:
                for m in sorted(set(ep.method) - {ref}):
                    c = paired_family(fam, m, ref, value=v)
                    c["metric"] = v
                    contrasts.append(c)
    return parts, pd.concat(contrasts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tag", default="source_eval", help="prefix of the public output tables")
    ap.add_argument("--extra", default="{}", help="JSON {name: {seed: run_id}} of additional learned runs to include (variants, controls)")
    ap.add_argument("--modes", default='{"residual_adapter": ["zero_residual", "shuffle_residual"]}')
    ap.add_argument("--no-tsg", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    t0 = time.time()
    out = new_run_dir("source_eval", a.run_id)
    dev = torch.device("cuda")
    torch.set_num_threads(1)
    src = Source(dev)
    extra = {}
    for name, seeds in json.loads(a.extra).items():
        extra[name] = {}
        for s, rid in seeds.items():
            r = json.load(open(os.path.join(os.environ["R2_RUNS"], "meta", rid, "result.json")))
            extra[name][s] = dict(checkpoint=r["checkpoint"], checkpoint_sha256=r["checkpoint_sha256"], run_id=rid)
    sel = current_selection(extra, json.loads(a.modes))
    suite = Suite(sel, src, dev, with_tsg=not a.no_tsg)
    B = Data() if not a.no_tsg else None
    rows, failures = [], []
    for ci in src.eval_cells("val"):
        ra = G.source_row_access(B, src, src.cell_name[ci]) if B is not None else None
        for s in cfg["episodes"]["meta_val_support_seeds"]:
            ep = src.eval_episode(ci, s)
            for K in cfg["episodes"]["k_values"]:
                for method, seed, pred in suite.run(ep, K, ra):
                    if isinstance(pred, str):
                        failures.append(dict(cell=ep.name, support_seed=s, K=K, method=method, model_seed=seed, status=pred))
                        continue
                    r, _ = score(ep, K, pred, dict(method=method, model_seed=seed))
                    rows.append(r)
            print(ep.name, s, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(public("manifests", "episodic_residual_r2", f"{a.tag}_episodes.tsv"), sep="\t", index=False, float_format="%.6g")
    summ, contrasts = summarise_source(df)
    summ.to_csv(public("manifests", "episodic_residual_r2", f"{a.tag}_summary.tsv"), sep="\t", index=False, float_format="%.6g")
    contrasts.to_csv(public("manifests", "episodic_residual_r2", f"{a.tag}_contrasts.tsv"), sep="\t", index=False, float_format="%.6g")
    # per model seed (family-equal), to show seed spread
    seedtab, _ = family_mean(df[df.model_seed >= 0], by=("K", "method", "model_seed"))
    seedtab.to_csv(public("manifests", "episodic_residual_r2", f"{a.tag}_by_model_seed.tsv"), sep="\t", index=False, float_format="%.6g")
    res = run_record(a.run_id, "source_eval", t0, n_rows=len(df), failures=failures, learned={k: {s: e.get("run_id", "") for s, e in v.items()} for k, v in sel["learned"].items()},
                     modes=sel["modes"], tsg_ft=sel["tsg_ft"] if not a.no_tsg else None, lambdas=sel["lambda"])
    dump(res, os.path.join(out, "result.json"))
    pd.set_option("display.width", 250, "display.max_rows", 400)
    print(summ[["K", "method", "delta_rmse", "centred_rmse", "centred_pearson", "retrieval_mrr_centred", "delta_rmse_shared_plate", "delta_rmse_disjoint_plate"]].to_string())
    print(contrasts[(contrasts.metric == "delta_rmse")].to_string())


if __name__ == "__main__":
    main()
