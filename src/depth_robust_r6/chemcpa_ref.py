"""Public reference: the original chemCPA (pinned 43e830eb), the earlier study's frozen P0 checkpoints.

Nothing is trained here. The checkpoints were fitted by the earlier study on the 209 TRAIN drug
groups (settings chosen on the 70 VALIDATION groups) at full observation, without any count
thinning. This adapter feeds them R6's own eval_input control cells -- the same cells and the same
stored thinned views every other model sees -- and scores the result against the R6 truth:

    delta_hat = sum_plate w * ( mean_cells chemCPA(control cell, drug, 0.5 uM, cell line) - reference mean )

The earlier AnnData is opened only because chemCPA's own loader builds the drug index, the drug
embedding table and the cell-line one-hot from it; no expression row of that file is given to the
model here. For conditions on several plates the earlier study's plate weighting is kept (each
plate's own eval_input cells), which favours the reference if anything.

`--setting dev` is a functional check only: those drugs were in chemCPA's training data.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import data as D
from .count_view import lognorm
from .evaluate import cond_rmse, drug_macro


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=["dev", "final"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--manifest-dir", default="manifests/chemcpa_moe_v2")
    a = ap.parse_args()
    if a.setting == "final":
        assert Path("manifests/depth_robust_r6/freeze_manifest.json").exists(), "freeze before the held-out evaluation"
    run = Path(os.environ["R6_ASSETS"]) / "runs" / a.setting / a.run_id
    run.mkdir(parents=True, exist_ok=False)
    prior = Path(os.environ["PRIOR_V2"])
    from src.chemcpa_moe_v2.train_cpa import _import_chemcpa, build, make_module
    args = SimpleNamespace(adata=str(prior / "v2/tahoe_cells.h5ad"), embedding=str(prior / "v2/tahoe_rdkit2D_embedding.parquet"),
                           hparams="configs/chemcpa_moe_v2/hparams_official.json", patience=3, seed=42,
                           n_experts=4, expert_rank=32)
    ChemCPA, load_dataset_splits, PDM = _import_chemcpa(str(prior / "external/chemCPA"))
    cfg, datasets, dataset, dm, dcfg = build(args, ChemCPA, load_dataset_splits, PDM)
    drug_to_idx = {x: i for i, x in enumerate(dataset.drugs_names_unique_sorted)}
    cov_onehot = dataset.atomic_covars_dict["cell_type"]
    import anndata as ad
    obs = ad.read_h5ad(args.adata, backed="r").obs[["cell_line_id", "cell_type"]].astype(str).drop_duplicates()
    name_of = dict(zip(obs.cell_line_id, obs.cell_type))
    del datasets, dm

    d = D.load()
    M = Path(a.manifest_dir)
    aliases = pd.read_csv(M / "drug_aliases.tsv", sep="\t")
    label_of = {str(r.group): sorted(str(x) for x in ast.literal_eval(r.drugs))[0].replace("+", "-").replace("_", "-").strip()
                for _, r in aliases.iterrows()}
    didx = np.array([drug_to_idx.get(label_of[g], -1) for g in d.splits.group_smiles.astype(str)])
    _, eval_drugs = d.split_spec(a.setting)
    L, Dg = d.conditions(eval_drugs)
    assert (didx[Dg] >= 0).all(), "an evaluation drug has no chemCPA index"
    cov = {l: torch.as_tensor(np.asarray(cov_onehot[name_of[cid]], dtype=np.float32)) for l, cid in enumerate(d.lines.cell_line_id.astype(str))
           if cid in name_of and name_of[cid] in cov_onehot}
    pw = np.load(Path(os.environ["R6_V2DATA"]) / "plate_weights.npy")
    ref = np.load(d.root / "ref_mean.npy")
    frozen = json.loads((M / "freeze_fixed.json").read_text())["checkpoint_sha256"]
    dev = "cuda"
    views = [(1.0, 0)] + [(p, s) for p in D.PS[1:] for s in D.THIN_SEEDS]
    ev_rows = np.concatenate([np.arange(r.eval_start, r.eval_end) for r in d.pools.itertuples()])
    n_dropped_plate = 0
    rec = dict(run_id=a.run_id, setting=a.setting, model="chemCPA 43e830eb, earlier study's P0_lr0.3 (frozen, not retrained)",
               n_conditions=int(len(L)), checkpoints={}, scores={})
    for sd in a.seeds:
        ck = prior / f"v2/runs_fix_e30/P0_lr0.3_s{sd}/best.pt"
        sha = D.sha256_file(ck)
        assert sha == frozen[f"P0_lr0.3_s{sd}"], "checkpoint differs from the earlier study's freeze"
        rec["checkpoints"][str(sd)] = dict(path=str(ck), sha256=sha)
        lm, _, _ = make_module(ChemCPA, cfg, dcfg, "P0", args)
        lm.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
        model = lm.to(dev).model.eval()
        out = {}
        for (p, s) in views:
            c, r = (d.cnt[ev_rows], d.crest[ev_rows]) if p == 1.0 else d.eval_views[(p, s)]
            x = lognorm(c.astype(np.float64), (c.sum(1) + r).astype(np.float64)).astype(np.float32)
            x = torch.as_tensor(x, device=dev).reshape(len(d.pools), D.EVAL_N, -1)
            P = np.zeros((len(L), d.Y.shape[2]), np.float32)
            with torch.no_grad():
                for i, (l, g) in enumerate(zip(L, Dg)):
                    mix = [(q, w) for q, w in enumerate(pw[l, g]) if w > 0 and d.pool_id[q, l] >= 0]
                    if sd == a.seeds[0] and (p, s) == views[0]:
                        n_dropped_plate += int(sum(1 for q, w in enumerate(pw[l, g]) if w > 0) - len(mix))
                    tot = sum(w for _, w in mix)
                    for q, w in mix:
                        g_in = x[d.pool_id[q, l]]
                        n = g_in.shape[0]
                        o, _ = model.predict(genes=g_in, drugs_idx=torch.full((n,), int(didx[g]), dtype=torch.long, device=dev),
                                             dosages=torch.full((n,), 0.5, dtype=torch.float32, device=dev),
                                             covariates=[cov[l].to(dev).unsqueeze(0).expand(n, -1).contiguous()])
                        dim = o.shape[1] // 2
                        P[i] += (w / tot) * (o[:, :dim].mean(0).float().cpu().numpy() - ref[q, l])
            out[(p, s)] = P
        np.savez_compressed(run / f"pred_{a.setting}_s{sd}.npz", line=L, drug=Dg,
                            **{f"p{p}_s{s}": v.astype(np.float16) for (p, s), v in out.items()})
        T = d.Y[L, Dg]
        rec["scores"][str(sd)] = {f"p{p}_s{s}": drug_macro(cond_rmse(v, T), Dg) for (p, s), v in out.items()}
        print(sd, json.dumps(rec["scores"][str(sd)]), flush=True)
        del lm, model
    rec["n_plate_terms_dropped_no_eval_cells"] = n_dropped_plate
    rec["commit"] = os.popen("git rev-parse HEAD").read().strip()
    D.dump_json(rec, run / "record.json")


if __name__ == "__main__":
    main()
