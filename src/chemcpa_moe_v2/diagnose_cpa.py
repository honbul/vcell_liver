"""Why the chemCPA arms score where they do — read off the frozen checkpoints, on validation only.

The primary metric subtracts the **observed** plate DMSO pool from the predicted treated mean, so
its error is exactly `predicted treated profile - observed treated profile`: a model that cannot
reproduce a cell line's absolute expression is penalised for that even if it had the response
right. This script separates the two:

```
observed mode   err = pred_treated      - obs_treated                      (the primary)
model mode      err = (pred_treated - pred_dmso) - (obs_treated - obs_dmso)
```

The model-mode error cancels any constant offset the decoder makes for that (cell line, plate), so
the gap between the two numbers is the share of the primary score that is reconstruction and not
response. It also reports how large the predicted response is at all: an arm whose `pred_treated -
pred_dmso` has RMS near zero is predicting "no change" however good its reconstruction is.

Nothing here selects anything, and nothing here touches the test split. It runs after selection,
on checkpoints that are already frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .train_cpa import ConditionEvaluator, _import_chemcpa, build, macro_delta_rmse, make_module


def chemcpa_own_curve(run: Path) -> dict:
    """chemCPA's own logged objective, so 'it did not train' can be ruled in or out."""
    files = sorted(run.glob("csv/version_*/metrics.csv"))
    if not files:
        return {}
    df = pd.read_csv(files[0])
    out = {}
    for col in ["reconstruction_loss", "adversary_drugs_loss", "adversary_covariates_loss"]:
        if col not in df.columns:
            continue
        s = df[["epoch", col]].dropna()
        if s.empty:
            continue
        first = s.groupby("epoch")[col].mean()
        out[col] = dict(epoch0=round(float(first.iloc[0]), 6),
                        last=round(float(first.iloc[-1]), 6),
                        by_epoch=[round(float(v), 6) for v in first.tolist()])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chemcpa-src", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--hparams", required=True)
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--expert-rank", type=int, default=32)
    ap.add_argument("--max-control-cells", type=int, default=64)
    args = ap.parse_args()

    import anndata as ad
    ChemCPA, load_dataset_splits, PerturbationDataModule = _import_chemcpa(args.chemcpa_src)
    sel = json.loads(Path(args.selection).read_text())
    cfg, datasets, dataset, dm, dcfg = build(args, ChemCPA, load_dataset_splits,
                                             PerturbationDataModule)
    adata = ad.read_h5ad(args.adata)
    drug_to_idx = {d: i for i, d in enumerate(dataset.drugs_names_unique_sorted)}
    ev = ConditionEvaluator(adata, Path(args.v2_dir), Path(args.manifest_dir), "val",
                            drug_to_idx, max_control_cells=args.max_control_cells, seed=0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rep: dict = {"n_val_conditions": len(ev.rows), "control_modes": {}}
    for arm, s in sel["aggregates"].items():
        lrs = s["lr_scale"]
        per_seed = {}
        for sd in args.seeds:
            run = Path(args.runs_dir) / f"{arm}_lr{lrs}_s{sd}"
            ck = run / "best.pt"
            if not ck.exists():
                continue
            lm, _, _ = make_module(ChemCPA, cfg, dcfg, arm, args)
            lm.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
            lm = lm.to(dev)
            obs, P_obs, T, L = ev.evaluate(lm.model, device=dev, control_mode="observed")
            mod, P_mod, _, _ = ev.evaluate(lm.model, device=dev, control_mode="model")
            per_seed[str(sd)] = dict(
                observed=round(obs, 6),
                model=round(mod, 6),
                reconstruction_share_pct=round((obs - mod) / obs * 100, 2) if obs else None,
                rms_predicted_response_observed=round(float(np.sqrt((P_obs ** 2).mean())), 6),
                rms_predicted_response_model=round(float(np.sqrt((P_mod ** 2).mean())), 6),
                rms_truth=round(float(np.sqrt((T ** 2).mean())), 6),
                macro_delta_rmse_of_zero=round(macro_delta_rmse(np.zeros_like(T), T, L), 6),
                chemcpa_objective=chemcpa_own_curve(run))
            del lm
            torch.cuda.empty_cache()
        if per_seed:
            rep["control_modes"][arm] = dict(lr_scale=lrs, per_seed=per_seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
