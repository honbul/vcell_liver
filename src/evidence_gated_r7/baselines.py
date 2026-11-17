"""Closed-form baselines (R4 recipe, imported unchanged) re-scored on the R7 store; alpha chosen on validation (source panel)."""
import json
import numpy as np
import pandas as pd

from .common import pman, dump
from .data import Store, TrainStats
from .evaluate import unit_metrics, summarise
from src.response_geometry_r4.baselines import fit_all, ALPHAS


def main():
    S = Store(); st = TrainStats(S)
    models = fit_all(S, st)
    val = S.val[S.source(S.val)]
    yv = S.delta(val); uv = S.units.iloc[val].reset_index(drop=True)
    rows = []
    for name, f in models.items():
        p = f(val)
        s = summarise(unit_metrics(p, yv, uv.cell.to_numpy()), uv, "source")
        rows.append(dict(model=name, val_rmse=s["rmse"], val_pearson=s["pearson"], val_centred_pearson=s["centred_pearson"], val_mrr=s["mrr"],
                         val_top20_direction=s["top20_direction"], val_rms_pred=s["rms_pred"]))
        print(rows[-1], flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(pman("baselines_validation.tsv"), sep="\t", index=False)
    sel = {}
    for fam in ["ecfp_x0_ridge", "cellmean_ecfp_ridge"]:
        t = tab[tab.model.str.startswith(fam + "_a")].sort_values(["val_rmse", "model"])
        sel[fam] = dict(model=t.model.iloc[0], alpha=int(t.model.iloc[0].split("_a")[-1]), val_rmse=float(t.val_rmse.iloc[0]))
    dump(dict(alpha_grid=ALPHAS, selection_rule="lowest validation source-panel delta RMSE", selected=sel), pman("baseline_selection.json"))
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
