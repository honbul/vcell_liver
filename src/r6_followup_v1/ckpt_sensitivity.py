"""Fixed-checkpoint input-use diagnosis (plan 6.1, before any retraining). Nothing is fitted.

For a finished additive-head checkpoint  delta = g(z) + h(z, drug)  this measures, at p = 1 and
p = 0.25, on the evaluation drugs of the checkpoint's own setting:
    h = 0            prediction with the drug-specific part removed: total RMSE and L_within
                     (the centred prediction is then 0, so L_within is the RMS of the centred truth)
    drug swap        same control set, drug input exchanged with another evaluated drug of the same
                     pool: change of the prediction = h_d - h_e (g cancels)
    sizes            RMS(g), RMS(h), RMS(h_d - h_e) against RMS of the measured contrast (y_d - y_e)
This is a sensitivity of jointly trained models. Removing h from a jointly trained model is NOT the
performance of a model trained without the drug input (that is arm S of the retraining comparison).

Checkpoints: original R6 runs (read-only) and, later, the follow-up's own source runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.depth_robust_r6.train import build_model as build_orig

from . import common as C
from . import metrics as M
from .train_fixed import build_model as build_fu
from .train_fixed import predict

VIEWS = {1.0: [(1.0, 0)], 0.25: [(0.25, s) for s in (0, 1, 2)]}


def pair_rms(X: np.ndarray, groups) -> np.ndarray:
    """Per condition: sqrt of the mean over the other drugs e of its pool of mean_g (x_d - x_e)^2."""
    out = np.full(len(X), np.nan)
    for idx in groups:
        m = len(idx)
        if m < 2:
            continue
        x = X[idx].astype(np.float64)
        K = x @ x.T / X.shape[1]
        dg = np.diag(K)
        d2 = dg[:, None] + dg[None, :] - 2 * K
        out[idx] = np.sqrt(np.clip(d2.sum(1) / (m - 1), 0, None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/r6_followup_v1/ckpt_sensitivity.json")
    ap.add_argument("--out", default="reports/r6_followup_v1/checkpoint_branch_sensitivity.tsv")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    dev = "cuda"
    d = C.load()
    rows = []
    for item in spec["sets"]:
        role, name = item["split_role"], item["name"]
        per_seed = []
        for rel in item["ckpts"]:
            root = C.orig_assets() if item["origin"] == "original_r6" else C.assets()
            ck = torch.load(root / "runs" / rel / "ckpt.pt", map_location="cpu", weights_only=False)
            assert ck["hp"].get("head_mode") == "additive"
            if item["origin"] == "original_r6":
                model = build_orig(d, ck["arm"], ck["hp"]).to(dev)
                held = C.test_drugs(d) if ck["setting"] == "final" else C.source_split(d, 2)[1]
                fold = -1 if ck["setting"] == "final" else 2
            else:
                model = build_fu(d, ck["arm"], ck["hp"]).to(dev)
                held = C.source_split(d, ck["fold"], ck["inner"])[1]; fold = ck["fold"]
            assert not (set(ck["fit_drugs"]) & set(held))
            model.load_state_dict(ck["state_dict"])
            L, Dg, parts = predict(model, d, held, dev, ck["y_scale"], views=[v for vs in VIEWS.values() for v in vs], parts=True)
            T = d.Y[L, Dg].astype(np.float64)
            pool = C.pool_of(d, L, Dg)
            groups = M.groups_of(np.zeros(len(L), int), pool)
            Tc = M.centre_by(T, groups)
            inpair = np.zeros(len(L), bool)
            for idx in groups:
                inpair[idx] = len(idx) >= 2
            t_pair = pair_rms(T, groups)
            for p, vs in VIEWS.items():
                acc = {}
                for v in vs:
                    g, h = parts[v]
                    full = g + h
                    lw_full = M.within_point(M.grams(full - T, groups), groups, len(L))
                    lw_g = M.within_point(M.grams(g - T, groups), groups, len(L))
                    hc = M.centre_by(h, groups)
                    num = (hc * Tc).sum(1); den = np.sqrt((hc ** 2).sum(1) * (Tc ** 2).sum(1))
                    vals = dict(total_rmse_full=M.cond_rmse(full, T), total_rmse_h0=M.cond_rmse(g, T),
                                within_rmse_full=lw_full, within_rmse_h0=lw_g,
                                rms_g=np.sqrt((g.astype(np.float64) ** 2).mean(1)), rms_h=np.sqrt((h.astype(np.float64) ** 2).mean(1)),
                                rms_h_centred=np.where(inpair, np.sqrt((hc ** 2).mean(1)), np.nan),
                                rms_drug_swap_change=pair_rms(h, groups), rms_truth=np.sqrt((T ** 2).mean(1)),
                                rms_truth_centred=np.where(inpair, np.sqrt((Tc ** 2).mean(1)), np.nan), rms_truth_pair_contrast=t_pair,
                                centred_h_truth_pearson=np.where(inpair & (den > 1e-12), num / np.maximum(den, 1e-12), np.nan))
                    for k, x in vals.items():
                        acc[k] = acc.get(k, 0) + x / len(vs)
                row = {k: M.drug_macro(x, Dg) for k, x in acc.items()}
                row.update(p=p, seed=ck["seed"], arm=ck["arm"], n_drugs=int(len(np.unique(Dg))), n_units=int(len(L)), fold=fold)
                per_seed.append(row)
            print(name, rel, json.dumps({k: round(float(x), 5) for k, x in per_seed[-1].items() if isinstance(x, float)}), flush=True)
        df = pd.DataFrame(per_seed)
        for p, gdf in df.groupby("p"):
            num = gdf.drop(columns=["p", "seed", "arm", "n_drugs", "n_units", "fold"]).mean()
            r = dict(C.std_fields(split_role=role), checkpoint_set=name, origin=item["origin"], arm=gdf.arm.iloc[0], fold=int(gdf.fold.iloc[0]),
                     model_seed=",".join(str(s) for s in gdf.seed), p=p, n_drugs=int(gdf.n_drugs.iloc[0]), n_units=int(gdf.n_units.iloc[0]), **num.to_dict())
            r["h0_total_diff"] = r["total_rmse_h0"] - r["total_rmse_full"]            # h=0 minus full (positive: h helps)
            r["h0_total_rel"] = r["h0_total_diff"] / r["total_rmse_full"]
            r["h0_within_diff"] = r["within_rmse_h0"] - r["within_rmse_full"]
            r["h0_within_rel"] = r["h0_within_diff"] / r["within_rmse_full"]
            r["swap_change_over_truth_contrast"] = r["rms_drug_swap_change"] / r["rms_truth_pair_contrast"]
            r["rms_h_over_rms_g"] = r["rms_h"] / r["rms_g"]
            rows.append(r)
    out = pd.DataFrame(rows)
    out["metric_definition"] = ("drug-macro means of per-condition values, averaged over thinning seeds then training seeds; h0 = prediction with the "
                                "drug-specific part set to 0 in a jointly trained model (a sensitivity, not a state-only model); "
                                "drug swap = same control set, drug exchanged within the pool")
    out.to_csv(a.out, sep="\t", index=False, float_format="%.6f")
    print(out.drop(columns=["metric_definition"]).T.to_string())


if __name__ == "__main__":
    main()
