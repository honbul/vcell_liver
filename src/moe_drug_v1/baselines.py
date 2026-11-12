"""The three minimum baselines (plan s8.4). If A/C cannot beat these, the structure comparison does
not become a practical result, whichever of A or C wins.

no_change : Delta = 0
line_mean : the mean train Delta of that cell line, reused for every drug
ridge     : a linear map from the same inputs the networks get, concat(basal_z, morgan), with the
            regularisation strength chosen on validation only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import evaluate as ev
from .data import Dataset


def fit_ridge(X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor, lam: float) -> torch.Tensor:
    """Weighted ridge with an unpenalised intercept, solved in closed form on the device."""
    n, d = X.shape
    Xb = torch.cat([X, torch.ones(n, 1, dtype=X.dtype, device=X.device)], 1)
    sw = w / w.sum()
    Xw = Xb * sw[:, None]
    A = Xb.T @ Xw
    B = Xw.T @ Y
    reg = torch.eye(d + 1, dtype=X.dtype, device=X.device) * lam
    reg[d, d] = 0.0
    return torch.linalg.solve(A + reg, B)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lams", default="1e-4,1e-3,1e-2,1e-1,1,10,100")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Dataset(args.data_dir, args.manifest_dir, device=dev)
    tr = ds.conditions("train")
    ev_c = ds.conditions(args.split)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ty, tl = tr.y, tr.line
    ey, el = ev_c.y.cpu().numpy(), ev_c.line.cpu().numpy()
    res = {}

    # no change
    pred0 = np.zeros_like(ey)
    res["no_change"], _ = ev.all_metrics(pred0, ey, el, ds.line_names, ds.snu423)
    np.save(out / f"pred_no_change_{args.split}.npy", pred0)

    # line mean of train responses
    means = torch.zeros(ds.n_lines, ds.n_genes, device=dev)
    for l in range(ds.n_lines):
        m = tl == l
        if int(m.sum()) > 0:
            means[l] = ty[m].mean(0)
    predm = means[ev_c.line].cpu().numpy()
    res["line_mean"], _ = ev.all_metrics(predm, ey, el, ds.line_names, ds.snu423)
    np.save(out / f"pred_line_mean_{args.split}.npy", predm)

    # ridge on the same inputs, lambda picked on validation only
    Xtr = torch.cat([ds.basal[tr.line], ds.fp[tr.drug]], 1).double()
    Xev = torch.cat([ds.basal[ev_c.line], ds.fp[ev_c.drug]], 1).double()
    va = ds.conditions("val")
    Xva = torch.cat([ds.basal[va.line], ds.fp[va.drug]], 1).double()
    vay, val_ = va.y.cpu().numpy(), va.line.cpu().numpy()
    scan = []
    best = None
    for lam in [float(x) for x in args.lams.split(",")]:
        W = fit_ridge(Xtr, ty.double(), tr.weight.double(), lam)
        p = (torch.cat([Xva, torch.ones(len(Xva), 1, dtype=torch.float64, device=dev)], 1) @ W)
        s = ev.macro_delta_rmse(p.float().cpu().numpy(), vay, val_)
        scan.append(dict(lam=lam, val_macro_delta_rmse=s))
        if best is None or s < best[1]:
            best = (lam, s, W)
    lam, _, W = best
    predr = (torch.cat([Xev, torch.ones(len(Xev), 1, dtype=torch.float64, device=dev)], 1) @ W)
    predr = predr.float().cpu().numpy()
    res["ridge"], _ = ev.all_metrics(predr, ey, el, ds.line_names, ds.snu423)
    res["ridge"]["lambda"] = lam
    res["ridge"]["lambda_scan"] = scan
    np.save(out / f"pred_ridge_{args.split}.npy", predr)

    (out / f"baselines_{args.split}.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:10s} macro={v['macro_delta_rmse']:.5f} centred_r={v['drug_centred_r']:.4f} "
              f"snu423={v.get('snu423_delta_rmse', float('nan')):.5f}", flush=True)


if __name__ == "__main__":
    main()
