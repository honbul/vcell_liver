"""R3-1a: the one mean predictor m_psi shared by every mean-anchored method.

Target: the prior study's condition truth Y[line, drug] = mean_cells(log1p CP10k) of the treated
well(s) minus the same plate's DMSO pool (full wells, TRAIN drugs only). The predicted treated
mean of a well is then  mu_hat = dmso_mean[plate, line] + m_psi(line, drug).

Candidates (all see: cell line identity / basal profile, and the drug's ECFP4 bits):
    nochange        0
    linemean        per-line mean response over the fitting drugs
    krr             linemean + Tanimoto-kernel ridge on the drug, one effect shared by all lines
    krr_line        linemean + multi-task kernel ridge: k_drug * (a + (1 - a) * [same line])
    mlp             linemean + small MLP(basal profile, ECFP4) -> 2,000 genes

`--mode cv` scores them by 3-fold cross-validation over the dev folds of the TRAIN drugs and writes
the choice; `--mode fit --setting dev|final` fits the chosen one and writes delta_hat[line, drug].
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .setting import DEV_EVAL_FOLD, run_dir, split_spec

LAMBDAS = [0.03, 0.1, 0.3, 1.0, 3.0]


def tanimoto(A, B):
    inter = A @ B.T
    return inter / (A.sum(1, keepdim=True) + B.sum(1)[None] - inter).clamp(min=1e-9)


def conditions(d, drugs, lines=None):
    L, G = np.nonzero(d.valid)
    keep = np.isin(G, drugs)
    if lines is not None:
        keep &= np.isin(L, lines)
    return L[keep], G[keep]


def line_mean(d, Lf, Gf, dev):
    Y = torch.as_tensor(d.Y[Lf, Gf], dtype=torch.float64, device=dev)
    nL = d.Y.shape[0]
    lm = torch.zeros(nL, Y.shape[1], dtype=torch.float64, device=dev)
    cnt = torch.zeros(nL, dtype=torch.float64, device=dev)
    li = torch.as_tensor(Lf, device=dev)
    lm.index_add_(0, li, Y); cnt.index_add_(0, li, torch.ones_like(li, dtype=torch.float64))
    return lm / cnt.clamp(min=1)[:, None]


def fit_predict(d, kind, hp, fit_drugs, pred_drugs, dev, seed=0):
    """Returns delta_hat [L, len(pred_drugs), G] (float32, cpu)."""
    Lf, Gf = conditions(d, fit_drugs)
    nL, G = d.Y.shape[0], d.Y.shape[2]
    pred_drugs = np.asarray(pred_drugs)
    if kind == "nochange":
        return torch.zeros(nL, len(pred_drugs), G)
    lm = line_mean(d, Lf, Gf, dev)
    base = lm[:, None, :].expand(nL, len(pred_drugs), G)
    if kind == "linemean":
        return base.float().cpu()
    Yr = torch.as_tensor(d.Y[Lf, Gf], dtype=torch.float64, device=dev) - lm[torch.as_tensor(Lf, device=dev)]
    fp = torch.as_tensor(d.fp, dtype=torch.float64, device=dev)
    if kind == "krr":
        fd = np.asarray(sorted(set(Gf.tolist())))
        pos = {g: i for i, g in enumerate(fd)}
        gi = torch.as_tensor([pos[g] for g in Gf], device=dev)
        R = torch.zeros(len(fd), G, dtype=torch.float64, device=dev); c = torch.zeros(len(fd), dtype=torch.float64, device=dev)
        R.index_add_(0, gi, Yr); c.index_add_(0, gi, torch.ones(len(gi), dtype=torch.float64, device=dev))
        R = R / c[:, None]                                   # drug-level mean residual over lines
        K = tanimoto(fp[fd], fp[fd])
        alpha = torch.linalg.solve(K + hp["lam"] * torch.eye(len(fd), dtype=torch.float64, device=dev), R)
        eff = tanimoto(fp[pred_drugs], fp[fd]) @ alpha       # [P, G]
        return (base + eff[None]).float().cpu()
    if kind == "krr_line":
        a = hp["a"]
        Lt, Gt = torch.as_tensor(Lf, device=dev), torch.as_tensor(Gf, device=dev)
        Kd = tanimoto(fp, fp)
        K = Kd[Gt][:, Gt] * (a + (1 - a) * (Lt[:, None] == Lt[None, :]).double())
        alpha = torch.linalg.solve(K + hp["lam"] * torch.eye(len(Lf), dtype=torch.float64, device=dev), Yr)
        out = torch.empty(nL, len(pred_drugs), G, dtype=torch.float64, device=dev)
        Kp = Kd[torch.as_tensor(pred_drugs, device=dev)][:, Gt]          # [P, n]
        for l in range(nL):
            out[l] = (Kp * (a + (1 - a) * (Lt == l).double())[None]) @ alpha
        return (base + out).float().cpu()
    if kind == "mlp":
        torch.manual_seed(seed)
        basal = torch.as_tensor(d.dmso_mean.mean(0), dtype=torch.float32, device=dev)   # [L, G] line profile
        bz = (basal - basal.mean(0)) / (basal.std(0) + 1e-3)
        f32 = fp.float()
        sdy = Yr.std().float()
        net = torch.nn.Sequential(torch.nn.Linear(G + f32.shape[1], hp["width"]), torch.nn.LayerNorm(hp["width"]),
                                  torch.nn.SiLU(), torch.nn.Dropout(hp["drop"]),
                                  torch.nn.Linear(hp["width"], hp["width"]), torch.nn.SiLU(),
                                  torch.nn.Linear(hp["width"], G)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        Lt, Gt = torch.as_tensor(Lf, device=dev), torch.as_tensor(Gf, device=dev)
        Yt = (Yr.float() / sdy)
        n = len(Lf); gen = torch.Generator(device="cpu").manual_seed(seed)
        for ep in range(hp["epochs"]):
            perm = torch.randperm(n, generator=gen).to(dev)
            for i in range(0, n, 256):
                b = perm[i:i + 256]
                loss = ((net(torch.cat([bz[Lt[b]], f32[Gt[b]]], 1)) - Yt[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        out = torch.empty(nL, len(pred_drugs), G, device=dev)
        with torch.no_grad():
            for l in range(nL):
                out[l] = net(torch.cat([bz[l][None].expand(len(pred_drugs), -1), f32[pred_drugs]], 1)) * sdy
        return (base.float() + out).cpu()
    raise ValueError(kind)


CANDIDATES = ([("nochange", {}), ("linemean", {})]
              + [("krr", dict(lam=l)) for l in LAMBDAS]
              + [("krr_line", dict(lam=l, a=0.5)) for l in LAMBDAS]
              + [("mlp", dict(width=256, drop=0.2, lr=1e-3, wd=1e-4, epochs=e)) for e in (20, 60)])


def rmse_cond(d, pred, drugs):
    """Mean over valid conditions of the per-condition RMSE over genes, and the drug-equal mean."""
    drugs = np.asarray(drugs)
    Yt = torch.as_tensor(d.Y[:, drugs]); v = torch.as_tensor(d.valid[:, drugs])
    r = ((pred - Yt) ** 2).mean(2).sqrt()
    per_drug = (r * v).sum(0) / v.sum(0).clamp(min=1)
    return float(r[v].mean()), float(per_drug[v.any(0)].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cv", "fit"], required=True)
    ap.add_argument("--setting", default="dev")
    ap.add_argument("--choice", default="configs/anchored_distribution_r3/mean_model.json")
    ap.add_argument("--table", default="reports/anchored_distribution_r3/tables/mean_model_cv.tsv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = D.load()
    train_drugs = np.asarray([int(g) for g in d.splits.group_index[d.splits.split == "train"]])
    t0 = time.time()
    if a.mode == "cv":
        rows = []
        for kind, hp in CANDIDATES:
            sc = []
            for f in range(D.N_FOLDS):
                fit = train_drugs[d.fold[train_drugs] != f]; ev = train_drugs[d.fold[train_drugs] == f]
                p = fit_predict(d, kind, hp, fit, ev, dev)
                sc.append(rmse_cond(d, p, ev))
            row = dict(kind=kind, hp=json.dumps(hp), rmse_cond=float(np.mean([s[0] for s in sc])),
                       rmse_drug=float(np.mean([s[1] for s in sc])),
                       rmse_cond_folds=json.dumps([round(s[0], 6) for s in sc]),
                       rmse_devfold=sc[DEV_EVAL_FOLD][0], wall_s=time.time() - t0)
            rows.append(row); print(row, flush=True)
        import pandas as pd
        T = pd.DataFrame(rows)
        Path(a.table).parent.mkdir(parents=True, exist_ok=True)
        T.to_csv(a.table, sep="\t", index=False, float_format="%.6g")
        cand = T[~T.kind.isin(["nochange", "linemean"])].sort_values("rmse_cond")
        best = cand.iloc[0]
        ch = dict(kind=best.kind, hp=json.loads(best.hp), cv_rmse_cond=float(best.rmse_cond),
                  cv_rmse_linemean=float(T[T.kind == "linemean"].rmse_cond.iloc[0]),
                  cv_rmse_nochange=float(T[T.kind == "nochange"].rmse_cond.iloc[0]),
                  rule="lowest 3-fold CV condition RMSE among Ridge-type and MLP candidates, TRAIN drugs only",
                  n_fold_drugs=[int((d.fold[train_drugs] == f).sum()) for f in range(D.N_FOLDS)])
        D.dump_json(ch, a.choice)
        print(json.dumps(ch, indent=1))
    else:
        ch = json.loads(Path(a.choice).read_text())
        lines, fit, ev, role = split_spec(d, a.setting)
        allp = np.arange(d.Y.shape[1])
        p = fit_predict(d, ch["kind"], ch["hp"], np.asarray(fit), allp, dev).numpy()
        lmn = fit_predict(d, "linemean", {}, np.asarray(fit), allp, dev).numpy()
        out = run_dir("mean", a.setting)
        np.save(out / "delta_hat.npy", p); np.save(out / "delta_linemean.npy", lmn)
        rec = dict(setting=a.setting, kind=ch["kind"], hp=ch["hp"], n_fit_drugs=len(fit),
                   fit_rmse_cond=rmse_cond(d, torch.as_tensor(p[:, fit]), fit)[0],
                   sha256_delta_hat=D.sha256_file(out / "delta_hat.npy"), path=str(out / "delta_hat.npy"))
        if a.setting == "dev":      # the dev-evaluation fold is development data
            rec["eval_rmse_cond"] = rmse_cond(d, torch.as_tensor(p[:, ev]), ev)[0]
            rec["eval_rmse_linemean"] = rmse_cond(d, torch.as_tensor(lmn[:, ev]), ev)[0]
            rec["eval_rmse_nochange"] = rmse_cond(d, torch.zeros(p[:, ev].shape), ev)[0]
        D.dump_json(rec, out / "record.json")
        print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
