"""Simple baselines. Everything fitted on source uses source-train rows only."""
import argparse, json, os
import numpy as np
import yaml

from .data import Data, condition_delta, MAN_DIR
from .tsg_adapter import sha256

ALPHAS = [1, 10, 100, 1000, 10000]


def base_dir():
    return os.path.join(os.environ["TB_RUNS"], "baselines_v1")


class SourceBaselines:
    """Loaded source-side baseline artefacts (scaler, ridge weights, same-drug group means)."""

    def __init__(self):
        man = json.load(open(os.path.join(MAN_DIR, "source_baselines.json")))
        for f, e in man["files"].items():
            assert sha256(os.path.join(base_dir(), f)) == e["sha256"], f"{f} changed after it was recorded"
        z = np.load(os.path.join(base_dir(), "source_ridge.npz"))
        self.mu, self.sd, self.W, self.xm, self.ym, self.alpha = z["mu"], z["sd"], z["W"], z["xm"], z["ym"], float(z["alpha"])
        g = np.load(os.path.join(base_dir(), "source_drug_group_mean.npz"), allow_pickle=False)
        self.group_ids = g["groups"].astype(str)
        self.group_mean = g["mean"]
        self.group_pos = {k: i for i, k in enumerate(self.group_ids)}
        self.group_ncells = g["n_cells"]

    def features(self, x0, fp):
        return np.concatenate([(np.asarray(x0, np.float64) - self.mu) / self.sd, np.asarray(fp, np.float64)], 1)

    def ridge_rows(self, x0, fp):
        return (self.features(x0, fp) - self.xm) @ self.W + self.ym

    def same_drug(self, drug_groups):
        idx = [self.group_pos[g] for g in drug_groups]      # KeyError if a scored drug has no source-train condition
        return self.group_mean[idx]


def fit_target_ridge(F_a, Y_a, alpha):
    """Dual-form ridge with an unpenalised intercept, for a handful of adapt rows."""
    xm, ym = F_a.mean(0), Y_a.mean(0)
    X = F_a - xm
    A = np.linalg.solve(X @ X.T + alpha * np.eye(len(X)), Y_a - ym)
    return lambda F: (F - xm) @ (X.T @ A) + ym


def fit_source(cfg):
    D = Data()
    os.makedirs(base_dir(), exist_ok=False)
    tr, va = D.source_split()
    x1 = D.source_x1()
    x0 = np.asarray(D.x0[: D.n_src], np.float64)
    delta = np.asarray(x1, np.float64) - x0
    di = D.rows.drug_idx.to_numpy()[: D.n_src]
    mu, sd = x0[tr].mean(0), x0[tr].std(0)
    sd[sd < 1e-8] = 1.0
    feat = lambda r: np.concatenate([(x0[r] - mu) / sd, D.fp[di[r]].astype(np.float64)], 1)
    xm, ym = np.zeros(978 + 2048), delta[tr].mean(0)
    # accumulate the normal equations in blocks
    n = len(tr)
    for i in range(0, n, 20000):
        xm += feat(tr[i:i + 20000]).sum(0)
    xm /= n
    XtX = np.zeros((len(xm), len(xm)))
    XtY = np.zeros((len(xm), 978))
    for i in range(0, n, 20000):
        F = feat(tr[i:i + 20000]) - xm
        XtX += F.T @ F
        XtY += F.T @ (delta[tr[i:i + 20000]] - ym)
    Fv = feat(va) - xm
    scores, Ws = {}, {}
    for al in ALPHAS:
        W = np.linalg.solve(XtX + al * np.eye(len(xm)), XtY)
        scores[al] = float(np.sqrt(((Fv @ W + ym - delta[va]) ** 2).mean(1)).mean())
        Ws[al] = W
    alpha = min(ALPHAS, key=lambda a_: (scores[a_], a_))
    np.savez(os.path.join(base_dir(), "source_ridge.npz"), mu=mu, sd=sd, W=Ws[alpha].astype(np.float32), xm=xm, ym=ym, alpha=alpha)

    # same-drug mean: condition deltas -> per cell -> equal mean over source-train cell lines
    rows_tr = D.rows.iloc[tr]
    cond, cd = condition_delta(rows_tr, delta[tr])
    cond["i"] = np.arange(len(cond))
    groups, means, ncell = [], [], []
    for g, sub in cond.groupby("drug_group", sort=True):
        per_cell = [cd[s.i.to_numpy()].mean(0) for _, s in sub.groupby("cell", sort=True)]
        groups.append(g)
        means.append(np.mean(per_cell, 0))
        ncell.append(len(per_cell))
    np.savez(os.path.join(base_dir(), "source_drug_group_mean.npz"), groups=np.array(groups), mean=np.stack(means).astype(np.float32),
             n_cells=np.array(ncell))
    man = dict(ridge_alpha=alpha, ridge_val_delta_rmse=scores, val_no_change_delta_rmse=float(np.sqrt((delta[va] ** 2).mean(1)).mean()),
               val_train_mean_delta_rmse=float(np.sqrt(((ym - delta[va]) ** 2).mean(1)).mean()),
               n_train_rows=int(n), n_val_rows=int(len(va)), n_drug_groups=len(groups), dir=base_dir(),
               files={f: dict(sha256=sha256(os.path.join(base_dir(), f))) for f in ["source_ridge.npz", "source_drug_group_mean.npz"]})
    json.dump(man, open(os.path.join(MAN_DIR, "source_baselines.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/transigen_adaptation_b1/protocol.yaml")
    a = ap.parse_args()
    fit_source(yaml.safe_load(open(a.config)))
