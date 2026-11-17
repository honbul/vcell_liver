"""Scoring for R1: Track A's primary metric and family bootstrap on prediction arrays.

Predictions are arrays [repeat, model row, drug] in the row order of `Data.models`. The error cube,
the macro normalised RMSE and the bootstrap come from Track A's metrics module unchanged.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import r1_solvers as S  # noqa: E402
from r1_data import a1core  # noqa: E402
import metrics as A1M  # noqa: E402  (Track A)


class Scorer:
    def __init__(self, data: R.Data):
        self.data = data
        cfg = data.cfg
        self.repeats = sorted(data.splits.repeat.unique())
        self.families = sorted(set(data.fam))
        fi = {f: i for i, f in enumerate(self.families)}
        self.fam_idx = np.array([fi[f] for f in data.fam])
        self.P = np.zeros((len(self.families), len(self.fam_idx)))          # family-averaging matrix
        self.P[self.fam_idx, np.arange(len(self.fam_idx))] = 1.0
        self.P /= self.P.sum(axis=1, keepdims=True)
        n, D = data.Y.shape
        self.folds = np.stack([data.outer_fold(r) for r in self.repeats])              # [rep, model]
        self.s_train = np.empty((len(self.repeats), n, D))                              # SD of the model's own outer-training set
        self.mu_train = np.empty((len(self.repeats), n, D))
        fold_sd = []
        for r in range(len(self.repeats)):
            for k in sorted(set(self.folds[r].tolist())):
                tr, te = self.folds[r] != k, self.folds[r] == k
                wt = S.norm_weights(a1core.family_weights(data.fam[tr]))
                mu, sd = S.wmean_sd(data.Y[tr], wt)
                self.s_train[r, te], self.mu_train[r, te] = sd, mu
                fold_sd.append(sd)
        self.s_bar = np.mean(fold_sd, axis=0)                                           # Track A ensemble normaliser
        b = cfg["metric"]["bootstrap"]
        self.W = A1M.bootstrap_weights(len(self.families), b["n"], b["seed"])
        self.ci = b["ci"]

    def family_mean(self, A: np.ndarray) -> np.ndarray:
        """[.., model, drug] -> [.., family, drug] within-family mean."""
        return np.einsum("fm,...md->...fd", self.P, A)

    def cube(self, pred: np.ndarray) -> np.ndarray:
        """F[repeat, drug, family] = within-family mean of squared normalised error."""
        assert pred.shape == self.s_train.shape and np.isfinite(pred).all()
        z2 = ((self.data.Y[None] - pred) / self.s_train) ** 2
        return np.swapaxes(self.family_mean(z2), 1, 2)

    def macro(self, F) -> float:
        return A1M.macro_nrmse(F)

    def summary_row(self, name: str, pred: np.ndarray) -> dict:
        F = self.cube(pred)
        per_rep = np.nanmean(A1M.drug_nrmse(F), axis=1)
        lo, hi = A1M.ci(A1M.boot_macro(F, self.W), *self.ci)
        Y = self.data.Y
        # ensemble of repeats (reported separately, never used for conclusions)
        ens = pred.mean(axis=0)
        z2 = ((Y - ens) / self.s_bar) ** 2
        ens_score = float(np.sqrt(self.family_mean(z2).mean(axis=0)).mean())
        # secondary: per-drug Spearman / raw RMSE / MAE on family means, averaged over repeats
        yf = self.family_mean(Y)
        sp, rm, ma = [], [], []
        for r in range(pred.shape[0]):
            pf = self.family_mean(pred[r])
            sp.append([A1M.safe_spearman(yf[:, d], pf[:, d]) for d in range(Y.shape[1])])
            rm.append(np.sqrt(self.family_mean((Y - pred[r]) ** 2).mean(axis=0)))
            ma.append(self.family_mean(np.abs(Y - pred[r])).mean(axis=0))
        sp = np.nanmean(np.array(sp), axis=0)
        row = dict(model=name, macro_drug_normalized_rmse=float(per_rep.mean()), ci_low=lo, ci_high=hi)
        row.update({f"repeat{r}": float(v) for r, v in zip(self.repeats, per_rep)})
        row.update(repeat_sd=float(per_rep.std(ddof=1)), ensemble_of_repeats_macro_nrmse=ens_score,
                   mean_spearman=float(np.nanmean(sp)), median_spearman=float(np.nanmedian(sp)),
                   mean_rmse_raw=float(np.mean(rm)), mean_mae_raw=float(np.mean(ma)))
        return row

    def difference_row(self, a: str, b: str, Fa: np.ndarray, Fb: np.ndarray) -> dict:
        d = A1M.boot_macro(Fa, self.W) - A1M.boot_macro(Fb, self.W)
        lo, hi = A1M.ci(d, *self.ci)
        da, db = A1M.drug_nrmse(Fa).mean(axis=0), A1M.drug_nrmse(Fb).mean(axis=0)
        return dict(model=a, reference=b, difference=self.macro(Fa) - self.macro(Fb), ci_low=lo, ci_high=hi,
                    boot_fraction_below_zero=float((d < 0).mean()), drugs_model_better=int((da < db).sum()),
                    drugs_reference_better=int((da > db).sum()), n_drugs=len(da))

    def per_drug(self, name: str, pred: np.ndarray) -> pd.DataFrame:
        F = self.cube(pred)
        Y = self.data.Y
        yf = self.family_mean(Y)
        sp = np.nanmean(np.array([[A1M.safe_spearman(yf[:, d], self.family_mean(pred[r])[:, d])
                                   for d in range(Y.shape[1])] for r in range(pred.shape[0])]), axis=0)
        rm = np.mean([np.sqrt(self.family_mean((Y - pred[r]) ** 2).mean(axis=0)) for r in range(pred.shape[0])], axis=0)
        ma = np.mean([self.family_mean(np.abs(Y - pred[r])).mean(axis=0) for r in range(pred.shape[0])], axis=0)
        return pd.DataFrame(dict(model=name, drug_id=self.data.drugs, nrmse=A1M.drug_nrmse(F).mean(axis=0),
                                 spearman=sp, rmse_raw=rm, mae_raw=ma))

    # ---- broad sensitivity versus drug preference (diagnostic) ----
    def preference_rows(self, name: str, pred: np.ndarray) -> dict:
        """Scale by the outer-training mean/SD of each drug, then split each cell's profile into its
        mean over drugs (broad sensitivity) and the remainder (drug preference). The prediction and
        the observation are each centred by their own cell mean."""
        from scipy.stats import spearmanr
        zy = (self.data.Y[None] - self.mu_train) / self.s_train
        zp = (pred - self.mu_train) / self.s_train
        broad_y, broad_p = zy.mean(axis=2), zp.mean(axis=2)                       # [rep, model]
        ry, rp = zy - broad_y[..., None], zp - broad_p[..., None]
        out = dict(model=name)
        bs, cell_rho, drug_rho, mse_b, mse_r = [], [], [], [], []
        for r in range(pred.shape[0]):
            by, bp = self.family_mean(broad_y[r][:, None])[:, 0], self.family_mean(broad_p[r][:, None])[:, 0]
            bs.append(spearmanr(by, bp)[0])
            rho = np.array([spearmanr(ry[r, i], rp[r, i])[0] for i in range(ry.shape[1])])
            cell_rho.append(self.family_mean(rho[:, None])[:, 0].mean())
            fy, fp = self.family_mean(ry[r]), self.family_mean(rp[r])
            drug_rho.append(np.nanmean([A1M.safe_spearman(fy[:, d], fp[:, d]) for d in range(fy.shape[1])]))
            mse_b.append(self.family_mean(((broad_y[r] - broad_p[r]) ** 2)[:, None]).mean())
            mse_r.append(self.family_mean((ry[r] - rp[r]) ** 2).mean())
        out.update(broad_sensitivity_spearman=float(np.mean(bs)),
                   preference_within_cell_spearman=float(np.mean(cell_rho)),
                   preference_per_drug_spearman=float(np.mean(drug_rho)),
                   broad_mse=float(np.mean(mse_b)), preference_mse=float(np.mean(mse_r)))
        return out


def load_oof(run_dir: str, name: str) -> dict:
    z = np.load(os.path.join(run_dir, "oof", f"{name}.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def save_oof(run_dir: str, name: str, pred: np.ndarray, models: list, drugs: list, **extra) -> None:
    os.makedirs(os.path.join(run_dir, "oof"), exist_ok=True)
    out = os.path.join(run_dir, "oof", f"{name}.npz")
    if os.path.exists(out):
        raise FileExistsError(f"refusing to overwrite {out}")
    np.savez(out[:-4] + ".tmp.npz", pred=pred, models=np.array(models), drugs=np.array(drugs), **extra)
    os.replace(out[:-4] + ".tmp.npz", out)
