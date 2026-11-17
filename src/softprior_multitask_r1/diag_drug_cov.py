#!/usr/bin/env python3
"""Stage R1-1 diagnostic D2: is there structure shared between drugs, is it stable, and can RNA predict it?

Everything is computed inside the 15 outer-training sets (and their inner folds). Outer-test labels and
LCCL labels are not read.

  (a) spectrum of the family-weighted drug x drug correlation of training AA (standardised per drug);
  (b) held-out reproducibility: share of the inner-test families' standardised AA variance captured by the
      top-k principal subspace of the inner-training families, against k/D (random subspace) and against a
      null in which each drug's training column is permuted independently;
  (c) family-bootstrap stability of the top-k subspace (mean squared cosine of principal angles);
  (d) predictability from RNA: inner-fold error of a whole-space Ridge whose targets are the scores of the
      top principal components (fitted on inner-training families), compared with single drugs.

Usage: diag_drug_cov.py --out-dir reports/softprior_multitask_r1/tables
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import r1_models as M  # noqa: E402
import r1_solvers as S  # noqa: E402
from r1_data import a1core  # noqa: E402

KS = [1, 2, 3, 5, 10, 20]


def wcov_eig(Z, wt):
    C = Z.T @ (wt[:, None] * Z)
    e, V = np.linalg.eigh(C)
    return e[::-1], V[:, ::-1]


def family_first_energy(A, fam):
    """Family-first mean of squared row norms."""
    f, inv = np.unique(fam, return_inverse=True)
    v = np.zeros(len(f))
    np.add.at(v, inv, (A ** 2).sum(axis=1))
    return float((v / np.bincount(inv)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=20)
    args = ap.parse_args()
    cfg = R.load_config()
    data = R.load_data(cfg)
    D = data.Y.shape[1]
    lams = cfg["objective"]["lambda_grid"]
    spec_rows, held_rows, stab_rows, pred_rows = [], [], [], []
    rs = np.random.RandomState(4242)
    for rep in sorted(data.splits.repeat.unique()):
        fold = data.outer_fold(rep)
        for k in sorted(set(fold.tolist())):
            sid = f"r{rep}f{k}"
            tr = fold != k
            X, Y, fam = data.rank[tr], data.Y[tr], data.fam[tr]
            wt = S.norm_weights(a1core.family_weights(fam))
            mu, sd = S.wmean_sd(Y, wt)
            Z = (Y - mu) / sd
            e, V = wcov_eig(Z, wt)
            spec_rows.append(dict(set=sid, **{f"ev_share_pc{i + 1}": e[i] / e.sum() for i in range(10)},
                                  participation_ratio=float(e.sum() ** 2 / (e ** 2).sum()),
                                  mean_offdiag_corr=float((np.sum(Z.T @ (wt[:, None] * Z)) - D) / (D * (D - 1)))))
            # (c) bootstrap stability of the subspace
            fams = np.unique(fam)
            aff = {kk: [] for kk in KS}
            for _ in range(args.n_boot):
                pick = rs.choice(fams, size=len(fams), replace=True)
                rows = np.concatenate([np.where(fam == f)[0] for f in pick])
                # resampled families keep equal total weight: weight rows by 1 / (rows of that family)
                cnt = np.array([np.sum(fam == f) for f in pick])
                wb = S.norm_weights(np.concatenate([np.full(c, 1.0 / c) for c in cnt]))
                mb, sb = S.wmean_sd(Y[rows], wb)
                Zb = (Y[rows] - mb) / np.where(sb > 1e-8, sb, 1.0)
                _, Vb = wcov_eig(Zb, wb)
                for kk in KS:
                    aff[kk].append(float(np.sum((V[:, :kk].T @ Vb[:, :kk]) ** 2) / kk))
            for kk in KS:
                stab_rows.append(dict(set=sid, k=kk, mean_sq_cosine=float(np.mean(aff[kk])),
                                      q05=float(np.quantile(aff[kk], 0.05)), random_expectation=kk / D))
            # (b) and (d) on the inner folds
            inner = a1core.group_folds(fam, cfg["split"]["inner_folds"], R.inner_seed(cfg, rep, k))
            for j in range(cfg["split"]["inner_folds"]):
                itr, ite = inner != j, inner == j
                wi = S.norm_weights(a1core.family_weights(fam[itr]))
                mi, si = S.wmean_sd(Y[itr], wi)
                Ztr, Zte = (Y[itr] - mi) / si, (Y[ite] - mi) / si
                _, Vi = wcov_eig(Ztr, wi)
                tot = family_first_energy(Zte, fam[ite])
                null = {kk: [] for kk in KS}
                for _ in range(args.n_null):
                    Zp = np.column_stack([Ztr[rs.permutation(len(Ztr)), d] for d in range(D)])
                    _, Vp = wcov_eig(Zp, wi)
                    for kk in KS:
                        null[kk].append(family_first_energy(Zte @ Vp[:, :kk], fam[ite]) / tot)
                for kk in KS:
                    held_rows.append(dict(set=sid, inner_fold=j, k=kk,
                                          heldout_share=family_first_energy(Zte @ Vi[:, :kk], fam[ite]) / tot,
                                          permuted_columns_null=float(np.mean(null[kk])), random_expectation=kk / D))
                # (d) Ridge from RNA to PC scores and to single drugs (whole space, eta = 0)
                n_pc = 10
                T_tr, T_te = Ztr @ Vi[:, :n_pc], Zte @ Vi[:, :n_pc]
                prep = M.Prep(X[itr], T_tr, fam[itr], X[ite], None, need_reduced=False)
                prep_d = M.Prep(X[itr], Y[itr], fam[itr], X[ite], None, need_reduced=False)
                f_te, inv = np.unique(fam[ite], return_inverse=True)
                cnt = np.bincount(inv)

                def ff(se):
                    agg = np.zeros((len(f_te), se.shape[1]))
                    np.add.at(agg, inv, se)
                    return (agg / cnt[:, None]).mean(axis=0)
                # error of predicting the training mean (zero in standardised units) on the same rows
                n_pc_loss = ff(prep.z_of(T_te) ** 2)
                n_d_loss = ff(prep_d.z_of(Y[ite]) ** 2)
                for lam in lams:
                    Pz = prep.predict_ridge_z("whole", lam)
                    l_pc = ff((Pz - prep.z_of(T_te)) ** 2)
                    Pd = prep_d.predict_ridge_z("whole", lam)
                    l_d = ff((Pd - prep_d.z_of(Y[ite])) ** 2)
                    row = dict(set=sid, inner_fold=j, lam=lam, n_test_families=len(f_te),
                               drugs_loss=float(l_d.mean()), drugs_null=float(n_d_loss.mean()))
                    for i in range(n_pc):
                        row[f"pc{i + 1}_loss"] = float(l_pc[i])
                        row[f"pc{i + 1}_null"] = float(n_pc_loss[i])
                    pred_rows.append(row)
    os.makedirs(args.out_dir, exist_ok=True)
    spec = pd.DataFrame(spec_rows)
    held = pd.DataFrame(held_rows)
    stab = pd.DataFrame(stab_rows)
    pred = pd.DataFrame(pred_rows)
    spec.to_csv(os.path.join(args.out_dir, "d2_spectrum_by_set.tsv"), sep="\t", index=False, float_format="%.5g")
    held_s = held.groupby("k")[["heldout_share", "permuted_columns_null", "random_expectation"]].agg(["mean", "min"])
    held_s.columns = [f"{a}_{b}" for a, b in held_s.columns]
    held_s.reset_index().to_csv(os.path.join(args.out_dir, "d2_heldout_subspace.tsv"), sep="\t", index=False, float_format="%.5g")
    stab_s = stab.groupby("k")[["mean_sq_cosine", "q05", "random_expectation"]].mean().reset_index()
    stab_s.to_csv(os.path.join(args.out_dir, "d2_bootstrap_subspace_stability.tsv"), sep="\t", index=False, float_format="%.5g")
    # Pool inner folds by test-family count and average over training sets. Each target is reported as
    # skill = 1 - loss / null, where null is the error of predicting the training mean on the same
    # held-out rows. (A first version reported the loss in training-SD units only; for components that
    # are fitted on the training families the held-out variance is far below the training variance, so
    # that number looked good without any prediction. Kept as *_v1_scale_error.tsv.)
    cols = [c for c in pred.columns if c.endswith("_loss") or c.endswith("_null")]
    pred_w = pred.copy()
    for c in cols:
        pred_w[c] = pred_w[c] * pred_w.n_test_families
    g = pred_w.groupby(["set", "lam"])[cols + ["n_test_families"]].sum()
    for c in cols:
        g[c] = g[c] / g.n_test_families
    pr = g.reset_index().groupby("lam")[cols].mean().reset_index()
    for t in ["drugs"] + [f"pc{i + 1}" for i in range(10)]:
        pr[f"{t}_skill"] = 1.0 - pr[f"{t}_loss"] / pr[f"{t}_null"]
    pr = pr[["lam"] + [c for c in pr.columns if c.endswith("_skill")] + ["drugs_loss", "drugs_null", "pc1_loss", "pc1_null",
                                                                          "pc2_loss", "pc2_null"]]
    pr.to_csv(os.path.join(args.out_dir, "d2_rna_predicts_components.tsv"), sep="\t", index=False, float_format="%.5g")
    print(spec.drop(columns="set").mean().to_string())
    print(held_s.to_string())
    print(stab_s.to_string())
    print(pr.to_string())


if __name__ == "__main__":
    main()
