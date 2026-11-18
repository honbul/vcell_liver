"""Train-only representation and fixed state atlas, per cell line.

Fit cells: wells of TRAIN drugs (all doses, both replicates) plus vehicle wells, at most
`fit_cells_per_well` cells per well (fixed seed) so that big wells do not dominate.
  - log1p(CP10k) on human genes; 2,000 HVG by binned normalised dispersion on fit cells
  - standardise with fit-cell mean / SD; PCA (n_pcs) on fit cells
  - Gaussian mixtures (K in candidates + sensitivity) on fit-cell PCA scores; soft membership
    s_k(z) = GMM posterior. Development/test cells are only projected, never fitted.
Outputs per line in cache_dir/atlas_<line>/: Z.npy, S_K<k>.npy, gmm_K<k>.npz, decoder.npz,
well_gene_means.npy + wells.tsv (row order), fit_info.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def lognorm(X: sp.csr_matrix, tot: np.ndarray) -> sp.csr_matrix:
    Y = sp.diags(1e4 / np.maximum(tot, 1)) @ X.astype(np.float32)
    Y = Y.tocsr()
    Y.data = np.log1p(Y.data)
    return Y


def hvg(Y: sp.csr_matrix, n: int, min_detect: float) -> np.ndarray:
    mean = np.asarray(Y.mean(0)).ravel()
    sq = np.asarray(Y.multiply(Y).mean(0)).ravel()
    var = sq - mean ** 2
    det = np.asarray((Y > 0).mean(0)).ravel()
    ok = det >= min_detect
    disp = np.log(np.maximum(var, 1e-12) / np.maximum(mean, 1e-12))
    bins = pd.qcut(pd.Series(mean[ok]).rank(method="first"), 20, labels=False).values
    z = np.full(len(mean), -np.inf)
    dz = disp[ok]
    zz = np.empty_like(dz)
    for b in range(20):
        m = bins == b
        zz[m] = (dz[m] - dz[m].mean()) / (dz[m].std() + 1e-8)
    z[ok] = zz
    return np.sort(np.argsort(-z)[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--line", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    A = cfg["atlas"]
    cache = cfg["cache_dir"]
    out = f"{cache}/atlas_{args.line}"
    os.makedirs(out, exist_ok=True)
    split = pd.read_csv(args.split, sep="\t").set_index("drug").role
    X = sp.load_npz(f"{cache}/counts_{args.line}.npz").tocsr()
    obs = pd.read_parquet(f"{cache}/obs_{args.line}.parquet")
    obs["role"] = np.where(obs.vehicle, "control", obs.drug.map(split))
    assert obs.role.notna().all()
    log(args.line, X.shape, obs.role.value_counts().to_dict())

    rng = np.random.default_rng(A["fit_seed"])
    fit_idx = []
    for _, idx in obs[obs.role.isin(["train", "control"])].groupby("Combo").groups.items():
        idx = np.asarray(idx)
        fit_idx.append(rng.choice(idx, min(len(idx), A["fit_cells_per_well"]), replace=False))
    fit_idx = np.sort(np.concatenate(fit_idx))
    Y = lognorm(X, obs.n_umi_human.values)
    del X
    genes = pd.read_csv(f"{cache}/genes_human.tsv", sep="\t")
    hv = hvg(Y[fit_idx], A["n_hvg"], A["min_detect_frac"])
    Yh = Y[:, hv].tocsr()
    del Y
    F = Yh[fit_idx].toarray()
    mu, sd = F.mean(0), F.std(0)
    sd = np.maximum(sd, 1e-3)
    pca = PCA(n_components=A["n_pcs"], svd_solver="full", random_state=0).fit((F - mu) / sd)
    comp = pca.components_.astype(np.float32)                # [d, G]
    W = (comp / sd).T                                         # [G, d]
    Z = np.asarray(Yh @ W) - (mu / sd) @ comp.T
    Z = Z.astype(np.float32)
    np.save(f"{out}/Z.npy", Z)
    np.savez(f"{out}/decoder.npz", mu=mu.astype(np.float32), sd=sd.astype(np.float32), comp=comp,
             hvg_index=hv, hvg_id=genes.id.values[hv], hvg_name=genes.gene_short_name.values[hv],
             explained_variance_ratio=pca.explained_variance_ratio_)
    Zf = Z[fit_idx]
    sub = rng.choice(len(Zf), min(len(Zf), 3000), replace=False)
    bw = float(np.median(pdist(Zf[sub])))
    info = dict(line=args.line, n_cells=int(len(obs)), n_fit=int(len(fit_idx)), n_hvg=int(len(hv)),
                pca_explained=float(pca.explained_variance_ratio_.sum()), mmd_bandwidth=bw, gmm={})
    for K in sorted(set(A["K_candidates"]) | set(A["K_sensitivity"])):
        gm = GaussianMixture(K, covariance_type=A["gmm_covariance"], reg_covar=A["gmm_reg_covar"],
                             n_init=A["gmm_n_init"], random_state=0, max_iter=500).fit(Zf)
        order = np.argsort(-gm.weights_)                    # states ordered by fit-cell weight
        S = gm.predict_proba(Z)[:, order].astype(np.float32)
        np.save(f"{out}/S_K{K}.npy", S)
        np.savez(f"{out}/gmm_K{K}.npz", weights=gm.weights_[order], means=gm.means_[order],
                 covariances=gm.covariances_[order])
        Sf = S[fit_idx]
        info["gmm"][K] = dict(converged=bool(gm.converged_), fit_occupancy=Sf.mean(0).round(4).tolist(),
                              mean_max_membership=float(Sf.max(1).mean()),
                              frac_max_membership_gt_0_9=float((Sf.max(1) > 0.9).mean()))
        log("K", K, info["gmm"][K])
    # gene-space (log1p CP10k, HVG) mean of every well, for the Delta-mean readout
    wells = obs.groupby("Combo").agg(drug=("drug", "first"), dose=("dose", "first"),
                                     replicate=("replicate", "first"), n=("cell", "size")).reset_index()
    code = pd.Categorical(obs.Combo, categories=wells.Combo).codes
    M = sp.csr_matrix((np.ones(len(obs)), (code, np.arange(len(obs)))), shape=(len(wells), len(obs)))
    G = np.asarray((M @ Yh).todense()) / wells.n.values[:, None]
    np.save(f"{out}/well_gene_means.npy", G.astype(np.float32))
    wells.to_csv(f"{out}/wells.tsv", sep="\t", index=False)
    obs[["cell", "Combo", "drug", "dose", "replicate", "culture_plate", "vehicle", "role", "n_umi_human"]].to_parquet(
        f"{out}/obs.parquet", index=False)
    np.save(f"{out}/fit_idx.npy", fit_idx)
    json.dump(info, open(f"{out}/fit_info.json", "w"), indent=1)
    log("done", json.dumps(info))


if __name__ == "__main__":
    main()
