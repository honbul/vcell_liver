#!/usr/bin/env python
"""F3 step 2b+3: pseudobulk deltas, DEG truth, and leave-one-cell-line-out (LOCO) baselines.

Inputs: per-line sufficient statistics from f3_pseudobulk.py (assets/f3/pseudobulk/<line>.npz).
Common gene set C = Ensembl IDs present in all 4 author-filtered panels (each panel = genes with
mean > 0.1 UMI/cell in that screen). Per line and target (>= min_cells cells):
  delta = mean log1p(CP10k) of perturbed cells - batch-matched control mean, where the control
  mean is the gem_group control means weighted by the target's gem_group cell composition.
  DEG truth: Welch t vs batch-matched control mean using pooled control variance; BH FDR<0.05
  over C per target.
Fold L (held out): visible = treated+control of the 3 source lines + L controls only.
  Panel_L = top-N genes by variance of pooled source deltas (source lines only) U measured target
  genes (identity only). Metrics exclude the perturbed gene itself.
Conditions:
  A   unseen context, seen target (target perturbed in >=1 source line)
  Bn  natural double OOD (target never perturbed in any source line)
  Bs  simulated double OOD: random 20% of L's targets (split S_L) removed from all source lines
  C   in-line unseen perturbation: train on L's own other 80% targets, test S_L
  D   split-half ceiling on S_L (half-0 delta predicts half-1 delta)
Hyperparameters chosen by inner leave-one-source-line-out (A/Bs) or 5-fold target CV inside L's
training targets (C); never with L's treated cells.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

LINES = ["K562", "RPE1", "HepG2", "Jurkat"]
RIDGE_K = [5, 10, 25, 50, 100, 200]
RIDGE_LAM = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def bh(p):
    """Benjamini-Hochberg along last axis."""
    p = np.asarray(p)
    m = p.shape[-1]
    o = np.argsort(p, axis=-1)
    ps = np.take_along_axis(p, o, -1) * m / np.arange(1, m + 1)
    ps = np.minimum.accumulate(ps[..., ::-1], axis=-1)[..., ::-1]
    q = np.empty_like(ps)
    np.put_along_axis(q, o, np.minimum(ps, 1), -1)
    return q


def load_common_genes(pb):
    sets = []
    for L in LINES:
        d = np.load(pb / f"{L}.npz")
        sets.append(set(d["var_ids"].tolist()))
    C = sorted(set.intersection(*sets))
    return np.array(C)


def build_line(pb, L, C, min_cells, fdr):
    d = np.load(pb / f"{L}.npz")
    vid = d["var_ids"]; pos = {g: i for i, g in enumerate(vid)}
    col = np.array([pos[g] for g in C])
    groups = d["groups"]; n = d["n"].astype(float); c = int(d["ctrl_index"])
    keep = np.where((np.arange(len(groups)) != c) & (n >= min_cells))[0]
    S = d["S"][:, col]; SS = d["SS"][:, col]; S0 = d["S0"][:, col]
    n0 = d["n0"].astype(float)
    n_gem = d["n_gem"].astype(float)
    nc_gem = n_gem[c]
    cmean_gem = d["cS"][:, col] / nc_gem[:, None]
    nc = n[c]
    cmean = S[c] / nc
    cvar = (SS[c] - nc * cmean ** 2) / (nc - 1)
    nt = n[keep]
    mean = S[keep] / nt[:, None]
    var = np.maximum((SS[keep] - nt[:, None] * mean ** 2) / (nt[:, None] - 1), 0)
    w = n_gem[keep] / nt[:, None]
    bm = w @ cmean_gem
    delta = mean - bm
    h0 = S0[keep] / n0[keep][:, None] - bm
    h1 = (S[keep] - S0[keep]) / (nt - n0[keep])[:, None] - bm
    se2a = var / nt[:, None]; se2b = cvar[None, :] / nc
    se = np.sqrt(se2a + se2b)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, delta / se, 0.0)
        df = (se2a + se2b) ** 2 / (se2a ** 2 / (nt[:, None] - 1) + se2b ** 2 / (nc - 1))
    df = np.nan_to_num(df, nan=1.0)
    p = 2 * stats.t.sf(np.abs(tstat), df)
    q = bh(p)
    deg = q < fdr
    # knockdown (linear scale, in the line's own panel) relative to line-level control
    SL = d["SL"]; self_col = d["self_col"]
    ctrl_lin = SL[c] / n[c]
    kd = np.full(len(keep), np.nan)
    for i, gi in enumerate(keep):
        if self_col[gi] >= 0 and ctrl_lin[self_col[gi]] > 0:
            kd[i] = 1 - (SL[gi, self_col[gi]] / n[gi]) / ctrl_lin[self_col[gi]]
    gid = d["group_gid"][keep]
    cpos = {g: i for i, g in enumerate(C)}
    tgene = np.array([cpos.get(g, -1) for g in gid])
    return dict(line=L, targets=groups[keep], gid=gid, n=nt, delta=delta, h0=h0, h1=h1, tstat=tstat,
                q=q, deg=deg, kd=kd, tgene=tgene, ctrl_mean=cmean, n_ctrl=nc)


# ---------------------------------------------------------------- predictors
def ridge_svd(Y):
    b = Y.mean(1)
    Yc = Y - b[:, None]
    U, s, _ = np.linalg.svd(Yc, full_matrices=False)
    return b, Yc, U, s


def fit_ridge(Y, rows, K, lam, svd=None):
    """Ahlmann-Eltze-style bilinear ridge. Y: genes x N training deltas; rows: panel index of each
    column's perturbed gene (>=0). Gene embedding G = top-K PCs of centred Y; perturbation
    embedding = G[row]; Y - b ~ G W P^T with ridge penalty lam on both sides."""
    b, Yc, U, s = svd if svd is not None else ridge_svd(Y)
    K = min(K, U.shape[1])
    G = U[:, :K] * s[:K]
    P = G[rows]
    A = np.linalg.solve(G.T @ G + lam * np.eye(K), G.T @ Yc)
    W = np.linalg.solve(P.T @ P + lam * np.eye(K), (A @ P).T).T
    return b, G, W


def ridge_predict(model, rows):
    b, G, W = model
    out = np.tile(b, (len(rows), 1))
    ok = rows >= 0
    if ok.any():
        out[ok] += (G @ W @ G[rows[ok]].T).T
    return out


def rmse_excl(P, Y, rows):
    m = np.ones_like(Y, dtype=bool)
    ok = rows >= 0
    m[np.where(ok)[0], rows[ok]] = False
    return np.sqrt(((P - Y) ** 2 * m).sum(1) / m.sum(1))


class Source:
    """Training material for one fold: dict line -> (targets, deltas on panel, panel row)."""

    def __init__(self, data, lines, panel, gpos, exclude=frozenset()):
        self.lines = lines
        self.d = {}
        for s in lines:
            D = data[s]
            keep = np.array([t not in exclude for t in D["targets"]])
            rows = np.array([gpos.get(g, -1) for g in D["tgene"][keep]])
            self.d[s] = dict(targets=D["targets"][keep], Y=D["delta"][keep][:, panel], rows=rows)
            self.d[s]["tidx"] = {t: i for i, t in enumerate(self.d[s]["targets"])}

    def global_mean(self, lines=None):
        lines = lines or self.lines
        return np.mean([self.d[s]["Y"].mean(0) for s in lines], axis=0)

    def same_target(self, targets, lines=None, order=None):
        """mean over lines having t; with order: first line in order having t. NaN rows if unseen."""
        lines = lines or self.lines
        out = np.full((len(targets), self.d[lines[0]]["Y"].shape[1]), np.nan)
        for i, t in enumerate(targets):
            if order is not None:
                for s in order:
                    if s in lines and t in self.d[s]["tidx"]:
                        out[i] = self.d[s]["Y"][self.d[s]["tidx"][t]]; break
            else:
                v = [self.d[s]["Y"][self.d[s]["tidx"][t]] for s in lines if t in self.d[s]["tidx"]]
                if v:
                    out[i] = np.mean(v, 0)
        return out

    def stacked(self, lines=None):
        lines = lines or self.lines
        Y = np.concatenate([self.d[s]["Y"] for s in lines]).T
        rows = np.concatenate([self.d[s]["rows"] for s in lines])
        ok = rows >= 0
        return Y[:, ok], rows[ok]


def select_ridge_inner(src):
    """Leave-one-source-line-out among source lines; minimise mean RMSE (self gene excluded)."""
    res = {(K, lam): [] for K in RIDGE_K for lam in RIDGE_LAM}
    for v in src.lines:
        tr = [s for s in src.lines if s != v]
        Y, rows = src.stacked(tr)
        svd = ridge_svd(Y)
        for K, lam in res:
            m = fit_ridge(Y, rows, K, lam, svd)
            P = ridge_predict(m, src.d[v]["rows"])
            res[(K, lam)].append(rmse_excl(P, src.d[v]["Y"], src.d[v]["rows"]).mean())
    res = {k: float(np.mean(v)) for k, v in res.items()}
    best = min(res, key=res.get)
    return best, res


def select_calibration_inner(src):
    """y ~ a * same-target(other sources) + b * global(other sources), least squares, inner LOCO."""
    xs, gs, ys = [], [], []
    for v in src.lines:
        tr = [s for s in src.lines if s != v]
        st = src.same_target(src.d[v]["targets"], lines=tr)
        ok = ~np.isnan(st[:, 0])
        g = src.global_mean(tr)
        xs.append(st[ok].ravel()); gs.append(np.tile(g, ok.sum())); ys.append(src.d[v]["Y"][ok].ravel())
    X = np.stack([np.concatenate(xs), np.concatenate(gs)], 1)
    coef, *_ = np.linalg.lstsq(X, np.concatenate(ys), rcond=None)
    return [float(c) for c in coef]


def select_ridge_inline(Y, rows, seed, nfold=5):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, nfold, Y.shape[1])
    res = {(K, lam): [] for K in RIDGE_K for lam in RIDGE_LAM}
    for f in range(nfold):
        tr = fold != f
        svd = ridge_svd(Y[:, tr])
        for K, lam in res:
            m = fit_ridge(Y[:, tr], rows[tr], K, lam, svd)
            P = ridge_predict(m, rows[~tr])
            res[(K, lam)].append(rmse_excl(P, Y[:, ~tr].T, rows[~tr]).mean())
    res = {k: float(np.mean(v)) for k, v in res.items()}
    return min(res, key=res.get), res


# ---------------------------------------------------------------- metrics
def per_target_metrics(P, Y, deg, tst, rows, m_src, ntop=20):
    Pm, Ym = P.mean(0), Y.mean(0)  # evaluation-side perturbation centroids (Systema-like)
    out = []
    for i in range(Y.shape[0]):
        msk = np.ones(Y.shape[1], bool)
        if rows[i] >= 0:
            msk[rows[i]] = False
        p, y, dg, t = P[i, msk], Y[i, msk], deg[i, msk], tst[i, msk]
        r = dict(rmse=float(np.sqrt(np.mean((p - y) ** 2))))
        r["pearson"] = float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 else np.nan
        pc, yc = p - m_src[msk], y - m_src[msk]
        r["pearson_centered"] = float(np.corrcoef(pc, yc)[0, 1]) if pc.std() > 0 else np.nan
        pp, yp = p - Pm[msk], y - Ym[msk]
        r["pearson_pertcentered"] = float(np.corrcoef(pp, yp)[0, 1]) if pp.std() > 1e-12 else np.nan
        nd = int(dg.sum())
        r["n_deg"] = nd
        cand = np.where(dg)[0] if nd >= ntop else np.arange(len(y))
        top = cand[np.argsort(-np.abs(t[cand]))[:ntop]]
        r["dir_top20"] = float(np.mean(np.sign(p[top]) == np.sign(y[top])))
        if 0 < nd < len(y):
            r["deg_auprc"] = float(average_precision_score(dg, np.abs(p)))
            r["deg_prevalence"] = nd / len(y)
            k = nd
            pred_top = np.argsort(-np.abs(p), kind="stable")[:k]
            r["deg_f1_at_k"] = float(dg[pred_top].mean()) if p.std() > 0 else float(nd / len(y))
        else:
            r["deg_auprc"] = r["deg_prevalence"] = r["deg_f1_at_k"] = np.nan
        out.append(r)
    return out


def retrieval_rank(P, Y, m_src, gene_mask):
    """Normalised rank (0 best, 0.5 chance) of the true target's prediction among all predictions
    in the condition, by Pearson over non-target readout genes after subtracting each side's
    centroid across the condition's targets (so a shared response cannot drive retrieval)."""
    Yc = (Y - Y.mean(0))[:, gene_mask]; Pc = (P - P.mean(0))[:, gene_mask]
    if len(Y) < 2:
        return np.full(len(Y), np.nan)
    Yz = (Yc - Yc.mean(1, keepdims=True)); Yz /= np.linalg.norm(Yz, axis=1, keepdims=True) + 1e-12
    Pz = (Pc - Pc.mean(1, keepdims=True)); nrm = np.linalg.norm(Pz, axis=1, keepdims=True)
    Pz = np.where(nrm > 1e-12, Pz / np.maximum(nrm, 1e-12), 0.0)
    Cm = Yz @ Pz.T
    ranks = np.array([stats.rankdata(-Cm[i])[i] for i in range(len(Cm))])
    return (ranks - 1) / (len(Cm) - 1)


def boot_ci(x, nboot, rng):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x)
    bs = rng.choice(x, (nboot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudobulk", default="assets/f3/pseudobulk")
    ap.add_argument("--assets", default="assets/f3")
    ap.add_argument("--out", default="reports/feedback/F3")
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--nboot", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    pb = Path(args.pseudobulk); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    adir = Path(args.assets); (adir / "matrices").mkdir(parents=True, exist_ok=True); (adir / "loco").mkdir(parents=True, exist_ok=True)
    C = load_common_genes(pb)
    data = {}
    for L in LINES:
        data[L] = build_line(pb, L, C, args.min_cells, args.fdr)
        D = data[L]
        np.savez(adir / "matrices" / f"{L}_delta.npz", genes=C, **{k: v for k, v in D.items() if k != "line"})
        print(L, "targets", len(D["targets"]), "median DEG", np.median(D["deg"].sum(1)), f"{time.time()-t0:.0f}s", flush=True)
    all_target_genes = sorted({g for L in LINES for g in data[L]["tgene"] if g >= 0})
    line_summary = []
    for L in LINES:
        D = data[L]
        line_summary.append(dict(line=L, n_targets=len(D["targets"]), n_ctrl=int(D["n_ctrl"]),
                                 n_targets_measured_in_C=int((D["tgene"] >= 0).sum()),
                                 median_cells_per_target=float(np.median(D["n"])),
                                 median_n_deg_fdr05=float(np.median(D["deg"].sum(1))),
                                 frac_targets_ge10_deg=float((D["deg"].sum(1) >= 10).mean()),
                                 median_kd=float(np.nanmedian(D["kd"])), frac_kd_ge50=float(np.nanmean(D["kd"] >= 0.5)),
                                 median_delta_l2=float(np.median(np.linalg.norm(D["delta"], axis=1)))))
    pd.DataFrame(line_summary).to_csv(out / "line_pseudobulk_summary.tsv", sep="\t", index=False)
    # cross-line same-target delta similarity (descriptive, all lines; no model selection uses it)
    sim_rows = []
    for a in LINES:
        for b in LINES:
            if a >= b:
                continue
            ta = {t: i for i, t in enumerate(data[a]["targets"])}
            common = [t for t in data[b]["targets"] if t in ta]
            ib = {t: i for i, t in enumerate(data[b]["targets"])}
            r = [np.corrcoef(data[a]["delta"][ta[t]], data[b]["delta"][ib[t]])[0, 1] for t in common]
            sim_rows.append(dict(line_a=a, line_b=b, n_common=len(common), median_same_target_pearson_C=float(np.median(r))))
    pd.DataFrame(sim_rows).to_csv(out / "cross_line_same_target_similarity.tsv", sep="\t", index=False)

    rows_all, hp, fold_info, splits = [], {}, [], {}
    rng_boot = np.random.default_rng(args.seed)
    for L in LINES:
        S = [s for s in LINES if s != L]
        D = data[L]
        # ---- panel from source lines only
        pooled = np.concatenate([data[s]["delta"] for s in S])
        v = pooled.var(0)
        hvg = np.argsort(-v)[: args.n_hvg]
        panel = np.array(sorted(set(hvg.tolist()) | set(all_target_genes)))
        gpos = {g: i for i, g in enumerate(panel)}
        np.save(adir / "loco" / f"panel_{L}.npy", C[panel])
        nontarget_mask = ~np.isin(panel, all_target_genes)
        # ---- control-profile similarity (L controls only)
        csim = {s: float(np.corrcoef(D["ctrl_mean"], data[s]["ctrl_mean"])[0, 1]) for s in S}
        order = sorted(S, key=lambda s: -csim[s])
        # ---- split S_L
        h = int(hashlib.md5(L.encode()).hexdigest()[:6], 16)
        rng = np.random.default_rng(args.seed + h)
        nT = len(D["targets"])
        test_idx = np.sort(rng.choice(nT, int(round(args.test_frac * nT)), replace=False))
        test_set = set(D["targets"][test_idx])
        splits[L] = sorted(test_set)
        src = Source(data, S, panel, gpos)
        m_src = src.global_mean()
        rows_L = np.array([gpos.get(g, -1) for g in D["tgene"]])
        Ytrue = D["delta"][:, panel]; DEG = D["deg"][:, panel]; TST = D["tstat"][:, panel]
        seen = np.array([any(t in src.d[s]["tidx"] for s in S) for t in D["targets"]])
        # ---- hyperparameters (sources only)
        (K, lam), rgrid = select_ridge_inner(src)
        ab = select_calibration_inner(src)
        Ys, rs = src.stacked()
        ridge = fit_ridge(Ys, rs, K, lam)
        src_x = Source(data, S, panel, gpos, exclude=frozenset(test_set))
        (Kx, lamx), _ = select_ridge_inner(src_x)
        Yx, rx = src_x.stacked()
        ridge_x = fit_ridge(Yx, rx, Kx, lamx)
        m_src_x = src_x.global_mean()
        # in-line
        tr_idx = np.setdiff1d(np.arange(nT), test_idx)
        ok_tr = rows_L[tr_idx] >= 0
        Yin = Ytrue[tr_idx][ok_tr].T; rin = rows_L[tr_idx][ok_tr]
        (Ki, lami), _ = select_ridge_inline(Yin, rin, args.seed + h)
        ridge_in = fit_ridge(Yin, rin, Ki, lami)
        m_in = Ytrue[tr_idx].mean(0)
        hp[L] = dict(ridge_A=dict(K=K, lam=lam, inner_rmse={f"{k[0]}_{k[1]}": v for k, v in rgrid.items()}),
                     calib_A=dict(a_same=ab[0], b_global=ab[1]), ridge_Bs=dict(K=Kx, lam=lamx), ridge_C=dict(K=Ki, lam=lami))
        fold_info.append(dict(heldout=L, sources=",".join(S), panel_size=len(panel), n_hvg=args.n_hvg,
                              n_target_genes_in_panel=len(all_target_genes), nearest_line_by_control=order[0],
                              ctrl_similarity=json.dumps({s: round(csim[s], 4) for s in S}),
                              n_targets_L=nT, n_seen=int(seen.sum()), n_natural_double_ood=int((~seen).sum()),
                              n_test_split=len(test_idx), n_test_split_seen=int(seen[test_idx].sum()),
                              ridge_A=f"K={K},lam={lam}", calib_A=f"a={ab[0]:.3f},b={ab[1]:.3f}",
                              ridge_Bs=f"K={Kx},lam={lamx}", ridge_C=f"K={Ki},lam={lami}"))
        print(L, fold_info[-1], f"{time.time()-t0:.0f}s", flush=True)

        test_pos = set(test_idx.tolist())

        def evaluate(cond, idx, preds, m_ref, truth=None):
            Yt = Ytrue[idx] if truth is None else truth
            for name, P in preds.items():
                mets = per_target_metrics(P, Yt, DEG[idx], TST[idx], rows_L[idx], m_ref)
                rr = retrieval_rank(P, Yt, m_ref, nontarget_mask)
                for j, i in enumerate(idx):
                    rows_all.append(dict(heldout=L, condition=cond, predictor=name, target=D["targets"][i],
                                         n_cells=D["n"][i], kd=D["kd"][i], in_test_split=bool(i in test_pos),
                                         retrieval_rank=rr[j], **mets[j]))
            np.savez_compressed(adir / "loco" / f"pred_{L}_{cond}.npz", targets=D["targets"][idx],
                                **{k: v.astype(np.float32) for k, v in preds.items()})

        zeros = lambda k: np.zeros((len(k), len(panel)))
        # A: seen targets, full sources
        ia = np.where(seen)[0]
        st = src.same_target(D["targets"][ia]); nn = src.same_target(D["targets"][ia], order=order)
        evaluate("A_unseen_ctx_seen_target", ia, {
            "no_change": zeros(ia), "global_mean": np.tile(m_src, (len(ia), 1)), "same_target_mean": st,
            "nearest_line_same_target": nn, "calibrated_same_target": ab[0] * st + ab[1] * m_src,
            "ridge_pca": ridge_predict(ridge, rows_L[ia])}, m_src)
        # Bn: natural double OOD
        ib = np.where(~seen)[0]
        if len(ib):
            evaluate("Bn_double_ood_natural", ib, {"no_change": zeros(ib), "global_mean": np.tile(m_src, (len(ib), 1)),
                                                    "ridge_pca": ridge_predict(ridge, rows_L[ib])}, m_src)
        # Bs: simulated double OOD (S_L removed from sources)
        evaluate("Bs_double_ood_simulated", test_idx, {
            "no_change": zeros(test_idx), "global_mean": np.tile(m_src_x, (len(test_idx), 1)),
            "ridge_pca": ridge_predict(ridge_x, rows_L[test_idx])}, m_src_x)
        # C: in-line unseen perturbation
        evaluate("C_inline_unseen_target", test_idx, {
            "no_change": zeros(test_idx), "global_mean": np.tile(m_in, (len(test_idx), 1)),
            "ridge_pca": ridge_predict(ridge_in, rows_L[test_idx])}, m_src)
        # D: split-half ceiling (seen ctx, seen target) on S_L; truth = half-1
        evaluate("D_splithalf_ceiling", test_idx, {"no_change": zeros(test_idx), "half0_delta": D["h0"][test_idx][:, panel]},
                 m_src, truth=D["h1"][test_idx][:, panel])
        print(L, "evaluated", f"{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows_all)
    df.to_csv(adir / "loco" / "per_target_metrics.tsv.gz", sep="\t", index=False)
    mets = ["rmse", "pearson", "pearson_centered", "pearson_pertcentered", "dir_top20", "deg_auprc", "deg_f1_at_k", "retrieval_rank"]
    summ = df.groupby(["heldout", "condition", "predictor"])[mets + ["deg_prevalence"]].mean()
    summ["n_targets"] = df.groupby(["heldout", "condition", "predictor"]).size()
    summ.reset_index().to_csv(out / "loco_summary.tsv", sep="\t", index=False, float_format="%.5f")
    # A restricted to S_L for the 2x2 table
    sub = df[(df.condition == "A_unseen_ctx_seen_target") & df.in_test_split].copy()
    sub["condition"] = "A_on_test_split"
    s2 = sub.groupby(["heldout", "condition", "predictor"])[mets].mean()
    s2["n_targets"] = sub.groupby(["heldout", "condition", "predictor"]).size()
    s2.reset_index().to_csv(out / "loco_A_on_test_split.tsv", sep="\t", index=False, float_format="%.5f")
    # 2x2 OOD: per fold, best predictor by mean RMSE (descriptive) + no-change
    cells = {("seen_ctx", "seen_target"): ("D_splithalf_ceiling", df),
             ("seen_ctx", "unseen_target"): ("C_inline_unseen_target", df),
             ("unseen_ctx", "seen_target"): ("A_on_test_split", sub),
             ("unseen_ctx", "unseen_target"): ("Bs_double_ood_simulated", df)}
    ood = []
    for L in LINES:
        for (ctx, tg), (cond, src_df) in cells.items():
            x = src_df[(src_df.heldout == L) & (src_df.condition == cond)]
            g = x.groupby("predictor")[mets].mean()
            best = g["rmse"].idxmin()
            nc = g.loc["no_change"]
            ood.append(dict(heldout=L, context=ctx, target=tg, condition=cond, n_targets=int(x.target.nunique()),
                            best_by_rmse=best, best_rmse=g.loc[best, "rmse"], no_change_rmse=nc["rmse"],
                            best_pearson=g.loc[best, "pearson"], best_pearson_centered=g.loc[best, "pearson_centered"],
                            best_deg_auprc=g.loc[best, "deg_auprc"], no_change_deg_auprc=nc["deg_auprc"],
                            best_retrieval_rank=g.loc[best, "retrieval_rank"],
                            best_pearson_pertcentered=g.loc[best, "pearson_pertcentered"],
                            max_pearson_pertcentered=g["pearson_pertcentered"].max(),
                            max_pearson_pertcentered_by=g["pearson_pertcentered"].idxmax() if g["pearson_pertcentered"].notna().any() else "",
                            max_pearson_centered=g["pearson_centered"].max(),
                            max_pearson_centered_by=g["pearson_centered"].idxmax() if g["pearson_centered"].notna().any() else ""))
    pd.DataFrame(ood).to_csv(out / "ood_2x2.tsv", sep="\t", index=False, float_format="%.5f")
    # paired bootstrap vs references
    refs = {"A_unseen_ctx_seen_target": ["same_target_mean", "no_change"], "Bn_double_ood_natural": ["global_mean", "no_change"],
            "Bs_double_ood_simulated": ["global_mean", "no_change"], "C_inline_unseen_target": ["global_mean", "no_change"]}
    pb_rows = []
    for (L, cond), x in df.groupby(["heldout", "condition"]):
        if cond not in refs:
            continue
        piv = {m: x.pivot(index="target", columns="predictor", values=m) for m in mets}
        for ref in refs[cond]:
            for pred in piv["rmse"].columns:
                if pred == ref:
                    continue
                for m in ["rmse", "pearson", "pearson_centered", "pearson_pertcentered", "deg_auprc", "retrieval_rank"]:
                    dlt = piv[m][pred] - piv[m][ref]
                    mu, lo, hi, nn_ = boot_ci(dlt.values, args.nboot, rng_boot)
                    pb_rows.append(dict(heldout=L, condition=cond, predictor=pred, reference=ref, metric=m,
                                        mean_diff=mu, ci95_lo=lo, ci95_hi=hi, n_targets=nn_))
    pd.DataFrame(pb_rows).to_csv(out / "paired_bootstrap.tsv", sep="\t", index=False, float_format="%.5f")
    # stratification (condition A): effect size tertile and knockdown tertile (within held-out line)
    A = df[df.condition == "A_unseen_ctx_seen_target"].copy()
    strat = []
    for L in LINES:
        D = data[L]
        eff = pd.Series(np.linalg.norm(D["delta"], axis=1), index=D["targets"])
        x = A[A.heldout == L].copy()
        x["effect_tertile"] = pd.qcut(eff.loc[x.target].values, 3, labels=["low", "mid", "high"])
        kdv = x.kd.values
        x["kd_bin"] = np.where(np.isnan(kdv), "not_measured", np.where(kdv >= 0.7, "kd>=70%", np.where(kdv >= 0.4, "kd40-70%", "kd<40%")))
        for col in ["effect_tertile", "kd_bin"]:
            g = x.groupby([col, "predictor"], observed=True)[["rmse", "pearson", "pearson_centered", "pearson_pertcentered", "deg_auprc", "dir_top20"]].mean()
            g["n_targets"] = x.groupby([col, "predictor"], observed=True).size()
            for (lev, pred), r in g.iterrows():
                strat.append(dict(heldout=L, stratifier=col, level=lev, predictor=pred, **r.to_dict()))
    pd.DataFrame(strat).to_csv(out / "loco_strata.tsv", sep="\t", index=False, float_format="%.5f")
    pd.DataFrame(fold_info).to_csv(out / "loco_folds.tsv", sep="\t", index=False)
    (out / "loco_hyperparams.json").write_text(json.dumps(hp, indent=1))
    (out / "loco_test_splits.json").write_text(json.dumps(dict(seed=args.seed, test_frac=args.test_frac,
                                                               min_cells=args.min_cells, splits=splits), indent=0))
    meta = dict(common_genes=len(C), n_target_genes_in_C=len(all_target_genes), runtime_s=round(time.time() - t0, 1),
                args=vars(args), metrics_note="self gene excluded; retrieval over non-target readout genes, centred by source mean")
    (out / "loco_run_meta.json").write_text(json.dumps(meta, indent=1))
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
