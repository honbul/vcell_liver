#!/usr/bin/env python
"""G3 / G1-b: simple baselines on PIE's own replogle_wdataset split, scored with PIE's own metrics.

Runs in the PIE env (imports pie.metrics.score, so every metric is computed by the exact code that
produced the published numbers). Data = PIE preprocessed dir (6,642-gene axis, PIE's DE labels:
Mann-Whitney + BH via gpudge, FDR < 0.05, `tested` mask; delta_p = pseudobulk log1p - ctrl mean).

Fold F (default hepg2): visible = train.json rows (3 source lines) + val.json rows (source lines,
pairs whose perturbation is in train for another line) + every line's control means.
All selection (ridge K/lambda, magnitude calibration, logistic DE head) uses only train/val rows of
the source lines. Test rows (held-out line, or unseen perturbations) are used only for scoring.

Predictors (outputs per row and gene: p_de, lfc_pred, delta_p_pred):
  no_change          p_de=0, lfc=0, delta=0
  no_change_kd       no_change + target-gene knockdown rule (KD rule, below)
  global_mean_kd     equal-weight mean over source lines of the per-line train means
                     (lfc, delta) and DE frequency as p_de; + KD rule
  context_mean_kd    (unseen_pert only) the test line's own train mean; + KD rule
  nearest_line_mean_kd (held-out line only) mean response of the source line whose control mean
                     correlates best with the held-out line's control mean; + KD rule
  same_target_raw    (unseen_ctx) mean over source train rows of the same perturbation;
                     p_de = mean(1 - FDR) over those rows
  same_target_cal    (unseen_ctx) lfc = a*same + b*global, delta likewise (least squares on val);
                     p_de = logistic(same/global features) fit on val
  nearest_line_same_target (unseen_ctx) the same perturbation's row in the source line whose
                     control mean correlates best with the held-out line's; p_de = 1 - FDR
  ridge_stacked_kd   Ahlmann-Eltze bilinear ridge on all source train rows (gene embedding = top-K
                     PCs of the train LFC matrix, perturbation embedding = target gene's row);
                     K, lambda by perturbation-grouped 5-fold CV on train rows; separate fits for
                     lfc and delta; p_de = logistic on out-of-fold features; + KD rule
  ridge_inline_kd    (unseen_pert) same, fit per test line on that line's own train rows
KD rule: if the perturbation is a gene on the axis, its own gene gets lfc = median train target
LFC, delta = median train (delta/ctrl) ratio x the context's ctrl mean, p_de >= train target DE rate.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pie.data.preprocessed import PreprocessedDir, target_gene_index
from pie.data.splits import load_split
from pie.metrics import PAIR_METRIC_KEYS, ScoringInputs, _binary_average_precision, _spearman, score

SETTINGS = ["unseen_ctx", "unseen_pert", "unseen_ctx_pert"]
RIDGE_K = [5, 10, 25, 50, 100, 200]
RIDGE_LAM = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
FDR = 0.05
BOOT_KEYS = ["binary_auprc", "sig_jaccard", "direction_match", "spearman_lfc",
             "discrimination_score_l1"]
EXTRA_KEYS = ["auprc_x", "direction_x", "spearman_lfc_x", "target_pde_pct", "dp_rmse", "dp_pearson",
              "dp_pertc_r"]


# ------------------------------------------------------------------ helpers
def logit_fit(X, y, l2=1e-4, iters=30):
    """Newton/IRLS logistic regression on standardised features; returns predict(X) -> prob."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break

    def predict(Xn):
        Zn = np.c_[np.ones(len(Xn)), (Xn - mu) / sd]
        return (1 / (1 + np.exp(-np.clip(Zn @ w, -30, 30)))).astype(np.float32)

    predict.coef = dict(intercept=float(w[0]), w=[float(v) for v in w[1:]],
                        mu=[float(v) for v in mu], sd=[float(v) for v in sd])
    return predict


def ridge_svd(Y):
    b = Y.mean(1)
    Yc = Y - b[:, None]
    U, s, _ = np.linalg.svd(Yc, full_matrices=False)
    return b, U, s, Yc


def ridge_fit(svd, rows, K, lam):
    """Same estimator as src/feedback/f3_loco.py fit_ridge (Ahlmann-Eltze 2025 style)."""
    b, U, s, Yc = svd
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
    return float(np.sqrt((((P - Y) ** 2) * m).sum(1) / m.sum(1)).mean())


class Data:
    def __init__(self, path):
        d = PreprocessedDir.open(Path(path))
        self.d = d
        self.genes = list(d.genes)
        self.gidx = {g: i for i, g in enumerate(self.genes)}
        self.ctxs = d.contexts
        self.lfc = np.asarray(d.lfc_true, dtype=np.float64)
        self.fdr = np.asarray(d.fdr, dtype=np.float32)
        self.tested = np.asarray(d.tested, dtype=bool)
        self.dp = np.asarray(d.delta_p, dtype=np.float32)
        self.ctrl = np.asarray(d.ctrl_means, dtype=np.float32)
        self.ctx_ids = np.asarray(d.ctx_ids, dtype=np.int64)
        self.keys = d.row_keys()
        self.index = d.row_index()
        self.de = (self.fdr < FDR) & self.tested & ~np.isnan(self.lfc)
        self.lfc0 = np.nan_to_num(self.lfc, nan=0.0)
        self.target = np.array([target_gene_index(p, self.gidx) for _, p in self.keys])
        self.ctx_of = {c: self.ctxs.index(c) for c in self.ctxs}

    def rows(self, split):
        out = []
        for key, perts in split.items():
            ctx = key.split(".", 1)[1]
            out += [self.index[(ctx, p)] for p in perts]
        return np.array(sorted(out), dtype=np.int64)

    def ctrl_of_rows(self, rows):
        return self.ctrl[self.ctx_ids[rows]]


def line_means(D, rows):
    """Per source line: mean lfc0, mean delta, DE frequency over its rows."""
    out = {}
    for c in sorted(set(D.ctx_ids[rows].tolist())):
        r = rows[D.ctx_ids[rows] == c]
        out[c] = dict(lfc=D.lfc0[r].mean(0), dp=D.dp[r].mean(0), defreq=D.de[r].mean(0),
                      n=len(r))
    return out


def eq_mean(lm, key, lines):
    return np.mean([lm[c][key] for c in lines], axis=0)


def kd_stats(D, rows):
    r = rows[D.target[rows] >= 0]
    t = D.target[r]
    tested = D.tested[r, t]
    ctrl = D.ctrl[D.ctx_ids[r], t]
    ok = tested & (ctrl > 0)
    return dict(lfc=float(np.median(D.lfc[r, t][tested])),
                ratio=float(np.median(D.dp[r, t][ok] / ctrl[ok])),
                de=float(D.de[r, t].mean()), n=int(len(r)))


def apply_kd(pred, D, rows, kd):
    p, l, dl = pred
    for i, r in enumerate(rows):
        t = D.target[r]
        if t >= 0:
            l[i, t] = kd["lfc"]
            dl[i, t] = kd["ratio"] * D.ctrl[D.ctx_ids[r], t]
            p[i, t] = max(p[i, t], kd["de"])
    return p, l, dl


def nearest_order(D, held_ctx, lines):
    ref = D.ctrl[held_ctx]
    sims = {c: float(np.corrcoef(D.ctrl[c], ref)[0, 1]) for c in lines}
    return sorted(lines, key=lambda c: -sims[c]), sims


def same_target_stats(D, train_rows, perts, exclude_ctx=None):
    """Per perturbation: mean lfc0 / delta / (1-FDR) / DE indicator over its train rows."""
    by = {}
    for r in train_rows:
        c, p = D.keys[r]
        if exclude_ctx is not None and D.ctx_of[c] == exclude_ctx:
            continue
        by.setdefault(p, []).append(r)
    G = len(D.genes)
    out = {k: np.zeros((len(perts), G), np.float32) for k in ("lfc", "dp", "conf", "de")}
    n = np.zeros(len(perts), np.int64)
    for i, p in enumerate(perts):
        rr = by.get(p, [])
        n[i] = len(rr)
        if rr:
            rr = np.array(rr)
            lf = D.lfc[rr]
            cnt = (~np.isnan(lf)).sum(0)
            out["lfc"][i] = np.where(cnt > 0, np.nansum(lf, 0) / np.maximum(cnt, 1), 0.0)
            out["dp"][i] = D.dp[rr].mean(0)
            out["conf"][i] = (1 - D.fdr[rr]).mean(0)
            out["de"][i] = D.de[rr].mean(0)
    return out, n


def mask_eval(D, rows):
    """Genes used for magnitude fits: tested, finite lfc, not the target gene."""
    m = D.tested[rows] & ~np.isnan(D.lfc[rows])
    t = D.target[rows]
    ok = t >= 0
    m[np.where(ok)[0], t[ok]] = False
    return m


# ------------------------------------------------------------------ ridge with grouped CV
def ridge_select(D, rows, seed, nfold=5, log=None):
    """K/lambda by perturbation-grouped CV on `rows` (all rows of a perturbation share a fold).
    Criterion: lfc0 RMSE excluding the target gene. Returns best, OOF predictions (lfc, dp)."""
    rows = rows[D.target[rows] >= 0]
    perts = sorted({D.keys[r][1] for r in rows})
    rng = np.random.default_rng(seed)
    pf = dict(zip(perts, rng.integers(0, nfold, len(perts))))
    fold = np.array([pf[D.keys[r][1]] for r in rows])
    res = {(K, lam): [] for K in RIDGE_K for lam in RIDGE_LAM}
    for f in range(nfold):
        tr, te = rows[fold != f], rows[fold == f]
        svd = ridge_svd(D.lfc0[tr].T)
        for K, lam in res:
            m = ridge_fit(svd, D.target[tr], K, lam)
            res[(K, lam)].append(rmse_excl(ridge_predict(m, D.target[te]), D.lfc0[te], D.target[te]))
    res = {k: float(np.mean(v)) for k, v in res.items()}
    best = min(res, key=res.get)
    oof_l = np.zeros((len(rows), len(D.genes)), np.float32)
    oof_d = np.zeros_like(oof_l)
    for f in range(nfold):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        ml = ridge_fit(ridge_svd(D.lfc0[rows[tr]].T), D.target[rows[tr]], *best)
        md = ridge_fit(ridge_svd(D.dp[rows[tr]].T.astype(np.float64)), D.target[rows[tr]], *best)
        oof_l[te] = ridge_predict(ml, D.target[rows[te]])
        oof_d[te] = ridge_predict(md, D.target[rows[te]])
    if log is not None:
        log.update(best_K=best[0], best_lam=best[1], cv_rmse=res[best],
                   cv_rmse_worst=max(res.values()), n_rows=int(len(rows)), n_perts=len(perts))
    return best, rows, oof_l, oof_d


def ridge_final(D, rows, best):
    rows = rows[D.target[rows] >= 0]
    ml = ridge_fit(ridge_svd(D.lfc0[rows].T), D.target[rows], *best)
    md = ridge_fit(ridge_svd(D.dp[rows].T.astype(np.float64)), D.target[rows], *best)
    return ml, md


def sample_mask(m, nmax, rng):
    idx = np.flatnonzero(m.ravel())
    if len(idx) > nmax:
        idx = np.sort(rng.choice(idx, nmax, replace=False))
    return idx


def ridge_feats(l, d, defreq):
    return np.stack([np.abs(l).ravel(), np.abs(d).ravel(), np.broadcast_to(defreq, l.shape).ravel()], 1)


# ------------------------------------------------------------------ scoring
def scoring_inputs(D, rows, pred):
    p, l, dl = pred
    return ScoringInputs(
        dataset="replogle", genes=D.genes,
        contexts=[D.keys[r][0] for r in rows], perts=[D.keys[r][1] for r in rows],
        p_de=np.asarray(p, np.float32), lfc_pred=np.asarray(l, np.float32),
        delta_p_pred=np.asarray(dl, np.float32),
        de_true=(D.fdr[rows] < FDR) & D.tested[rows], tested=D.tested[rows],
        lfc_true=D.lfc[rows], delta_p_true=D.dp[rows], ctrl_means=D.ctrl_of_rows(rows))


def extra_metrics(D, rows, pred):
    """Target-gene-excluded versions of PIE's row metrics (same definitions as pie.metrics:
    AUPRC over tested genes, direction over true DE genes, Spearman over true DE genes, but the
    perturbed gene removed), the percentile rank of the target gene's p_de among tested genes,
    and magnitude-aware metrics on PIE's delta (log1p) scale (target excluded): RMSE, Pearson,
    perturbation-centred Pearson (centroids over this condition's rows, per context)."""
    prob_all = np.nan_to_num(np.asarray(pred[0], np.float32), nan=0.0)
    lpred = np.asarray(pred[1], np.float32)
    prob_all[np.isnan(lpred)] = 0.0
    Y = D.dp[rows].astype(np.float64)
    P = np.asarray(pred[2], np.float64)
    ctx = D.ctx_ids[rows]
    Yc, Pc = Y.copy(), P.copy()
    for c in np.unique(ctx):
        m = ctx == c
        Yc[m] -= Y[m].mean(0)
        Pc[m] -= P[m].mean(0)
    out = []
    for i, r in enumerate(rows):
        keep = np.ones(Y.shape[1], bool)
        if D.target[r] >= 0:
            keep[D.target[r]] = False
        y, p, yc, pc = Y[i, keep], P[i, keep], Yc[i, keep], Pc[i, keep]
        t = D.target[r]
        tested, de, lt = D.tested[r], D.de[r], D.lfc[r]
        kt = tested & keep
        labels = de[kt]
        auprc_x = 0.0
        if labels.any() and not labels.all():
            auprc_x = _binary_average_precision(prob_all[i][kt], labels)
        dex = de & keep
        n_de = int(dex.sum())
        dir_x = 0.0
        if n_de:
            pl = lpred[i][dex]
            dir_x = float(np.mean(~np.isnan(pl) & (np.sign(lt[dex]) == np.sign(pl))))
        tgt_pct = math.nan
        if t >= 0 and tested[t]:
            others = prob_all[i][tested & keep]
            tgt_pct = float((others < prob_all[i][t]).mean() + 0.5 * (others == prob_all[i][t]).mean())
        out.append(dict(
            auprc_x=auprc_x, direction_x=dir_x, spearman_lfc_x=_spearman(lt[dex], lpred[i][dex]),
            target_is_de=float(de[t]) if t >= 0 else math.nan, target_pde_pct=tgt_pct,
            target_lfc_pred=float(lpred[i][t]) if t >= 0 else math.nan,
            dp_rmse=float(np.sqrt(np.mean((p - y) ** 2))),
            dp_pearson=float(np.corrcoef(p, y)[0, 1]) if p.std() > 1e-12 else math.nan,
            dp_pertc_r=float(np.corrcoef(pc, yc)[0, 1]) if pc.std() > 1e-12 else math.nan))
    return pd.DataFrame(out)


def context_mean_of_pairs(df, keys):
    """PIE aggregation: per-context mean over pairs (NaN skipped), then mean over contexts."""
    g = df.groupby("context")[keys].mean()
    return g.mean(axis=0)


def paired_boot(a, b, keys, nboot, rng):
    """a, b: per-pair frames (context, perturbation, keys) for the same pairs. Cluster bootstrap
    over perturbations; statistic = PIE-style context-mean of pair means; returns rows of
    (metric, mean_a, mean_b, diff, lo, hi, p_two_sided)."""
    m = a.merge(b, on=["context", "perturbation"], suffixes=("_a", "_b"))
    perts = m["perturbation"].unique()
    pidx = {p: i for i, p in enumerate(perts)}
    pi = m["perturbation"].map(pidx).to_numpy()
    ctx = m["context"].to_numpy()
    cs = np.unique(ctx)
    out = []
    draws = rng.integers(0, len(perts), (nboot, len(perts)))
    for k in keys:
        va, vb = m[f"{k}_a"].to_numpy(float), m[f"{k}_b"].to_numpy(float)
        ok = ~np.isnan(va) & ~np.isnan(vb)

        def stat(weights):
            res_a, res_b = [], []
            for c in cs:
                sel = (ctx == c) & ok
                w = weights[pi[sel]]
                if w.sum() == 0:
                    continue
                res_a.append((w * va[sel]).sum() / w.sum())
                res_b.append((w * vb[sel]).sum() / w.sum())
            return np.mean(res_a), np.mean(res_b)

        sa, sb = stat(np.ones(len(perts)))
        diffs = np.empty(nboot)
        for i in range(nboot):
            w = np.bincount(draws[i], minlength=len(perts)).astype(float)
            x, y = stat(w)
            diffs[i] = x - y
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        out.append(dict(metric=k, mean_a=sa, mean_b=sb, diff=sa - sb, lo=lo, hi=hi, p_boot=min(p, 1.0),
                        n_pairs=int(ok.sum()), n_perts=len(perts)))
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="PIE preprocessed dir")
    ap.add_argument("--splits", required=True, help="dir with <setting>/<fold>/{train,val,test}.json")
    ap.add_argument("--fold", default="hepg2")
    ap.add_argument("--pub", required=True, help="published eval dir of the fold (eval/<setting>/)")
    ap.add_argument("--repro", default=None, help="our pie eval dir of the fold (optional)")
    ap.add_argument("--out", required=True, help="report dir (small tables)")
    ap.add_argument("--assets", required=True, help="assets dir (predictions)")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--max-logit", type=int, default=3_000_000)
    args = ap.parse_args()
    t0 = time.time()
    out, adir = Path(args.out), Path(args.assets)
    out.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    D = Data(args.pre)
    sp = Path(args.splits)
    tr = D.rows(load_split(sp / "unseen_ctx" / args.fold / "train.json"))
    va = D.rows(load_split(sp / "unseen_ctx" / args.fold / "val.json"))
    for s in SETTINGS:  # same train/val for all settings (model card); verify
        assert np.array_equal(tr, D.rows(load_split(sp / s / args.fold / "train.json")))
        assert np.array_equal(va, D.rows(load_split(sp / s / args.fold / "val.json")))
    tests = {s: D.rows(load_split(sp / s / args.fold / "test.json")) for s in SETTINGS}
    held = D.ctx_of[args.fold]
    src_lines = sorted(set(D.ctx_ids[tr].tolist()))
    assert held not in src_lines and not (set(D.ctx_ids[tr]) | set(D.ctx_ids[va])) & {held}
    G = len(D.genes)
    sel = dict(fold=args.fold, source_lines=[D.ctxs[c] for c in src_lines],
               n_train=int(len(tr)), n_val=int(len(va)),
               n_test={s: int(len(v)) for s, v in tests.items()})
    lm = line_means(D, tr)
    kd = kd_stats(D, tr)
    sel["kd_rule"] = kd
    order, sims = nearest_order(D, held, src_lines)
    sel["nearest_line_order"] = [D.ctxs[c] for c in order]
    sel["ctrl_pearson_to_heldout"] = {D.ctxs[c]: v for c, v in sims.items()}
    glob = {k: eq_mean(lm, k, src_lines) for k in ("lfc", "dp", "defreq")}

    def const_pred(n, p, l, d):
        return (np.tile(p, (n, 1)).astype(np.float32), np.tile(l, (n, 1)).astype(np.float32),
                np.tile(d, (n, 1)).astype(np.float32))

    preds = {s: {} for s in SETTINGS}
    # ---------------- no_change (+KD), global, nearest-line mean
    for s, rows in tests.items():
        n = len(rows)
        z = np.zeros(G, np.float32)
        preds[s]["no_change"] = const_pred(n, z, z, z)
        preds[s]["no_change_kd"] = apply_kd(const_pred(n, z, z, z), D, rows, kd)
        preds[s]["global_mean"] = const_pred(n, glob["defreq"], glob["lfc"], glob["dp"])
        preds[s]["global_mean_kd"] = apply_kd(
            const_pred(n, glob["defreq"], glob["lfc"], glob["dp"]), D, rows, kd)
    for s in ("unseen_ctx", "unseen_ctx_pert"):
        rows = tests[s]
        c0 = order[0]
        preds[s]["nearest_line_mean_kd"] = apply_kd(
            const_pred(len(rows), lm[c0]["defreq"], lm[c0]["lfc"], lm[c0]["dp"]), D, rows, kd)
    rows = tests["unseen_pert"]
    cp = np.zeros((len(rows), G), np.float32)
    cl, cd = cp.copy(), cp.copy()
    for i, r in enumerate(rows):
        c = D.ctx_ids[r]
        cp[i], cl[i], cd[i] = lm[c]["defreq"], lm[c]["lfc"], lm[c]["dp"]
    preds["unseen_pert"]["context_mean"] = (cp.copy(), cl.copy(), cd.copy())
    preds["unseen_pert"]["context_mean_kd"] = apply_kd((cp, cl, cd), D, rows, kd)

    # ---------------- same-target family (unseen_ctx only)
    rows = tests["unseen_ctx"]
    perts = [D.keys[r][1] for r in rows]
    st, nst = same_target_stats(D, tr, perts)
    assert (nst > 0).all(), "unseen_ctx test perturbation without a source train row"
    sel["same_target_n_source_rows"] = {str(k): int(v) for k, v in zip(*np.unique(nst, return_counts=True))}
    preds["unseen_ctx"]["same_target_raw"] = (st["conf"].copy(), st["lfc"].copy(), st["dp"].copy())
    # nearest line, same target
    by_line = {c: {} for c in src_lines}
    for r in tr:
        c, p = D.keys[r]
        by_line[D.ctx_of[c]][p] = r
    nl = [np.zeros((len(rows), G), np.float32) for _ in range(3)]
    used = []
    for i, p in enumerate(perts):
        for c in order:
            if p in by_line[c]:
                r = by_line[c][p]
                nl[0][i], nl[1][i], nl[2][i] = 1 - D.fdr[r], D.lfc0[r], D.dp[r]
                used.append(D.ctxs[c])
                break
    sel["nearest_line_same_target_used"] = pd.Series(used).value_counts().to_dict()
    preds["unseen_ctx"]["nearest_line_same_target"] = tuple(nl)
    # calibration on val rows (source lines only; features from the other lines' train rows)
    vperts = [D.keys[r][1] for r in va]
    vst, vn = same_target_stats(D, tr, vperts)
    keep = vn > 0
    vrows = va[keep]
    vctx = D.ctx_ids[vrows]
    gl_other = {k: np.stack([eq_mean(lm, k, [c for c in src_lines if c != v]) for v in vctx])
                for k in ("lfc", "dp", "defreq")}
    m = mask_eval(D, vrows)
    coefs = {}
    for key, Y in (("lfc", D.lfc0[vrows]), ("dp", D.dp[vrows].astype(np.float64))):
        X = np.stack([vst[key][keep][m], gl_other[key][m]], 1)
        coefs[key] = [float(c) for c in np.linalg.lstsq(X, Y[m], rcond=None)[0]]
    sel["same_target_cal_coef"] = coefs
    sel["same_target_cal_n_val_rows"] = int(len(vrows))

    def st_feats(stt, sel_rows_idx, gdef, glfc):
        return np.stack([stt["conf"][sel_rows_idx].ravel(), stt["de"][sel_rows_idx].ravel(),
                         np.abs(stt["lfc"][sel_rows_idx]).ravel(), np.abs(stt["dp"][sel_rows_idx]).ravel(),
                         np.broadcast_to(gdef, stt["lfc"][sel_rows_idx].shape).ravel(),
                         np.abs(np.broadcast_to(glfc, stt["lfc"][sel_rows_idx].shape)).ravel()], 1)

    vidx = np.where(keep)[0]
    Xv = st_feats(vst, vidx, gl_other["defreq"], gl_other["lfc"])
    tv = D.tested[vrows].ravel()
    yv = D.de[vrows].ravel()
    pick = sample_mask(tv.reshape(len(vrows), G), args.max_logit, rng)
    head = logit_fit(Xv[pick], yv[pick].astype(float))
    sel["same_target_cal_logit"] = head.coef
    Xt = st_feats(st, np.arange(len(rows)), glob["defreq"], glob["lfc"])
    pcal = head(Xt).reshape(len(rows), G)
    lcal = coefs["lfc"][0] * st["lfc"] + coefs["lfc"][1] * glob["lfc"]
    dcal = coefs["dp"][0] * st["dp"] + coefs["dp"][1] * glob["dp"]
    preds["unseen_ctx"]["same_target_cal"] = (pcal, lcal.astype(np.float32), dcal.astype(np.float32))

    # ---------------- ridge (stacked over source lines): unseen_ctx, unseen_ctx_pert, unseen_pert
    rlog = {}
    best, rrows, oof_l, oof_d = ridge_select(D, tr, args.seed, log=rlog)
    ml, md = ridge_final(D, tr, best)
    gdef_rows = np.stack([lm[D.ctx_ids[r]]["defreq"] for r in rrows])
    Xo = ridge_feats(oof_l, oof_d, gdef_rows)
    pick = sample_mask(D.tested[rrows], args.max_logit, rng)
    rhead = logit_fit(Xo[pick], D.de[rrows].ravel()[pick].astype(float))
    rlog["logit"] = rhead.coef
    sel["ridge_stacked"] = rlog
    for s, rows in tests.items():
        t = D.target[rows]
        l = ridge_predict(ml, t).astype(np.float32)
        d = ridge_predict(md, t).astype(np.float32)
        defr = (glob["defreq"] if s != "unseen_pert"
                else np.stack([lm[D.ctx_ids[r]]["defreq"] for r in rows]))
        p = rhead(ridge_feats(l, d, defr)).reshape(len(rows), G)
        preds[s]["ridge_stacked_kd"] = apply_kd((p, l, d), D, rows, kd)
    # ---------------- ridge in-line (unseen_pert): per test line, that line's own train rows
    rows = tests["unseen_pert"]
    P = [np.zeros((len(rows), G), np.float32) for _ in range(3)]
    sel["ridge_inline"] = {}
    for c in src_lines:
        lrows = tr[D.ctx_ids[tr] == c]
        lg = {}
        b2, rr2, ol, od = ridge_select(D, lrows, args.seed + c, log=lg)
        h2 = logit_fit(*(lambda X, pk: (X[pk], D.de[rr2].ravel()[pk].astype(float)))(
            ridge_feats(ol, od, lm[c]["defreq"]), sample_mask(D.tested[rr2], args.max_logit, rng)))
        lg["logit"] = h2.coef
        sel["ridge_inline"][D.ctxs[c]] = lg
        m1, m2 = ridge_final(D, lrows, b2)
        sel_i = np.where(D.ctx_ids[rows] == c)[0]
        t = D.target[rows[sel_i]]
        l = ridge_predict(m1, t).astype(np.float32)
        d = ridge_predict(m2, t).astype(np.float32)
        P[0][sel_i] = h2(ridge_feats(l, d, lm[c]["defreq"])).reshape(len(sel_i), G)
        P[1][sel_i], P[2][sel_i] = l, d
    preds["unseen_pert"]["ridge_inline_kd"] = apply_kd(tuple(P), D, rows, kd)
    sel["t_fit_s"] = round(time.time() - t0, 1)
    print(f"[g3] fitted baselines in {sel['t_fit_s']} s", flush=True)

    # ---------------- PIE per-pair (published; reproduced if given)
    def read_gran(base, s):
        f = Path(base) / s / "granular_best_auprc.csv"
        return pd.read_csv(f) if f.is_file() else None

    def read_metrics(base, s):
        f = Path(base) / s / "metrics_best_auprc.csv"
        return pd.read_csv(f) if f.is_file() else None

    summary, pairs_all, boots, extras = [], [], [], []
    for s, rows in tests.items():
        pie_pub = read_gran(args.pub, s)
        pie_rep = read_gran(args.repro, s) if args.repro else None
        pie_pred = None
        if args.repro:
            pq_path = Path(args.repro) / s / "predictions_best_auprc.parquet"
            if pq_path.is_file():
                tbl = pd.read_parquet(pq_path)
                key = {(c, p): i for i, (c, p) in enumerate(zip(tbl["context"], tbl["perturbation"]))}
                order_i = [key[D.keys[r]] for r in rows]
                pie_pred = tuple(np.stack(tbl[col].to_numpy()[order_i]).astype(np.float32)
                                 for col in ("p_de", "lfc_pred", "delta_p_pred"))
                preds[s]["PIE_repro"] = pie_pred
                # diagnostic only: PIE + the same target-gene rule as the *_kd baselines
                preds[s]["PIE_repro_kd"] = apply_kd(tuple(a.copy() for a in pie_pred), D, rows, kd)
        for name, pred in preds[s].items():
            res = score([scoring_inputs(D, rows, pred)])
            pp = res.per_pair.drop(columns=["dataset"]).assign(setting=s, predictor=name)
            ex = extra_metrics(D, rows, pred)
            ex = ex.assign(context=[D.keys[r][0] for r in rows], perturbation=[D.keys[r][1] for r in rows])
            pp = pp.merge(ex, on=["context", "perturbation"])
            pairs_all.append(pp)
            row = dict(setting=s, predictor=name, n_rows=len(rows), **res.context_mean)
            row.update(context_mean_of_pairs(pp, EXTRA_KEYS).to_dict())
            summary.append(row)
        for tag, g in (("PIE_published", pie_pub), ("PIE_repro_csv", pie_rep)):
            if g is None:
                continue
            g = g.assign(setting=s, predictor=tag)
            pairs_all.append(g)
            mt = read_metrics(args.pub if tag == "PIE_published" else args.repro, s)
            allrow = mt[mt["context"] == "all"].iloc[0].to_dict()
            summary.append(dict(setting=s, predictor=tag, n_rows=int(allrow["n_rows"]),
                                **{k: allrow[k] for k in mt.columns if k not in ("context", "n_rows")}))
        if pie_pub is not None:
            base = pie_pub[["context", "perturbation", *PAIR_METRIC_KEYS]]
            for pp in [x for x in pairs_all if x["setting"].iloc[0] == s and
                       x["predictor"].iloc[0] in preds[s]]:
                name = pp["predictor"].iloc[0]
                for b in paired_boot(base, pp[["context", "perturbation", *PAIR_METRIC_KEYS]],
                                     BOOT_KEYS, args.nboot, rng):
                    boots.append(dict(setting=s, a="PIE_published", b=name, **b))
        if pie_pred is not None:
            pa = [x for x in pairs_all if x["setting"].iloc[0] == s and x["predictor"].iloc[0] == "PIE_repro"][0]
            for pp in [x for x in pairs_all if x["setting"].iloc[0] == s and
                       x["predictor"].iloc[0] in preds[s] and x["predictor"].iloc[0] != "PIE_repro"]:
                name = pp["predictor"].iloc[0]
                for b in paired_boot(pa[["context", "perturbation", *EXTRA_KEYS]],
                                     pp[["context", "perturbation", *EXTRA_KEYS]],
                                     EXTRA_KEYS, args.nboot, rng):
                    boots.append(dict(setting=s, a="PIE_repro", b=name, **b))
        # save held-out-line predictions for the F3-metric script
        if s != "unseen_pert":
            np.savez_compressed(adir / f"pred_{args.fold}_{s}.npz",
                                contexts=np.array([D.keys[r][0] for r in rows]),
                                perts=np.array([D.keys[r][1] for r in rows]),
                                genes=np.array(D.genes),
                                **{f"{n}__delta": v[2] for n, v in preds[s].items()},
                                **{f"{n}__lfc": v[1] for n, v in preds[s].items()},
                                **{f"{n}__pde": v[0] for n, v in preds[s].items()},
                                truth_delta=D.dp[rows], ctrl=D.ctrl[held])
    summ = pd.DataFrame(summary)
    summ.to_csv(out / f"pie_split_summary_{args.fold}.tsv", sep="\t", index=False, float_format="%.5f")
    pd.concat(pairs_all, ignore_index=True).to_csv(
        adir / f"pie_split_per_pair_{args.fold}.tsv.gz", sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(boots).to_csv(out / f"pie_split_paired_bootstrap_{args.fold}.tsv", sep="\t",
                               index=False, float_format="%.5f")
    # stratification by true DE count (per pair) for the headline metric
    strat = []
    allp = pd.concat(pairs_all, ignore_index=True)
    for s, rows in tests.items():
        nde = pd.DataFrame(dict(context=[D.keys[r][0] for r in rows], perturbation=[D.keys[r][1] for r in rows],
                                n_true_de=D.de[rows].sum(1)))
        q = np.quantile(nde["n_true_de"], [1 / 3, 2 / 3])
        nde["de_tertile"] = np.digitize(nde["n_true_de"], q, right=True)
        sub = allp[allp["setting"] == s].merge(nde, on=["context", "perturbation"])
        g = sub.groupby(["predictor", "de_tertile"])[["binary_auprc", "direction_match", "spearman_lfc",
                                                      "auprc_x", "direction_x"]].mean()
        g = g.reset_index().assign(setting=s, tertile_edges=f"{q[0]:.0f}/{q[1]:.0f}")
        g["n_pairs"] = sub.groupby(["predictor", "de_tertile"]).size().values
        strat.append(g)
    pd.concat(strat).to_csv(out / f"pie_split_strata_{args.fold}.tsv", sep="\t", index=False,
                            float_format="%.5f")
    sel["t_total_s"] = round(time.time() - t0, 1)
    (out / f"pie_split_selection_{args.fold}.json").write_text(json.dumps(sel, indent=1, default=str))
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summ.to_string())
    print(f"[g3] done in {sel['t_total_s']} s")


if __name__ == "__main__":
    main()
