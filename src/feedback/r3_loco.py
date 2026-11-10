"""R3: chemical leave-one-cell-line-out (LOCO) baselines with SNU-423 held out (Tahoe-100M pseudobulk).

Input: assets/r3/prep (r3_prep.py): Delta[line, well, gene] = log1p(mean CP10k) of a (well, line) minus that
line's plate-matched DMSO pool. Test line t = SNU-423 (CVCL_0366). All SNU-423 treated wells are held out;
at test time only SNU-423 DMSO (profile + plate DMSO) is visible.

Training sets: tr49 = the other 49 lines; tr48 = also without HepG2/C3A (liver-proxy sensitivity).
Gene panel (per training set): top-2,000 genes by variance of training-line Delta over training-drug wells.

Conditions (evaluation units)
  aW  unseen line, seen drug x dose, Tahoe-LOCO information set: training-line Delta from every well,
      including the test well itself. Because 975/1,137 drug x doses sit in exactly one well, every
      "same drug x dose" predictor here shares the test well (and its plate DMSO) with the truth -> class W.
      Unit = drug x dose (truth = mean over SNU-423 wells with >= 50 cells).
  aP  unseen line, seen drug (x dose if replicated on another plate), plate-disjoint information set:
      a test well on plate p is predicted only from training-line wells on plates != p. Unit = SNU-423 well.
  b   unseen line + unseen drug: random 20% of drugs (all doses) removed from every line (seed fixed,
      no data used). Unit = drug x dose of a test drug.
  c   within-SNU-423 unseen drug: same test drugs, trained on SNU-423's own training-drug conditions.

Information classes: none / train (training lines, no test-well data) / plate (shares the test well's plate
DMSO reference but not the well; diagnostic) / W (uses training-line cells from the test well; oracle-like) /
snu_treated (uses other SNU-423 treated wells; aP noise-ceiling reference only).

Metrics on the panel: RMSE, Pearson, perturbation-centred Pearson (subtract mean truth / mean prediction over
the evaluated units), retrieval rank (0 best, 0.5 chance; centred profiles), DEG AUPRC / precision@k / sign
agreement on DEGs, where SNU-423 DEGs are defined from the DMSO well-vs-well null (see deg_null()).
Strata: F4 effect-strength flags of the SNU-423 well(s) (detectable = R > max DMSO-pair R; strong = 2x).
Bootstrap: drug-clustered paired bootstrap (2,000) vs no_change and vs the best class none/train predictor.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
TEST = "CVCL_0366"
HEPG2 = "CVCL_1098"
SEED = 20261110


# ---------------------------------------------------------------- helpers
def nanmean0(x, axis):
    return np.nanmean(x, axis=axis)


def set_line_mean(Dp, lines_idx, wells_idx):
    """per-line mean over a set of wells, then mean over lines (NaN-aware). returns G-vector (NaN if empty)."""
    if len(wells_idx) == 0:
        return None
    x = Dp[np.ix_(lines_idx, wells_idx)]
    v = np.nanmean(np.nanmean(x, axis=1), axis=0)
    return v if np.isfinite(v[0]) else None


def tanimoto(A, B):
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-12)


def morgan(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out, ok = np.zeros((len(smiles), 2048), np.float32), np.zeros(len(smiles), bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s)) if isinstance(s, str) else None
        if m is not None:
            out[i] = gen.GetFingerprintAsNumPy(m)
            ok[i] = True
    return out, ok


# ---------------------------------------------------------------- DEG truth
def deg_null(dmso_E, dmso_n, dmso_wells, prof_t, q=0.999, nbins=40):
    """SNU-423 DEG null from same-plate DMSO well pairs (sampling + well effects, as in F4).
    u = (E_a - E_b)/sqrt(1/n_a + 1/n_b) per gene; genes binned by SNU-423 DMSO expression; tau_bin = q-quantile
    of |u| in the bin. A treated unit with cell-count factor h = sum_w(1/n_w + 1/n_pool_w)/k^2 calls gene g a DEG
    if |Delta_g| > tau_bin(g) * sqrt(h). Leave-one-pair-out false-call rate is returned as a calibration check."""
    pairs = []
    for p, g in dmso_wells.groupby("plate"):
        if len(g) == 2:
            a, b = g.index
            h = 1 / dmso_n[a] + 1 / dmso_n[b]
            pairs.append((dmso_E[a] - dmso_E[b]) / np.sqrt(h))
    U = np.abs(np.vstack(pairs))  # [n_pairs, G]
    edges = np.quantile(prof_t, np.linspace(0, 1, nbins + 1))
    gbin = np.clip(np.searchsorted(edges, prof_t, side="right") - 1, 0, nbins - 1)
    tau = np.zeros(nbins)
    for b in range(nbins):
        tau[b] = np.quantile(U[:, gbin == b].ravel(), q)
    # leave-one-pair-out calibration
    fcr = []
    for i in range(len(U)):
        rest = np.delete(U, i, axis=0)
        t_i = np.array([np.quantile(rest[:, gbin == b].ravel(), q) for b in range(nbins)])
        fcr.append(float((U[i] > t_i[gbin]).mean()))
    return tau[gbin], dict(q=q, nbins=nbins, n_pairs=len(U), loo_false_call_rate_mean=float(np.mean(fcr)),
                           loo_false_call_rate_max=float(np.max(fcr)))


# ---------------------------------------------------------------- metrics
def unit_metrics(P, Y, deg, grp):
    """P, Y: [n, G]; deg: bool [n, G]; grp: plate-group label per unit. Per-unit metrics; centring is set-level
    (pert-centred) or within plate group (plate-centred: removes the shared plate-DMSO-reference component)."""
    n = len(Y)
    Pm, Ym = P.mean(0), Y.mean(0)
    grp = pd.Series(np.asarray(grp))
    Pg, Yg = np.zeros_like(P), np.zeros_like(Y)
    for _, ix in grp.groupby(grp).groups.items():
        ix = np.asarray(ix)
        Pg[ix] = P[ix].mean(0); Yg[ix] = Y[ix].mean(0)
    gsize = grp.map(grp.value_counts()).to_numpy()
    rows = []
    for i in range(n):
        p, y = P[i], Y[i]
        r = dict(rmse=float(np.sqrt(np.mean((p - y) ** 2))))
        r["pearson"] = float(np.corrcoef(p, y)[0, 1]) if p.std() > 1e-12 else np.nan
        pp, yp = p - Pm, y - Ym
        r["pearson_pertcentered"] = float(np.corrcoef(pp, yp)[0, 1]) if pp.std() > 1e-12 else np.nan
        pq_, yq_ = p - Pg[i], y - Yg[i]
        r["pearson_platecentered"] = float(np.corrcoef(pq_, yq_)[0, 1]) if (pq_.std() > 1e-12 and gsize[i] >= 5) else np.nan
        dg = deg[i]
        nd = int(dg.sum())
        r["n_deg"] = nd
        if nd >= 5 and nd < len(y) and p.std() > 1e-12:
            r["deg_auprc"] = float(average_precision_score(dg, np.abs(p)))
            r["deg_prevalence"] = nd / len(y)
            top = np.argsort(-np.abs(p), kind="stable")[:nd]
            r["deg_prec_at_k"] = float(dg[top].mean())
            r["deg_sign_agree"] = float(np.mean(np.sign(p[dg]) == np.sign(y[dg])))
        else:
            r["deg_auprc"] = r["deg_prevalence"] = r["deg_prec_at_k"] = r["deg_sign_agree"] = np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df["retrieval_rank"] = retrieval_rank(P, Y)
    return df


def retrieval_rank(P, Y):
    """normalised rank (0 best, 0.5 chance) of each unit's own prediction among all predictions of the set,
    Pearson on set-centred profiles."""
    if len(Y) < 2:
        return np.full(len(Y), np.nan)
    Yc, Pc = Y - Y.mean(0), P - P.mean(0)
    Yz = Yc - Yc.mean(1, keepdims=True); Yz /= np.linalg.norm(Yz, axis=1, keepdims=True) + 1e-12
    Pz = Pc - Pc.mean(1, keepdims=True); nrm = np.linalg.norm(Pz, axis=1, keepdims=True)
    if np.all(nrm < 1e-12):
        return np.full(len(Y), np.nan)
    Pz = np.where(nrm > 1e-12, Pz / np.maximum(nrm, 1e-12), 0.0)
    C = Yz @ Pz.T
    ranks = np.array([stats.rankdata(-C[i])[i] for i in range(len(C))])
    return (ranks - 1) / (len(C) - 1)


# ---------------------------------------------------------------- simple fitting helpers
def lsq2(acc):
    XtX, Xty = acc
    return np.linalg.solve(XtX + 1e-9 * np.eye(len(XtX)), Xty)


def acc_add(acc, feats, y):
    m = np.isfinite(y)
    for f in feats:
        m &= np.isfinite(f)
    F = np.stack([f[m] for f in feats], 1).astype(np.float64)
    acc[0] += F.T @ F
    acc[1] += F.T @ y[m].astype(np.float64)


def krr_fit_predict(Ktr, Ytr, Kte, lam):
    a = np.linalg.solve(Ktr + lam * np.eye(len(Ktr)), Ytr)
    return Kte @ a


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--f4-effects", default="reports/feedback/F4/snu423_effect_strength.tsv")
    ap.add_argument("--n-panel", type=int, default=2000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--min-cells-truth", type=int, default=50)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--train-sets", default="tr49,tr48")
    ap.add_argument("--summary-only", action="store_true", help="recompute summaries from r3_unit_metrics.tsv.gz")
    args = ap.parse_args()
    t0 = time.time()
    Pd, O, R = Path(args.prep_dir), Path(args.out_dir), Path(args.report_dir)
    O.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
    if args.summary_only:
        summarize(pd.read_csv(R / "r3_unit_metrics.tsv.gz", sep="\t"), R, args, log)
        return

    D = np.load(Pd / "delta.npy", mmap_mode="r")
    NC = np.load(Pd / "ncells.npy")
    wells = pd.read_csv(Pd / "wells.tsv", sep="\t")
    lines = pd.read_csv(Pd / "lines.tsv", sep="\t")
    genes = pd.read_csv(Pd / "genes.tsv", sep="\t")
    prof = np.load(Pd / "dmso_line_profile.npy")
    dmso_E = np.load(Pd / "dmso_well_E.npy")
    dmso_n = np.load(Pd / "dmso_ncells.npy")
    dmso_w = pd.read_csv(Pd / "dmso_wells.tsv", sep="\t")
    plate_n = np.load(Pd / "dmso_plate_ncells.npy")
    plates = sorted(wells.plate.unique())
    pidx = {p: i for i, p in enumerate(plates)}
    L = list(lines.cell_line_id)
    it = L.index(TEST)
    nW = len(wells)
    f4 = pd.read_csv(args.f4_effects, sep="\t").set_index("sample")
    wells["R_f4"] = wells["sample"].map(f4.R_vs_sampling)
    wells["detectable"] = wells["sample"].map(f4.resp_strict).fillna(False).astype(bool)
    wells["strong"] = wells["sample"].map(f4.resp_strong).fillna(False).astype(bool)
    wells["moa"] = wells["moa-fine"].fillna("unclear")
    wells["cond"] = wells.drug + "|" + wells.dose_uM.astype(str)
    conds = sorted(wells.cond.unique())
    cid = {c: i for i, c in enumerate(conds)}
    wells["cid"] = wells.cond.map(cid)
    cond_wells = {k: np.asarray(v) for k, v in wells.groupby("cid").groups.items()}
    cdf = wells.drop_duplicates("cid").set_index("cid").sort_index()[["drug", "dose_uM", "moa", "moa-broad", "canonical_smiles"]]
    nC = len(conds)

    # drug split for b/c (no data used); Adagrasib (plate control drug) kept in training
    drugs = sorted(wells.drug.unique())
    rng = np.random.default_rng(SEED)
    cand_drugs = [d for d in drugs if d != "Adagrasib"]
    test_drugs = set(rng.choice(cand_drugs, int(round(args.test_frac * len(drugs))), replace=False))
    wells["test_drug"] = wells.drug.isin(test_drugs)
    cdf["test_drug"] = cdf.drug.isin(test_drugs)
    log(f"wells={nW} conds={nC} drugs={len(drugs)} test_drugs={len(test_drugs)}")

    # SNU-423 truth validity
    snu_ok = (NC[it] >= args.min_cells_truth)
    wells["snu_ok"] = snu_ok
    wells["snu_ncells"] = NC[it]

    # fingerprints
    fp_all, fp_ok = morgan(cdf.drop_duplicates("drug").set_index("drug").loc[drugs, "canonical_smiles"].tolist())
    fp = pd.DataFrame(fp_all, index=drugs)
    log(f"morgan fp ok {fp_ok.sum()}/{len(drugs)}")

    dmso_ctl_rows = []
    all_rows, unit_rows, boot_rows, sel = [], [], [], {"test_drugs": sorted(test_drugs), "fp_ok": int(fp_ok.sum())}

    def evaluate(cond_name, train_set, units, Y, preds, info, deg, extra):
        """units: DataFrame (unit_id, drug, strata...), Y [n,G], preds {name: [n,G]}"""
        for name, Pm in preds.items():
            m = unit_metrics(Pm.astype(np.float64), Y, deg, units["plates"].to_numpy())
            m["predictor"] = name; m["info"] = info[name]
            m = pd.concat([units.reset_index(drop=True), m], axis=1)
            m["condition"] = cond_name; m["train_set"] = train_set
            unit_rows.append(m)

    # ------------------------------------------------------------ DEG null (SNU-423 DMSO wells, all genes)
    tau_all, deg_info = deg_null(dmso_E[:, it, :], dmso_n[:, it], dmso_w, prof[it])
    sel["deg_null"] = deg_info
    log("deg null", deg_info)

    def h_factor(widx):
        h = [1 / NC[it, w] + 1 / plate_n[pidx[wells.plate[w]], it] for w in widx]
        return float(np.sum(h)) / len(widx) ** 2

    for ts in args.train_sets.split(","):
        excl = {TEST} | ({HEPG2} if ts == "tr48" else set())
        Tr = np.array([i for i, l in enumerate(L) if l not in excl])
        trw = np.where(~wells.test_drug.to_numpy())[0]
        # ---- gene panel from training lines, training-drug wells only
        var = np.zeros(D.shape[2])
        for g0 in range(0, D.shape[2], 1000):
            x = np.asarray(D[np.ix_(Tr, trw, np.arange(g0, min(g0 + 1000, D.shape[2])))], np.float64)
            var[g0:g0 + x.shape[2]] = np.nanvar(x.reshape(-1, x.shape[2]), axis=0)
        panel = np.sort(np.argsort(-var)[: args.n_panel])
        Dp = np.asarray(D[:, :, panel], np.float32)  # [50, nW, G]
        tau = tau_all[panel]
        G = len(panel)
        snu_expr = prof[it, panel]
        sel[f"{ts}_panel"] = dict(n=G, n_lines=len(Tr), frac_panel_expr_snu_cp10k_ge0p1=float((np.expm1(snu_expr) >= 0.1).mean()),
                                  top_symbols=genes.gene_symbol.iloc[panel[np.argsort(-var[panel])[:20]]].tolist())
        np.save(O / f"panel_{ts}.npy", genes.token_id.to_numpy()[panel])
        log(ts, "panel ready", G)

        # condition-level training-line responses Dc [50, nC, G]
        Dc = np.full((len(L), nC, G), np.nan, np.float32)
        for c, ws in cond_wells.items():
            Dc[:, c] = np.nanmean(Dp[:, ws], axis=1)
        # SNU truth per condition and per well
        Ywell = Dp[it]
        Yc = np.full((nC, G), np.nan)
        hc = np.full(nC, np.nan)
        cond_ok = np.zeros(nC, bool)
        for c, ws in cond_wells.items():
            ok = [w for w in ws if snu_ok[w]]
            if ok:
                Yc[c] = np.mean(Ywell[ok], axis=0)
                hc[c] = h_factor(ok)
                cond_ok[c] = True
        cstr = pd.DataFrame({"cid": np.arange(nC)})
        cstr["detectable"] = [bool(wells.loc[[w for w in cond_wells[c] if snu_ok[w]], "detectable"].any()) if cond_ok[c] else False for c in range(nC)]
        cstr["strong"] = [bool(wells.loc[[w for w in cond_wells[c] if snu_ok[w]], "strong"].any()) if cond_ok[c] else False for c in range(nC)]
        cstr["R_f4_max"] = [float(wells.loc[[w for w in cond_wells[c] if snu_ok[w]], "R_f4"].max()) if cond_ok[c] else np.nan for c in range(nC)]
        cstr["n_wells_snu"] = [int(sum(snu_ok[w] for w in cond_wells[c])) for c in range(nC)]
        degc = np.abs(Yc) > tau[None, :] * np.sqrt(hc)[:, None]
        degw = np.abs(Ywell) > tau[None, :] * np.sqrt(np.array([h_factor([w]) if snu_ok[w] else np.nan for w in range(nW)]))[:, None]

        # DMSO-profile similarity of lines (HVG of training-line DMSO profiles)
        pv = prof[Tr].var(0)
        hv = np.argsort(-pv)[:2000]
        mu = prof[Tr][:, hv].mean(0)
        Zc = prof[:, hv] - mu
        Zn = (Zc - Zc.mean(1, keepdims=True)); Zn /= np.linalg.norm(Zn, axis=1, keepdims=True)
        sim = Zn @ Zn.T
        order_t = Tr[np.argsort(-sim[it, Tr])]
        sel[f"{ts}_nearest_lines_to_snu423"] = [(lines.cell_name[i], round(float(sim[it, i]), 3)) for i in order_t[:10]]
        _, _, Vt = np.linalg.svd(Zc[Tr], full_matrices=False)
        Zpc = Zc @ Vt[:10].T
        Zpc /= Zpc[Tr].std(0)

        # ---- negative control: DMSO wells (no drug). Delta of a DMSO well vs its plate DMSO pool, SNU-423 vs the
        # mean of training lines in the same DMSO well. Any agreement here is a shared well (technical) effect.
        plate_E = np.load(Pd / "dmso_plate_E.npy")
        dctl = []
        for k, r in dmso_w.iterrows():
            q = pidx[r.plate]
            sd = (dmso_E[k, it, :] - plate_E[q, it, :])[panel]
            okl = Tr[dmso_n[k, Tr] >= 20]
            od = np.mean(dmso_E[k][okl][:, panel] - plate_E[q][okl][:, panel], axis=0)
            dctl.append(dict(train_set=ts, dmso_well=r["sample"], plate=r.plate, snu_cells=int(dmso_n[k, it]),
                             pearson_snu_vs_trainlines_same_dmso_well=float(np.corrcoef(sd, od)[0, 1]),
                             rms_snu=float(np.sqrt(np.mean(sd ** 2))), rms_trainlines_mean=float(np.sqrt(np.mean(od ** 2))),
                             rmse_no_change=float(np.sqrt(np.mean(sd ** 2))), rmse_same_well=float(np.sqrt(np.mean((sd - od) ** 2)))))
        dmso_ctl_rows.extend(dctl)
        sel[f"{ts}_dmso_well_control"] = dict(median_pearson=float(np.median([d["pearson_snu_vs_trainlines_same_dmso_well"] for d in dctl])),
                                              n=len(dctl))
        log(ts, "dmso well control", sel[f"{ts}_dmso_well_control"])

        # ======================================================== condition aW (cond level, Tahoe-LOCO incl. test well)
        DcT = Dc[Tr]
        valid = np.isfinite(DcT[:, :, 0])
        S = np.nansum(DcT, axis=0); N = valid.sum(0)
        M = S / np.maximum(N, 1)[:, None]
        glob = np.nanmean(M, axis=0)
        # LOSO calibration a*same + b*global, and kNN-k selection
        acc = [np.zeros((2, 2)), np.zeros(2)]
        ks = [1, 2, 3, 5, 10, 20, len(Tr) - 1]
        kerr = {k: [0.0, 0] for k in ks}
        acc_knn = {k: [np.zeros((2, 2)), np.zeros(2)] for k in ks}
        for j, l in enumerate(Tr):
            y = DcT[j]
            Ml = (S - np.nan_to_num(y)) / np.maximum(N - valid[j], 1)[:, None]
            gl = np.nanmean(Ml, axis=0)
            acc_add(acc, [Ml.ravel(), np.broadcast_to(gl, Ml.shape).ravel()], y.ravel())
            nb = [i for i in Tr[np.argsort(-sim[l, Tr])] if i != l]
            cs, cn = np.zeros((nC, G)), np.zeros(nC)
            done = 0
            for k in ks:
                for i in nb[done:k]:
                    x = Dc[i]; v = np.isfinite(x[:, 0])
                    cs[v] += x[v]; cn[v] += 1
                done = k
                pk = cs / np.maximum(cn, 1)[:, None]
                pk[cn == 0] = np.nan
                e = (pk - y) ** 2
                kerr[k][0] += np.nansum(e); kerr[k][1] += np.isfinite(e).sum()
                acc_add(acc_knn[k], [pk.ravel(), np.broadcast_to(gl, pk.shape).ravel()], y.ravel())
        ab = lsq2(acc)
        krmse = {k: float(np.sqrt(v[0] / v[1])) for k, v in kerr.items()}
        kbest = min(krmse, key=krmse.get)
        abk = lsq2(acc_knn[kbest])
        sel[f"{ts}_aW"] = dict(calib_a_b=ab.round(4).tolist(), knn_loso_rmse=krmse, knn_k=kbest, knn_calib_a_b=abk.round(4).tolist())
        log(ts, "aW calib", ab, "knn", kbest, krmse)

        def knn_pred(k, base_idx=order_t):
            nbk = base_idx[:k]
            return np.nanmean(Dc[nbk], axis=0)

        # bilinear ridge: Delta[l,c] ~ kron(e_c, [1, z_l]) B, e_c = PCA scores of training-line mean response
        def ridge_feats(Mf, K, J, lines_idx, gl):
            U, s, _ = np.linalg.svd(np.nan_to_num(Mf - gl), full_matrices=False)
            E = U[:, :K] * s[:K]
            E = E / (E.std(0) + 1e-12)
            Zt = np.concatenate([np.ones((len(lines_idx), 1)), Zpc[lines_idx, :J]], 1)
            return E, Zt

        def ridge_X(E, Zt):
            return (E[None, :, :, None] * Zt[:, None, None, :]).reshape(len(Zt) * len(E), -1)

        folds = np.array_split(np.random.default_rng(SEED).permutation(Tr), 5)
        grid = [(K, J) for K in (10, 30, 100) for J in (0, 3, 10)]
        lams = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
        cv = {}
        for K, J in grid:
            err = {lam: [0.0, 0] for lam in lams}
            for f in folds:
                trl = np.array([l for l in Tr if l not in f])
                Mf = np.nanmean(Dc[trl], axis=0); gf = np.nanmean(Mf, axis=0)
                E, Zt = ridge_feats(Mf, K, J, trl, gf)
                X = ridge_X(E, Zt); Yr = (Dc[trl] - gf).reshape(-1, G)
                ok = np.isfinite(Yr[:, 0])
                XtX = X[ok].T @ X[ok]; XtY = X[ok].T @ Yr[ok]
                _, Zv = ridge_feats(Mf, K, J, f, gf)
                Xv = ridge_X(E, Zv); Yv = (Dc[f] - gf).reshape(-1, G); okv = np.isfinite(Yv[:, 0])
                for lam in lams:
                    B = np.linalg.solve(XtX + lam * np.eye(len(XtX)), XtY)
                    e = (Xv[okv] @ B - Yv[okv]) ** 2
                    err[lam][0] += e.sum(); err[lam][1] += e.size
            for lam in lams:
                cv[(K, J, lam)] = float(np.sqrt(err[lam][0] / err[lam][1]))
        Kb, Jb, lb = min(cv, key=cv.get)
        E, Zt = ridge_feats(M, Kb, Jb, Tr, glob)
        X = ridge_X(E, Zt); Yr = (DcT - glob).reshape(-1, G); ok = np.isfinite(Yr[:, 0])
        B = np.linalg.solve(X[ok].T @ X[ok] + lb * np.eye(X.shape[1]), X[ok].T @ Yr[ok])
        _, Zte = ridge_feats(M, Kb, Jb, np.array([it]), glob)
        ridge_pred = ridge_X(E, Zte) @ B + glob
        sel[f"{ts}_aW"]["ridge"] = dict(K=Kb, J=Jb, lam=lb, cv_rmse=cv[(Kb, Jb, lb)],
                                        cv_grid={f"K{k}_J{j}_lam{l:g}": v for (k, j, l), v in cv.items()})
        log(ts, "aW ridge", Kb, Jb, lb)

        ev = np.where(cond_ok)[0]
        knnp = knn_pred(kbest)
        nn1 = knn_pred(1)
        preds = {
            "no_change": np.zeros((len(ev), G)),
            "global_mean": np.broadcast_to(glob, (len(ev), G)).copy(),
            "same_dd_mean": M[ev],
            "same_dd_calibrated": ab[0] * M[ev] + ab[1] * glob,
            "nearest_line_same_dd": np.where(np.isfinite(nn1[ev]), nn1[ev], M[ev]),
            f"knn{kbest}_same_dd": np.where(np.isfinite(knnp[ev]), knnp[ev], M[ev]),
            f"knn{kbest}_same_dd_calibrated": abk[0] * np.where(np.isfinite(knnp[ev]), knnp[ev], M[ev]) + abk[1] * glob,
            "ridge_line_pca": ridge_pred[ev],
        }
        info = {k: ("none" if k in ("no_change",) else "train" if k == "global_mean" else "W") for k in preds}
        units = cdf.loc[ev].reset_index().merge(cstr, on="cid")
        units["unit"] = units.cid.map(lambda c: conds[c])
        units["plates"] = units.cid.map(lambda c: ",".join(sorted(wells.plate[[w for w in cond_wells[c] if snu_ok[w]]].unique())))
        units["n_train_wells"] = units.cid.map(lambda c: len(cond_wells[c]))
        evaluate("aW", ts, units, Yc[ev], preds, info, degc[ev], None)
        np.savez_compressed(O / f"pred_aW_{ts}.npz", cid=ev, Y=Yc[ev].astype(np.float32),
                            **{k: v.astype(np.float32) for k, v in preds.items()})

        # ======================================================== condition aP (well level, plate-disjoint)
        evw = np.where(snu_ok)[0]
        plate_of = wells.plate.to_numpy(); drug_of = wells.drug.to_numpy(); cid_of = wells.cid.to_numpy()
        DpT = Dp[Tr]
        # per-line per-plate sums for global_off and plate_mean
        Pall = {}
        for p in plates:
            wp = np.where(plate_of == p)[0]
            Pall[p] = (np.nansum(DpT[:, wp], axis=1), np.isfinite(DpT[:, wp, 0]).sum(1))
        def lm_from_sum(sums, cnts):
            v = cnts > 0
            return (sums[v] / cnts[v][:, None]).mean(0) if v.any() else None

        def sets_for(w):
            p, d, c = plate_of[w], drug_of[w], cid_of[w]
            dd = np.where((cid_of == c) & (plate_of != p))[0]
            dr = np.where((drug_of == d) & (plate_of != p))[0]
            return dd, dr

        # global_off per plate
        glob_off = {}
        for p in plates:
            s = sum(Pall[q][0] for q in plates if q != p); n = sum(Pall[q][1] for q in plates if q != p)
            glob_off[p] = lm_from_sum(s, n)
        # LOSO calibration for hier on all wells (training lines)
        acc_h = [np.zeros((2, 2)), np.zeros(2)]
        for w in range(nW):
            dd, dr = sets_for(w)
            S_ = dd if len(dd) else dr
            if len(S_) == 0:
                continue
            per = np.nanmean(DpT[:, S_], axis=1)  # [nTr, G]
            vv = np.isfinite(per[:, 0])
            tot = np.nansum(per, axis=0); cnt = vv.sum()
            for j in range(len(Tr)):
                y = DpT[j, w]
                if not np.isfinite(y[0]) or cnt - vv[j] < 1:
                    continue
                loo = (tot - (per[j] if vv[j] else 0)) / (cnt - vv[j])
                acc_add(acc_h, [loo, glob_off[plate_of[w]]], y)
        abh = lsq2(acc_h)
        sel[f"{ts}_aP"] = dict(hier_calib_a_b=abh.round(4).tolist())
        P = {k: np.zeros((len(evw), G)) for k in ["no_change", "global_offplate", "same_dd_offplate_or_drug", "same_drug_offplate",
                                                  "hier_offplate_calibrated", f"knn{kbest}_hier_offplate", "plate_mean_otherdrugs",
                                                  "same_well_lines_mean", "snu423_replicate_offplate"]}
        src = []
        nb_k = order_t[:kbest]
        for i, w in enumerate(evw):
            p = plate_of[w]
            dd, dr = sets_for(w)
            go = glob_off[p]
            P["global_offplate"][i] = go
            hdd = set_line_mean(Dp, Tr, dd) if len(dd) else None
            hdr = set_line_mean(Dp, Tr, dr) if len(dr) else None
            h = hdd if hdd is not None else (hdr if hdr is not None else go)
            src.append("dd" if hdd is not None else ("drug" if hdr is not None else "global"))
            P["same_dd_offplate_or_drug"][i] = h
            P["same_drug_offplate"][i] = hdr if hdr is not None else go
            P["hier_offplate_calibrated"][i] = abh[0] * h + abh[1] * go
            S_ = dd if len(dd) else dr
            kn = set_line_mean(Dp, nb_k, S_) if len(S_) else None
            P[f"knn{kbest}_hier_offplate"][i] = kn if kn is not None else h
            wp = np.where((plate_of == p) & (drug_of != drug_of[w]))[0]
            P["plate_mean_otherdrugs"][i] = set_line_mean(Dp, Tr, wp)
            sw = set_line_mean(Dp, Tr, np.array([w]))
            P["same_well_lines_mean"][i] = sw if sw is not None else go
            rep = np.array([x for x in dd if snu_ok[x]])
            P["snu423_replicate_offplate"][i] = Dp[it, rep].mean(0) if len(rep) else np.nan
        info = {"no_change": "none", "global_offplate": "train", "same_dd_offplate_or_drug": "train", "same_drug_offplate": "train",
                "hier_offplate_calibrated": "train", f"knn{kbest}_hier_offplate": "train", "plate_mean_otherdrugs": "plate",
                "same_well_lines_mean": "W", "snu423_replicate_offplate": "snu_treated"}
        unitsw = wells.loc[evw, ["sample", "plate", "drug", "dose_uM", "moa", "moa-broad", "cid", "detectable", "strong", "R_f4", "snu_ncells"]].reset_index(drop=True)
        unitsw["unit"] = unitsw["sample"]
        unitsw["plates"] = unitsw["plate"]
        unitsw["offplate_source"] = src
        sel[f"{ts}_aP"]["offplate_source_counts"] = pd.Series(src).value_counts().to_dict()
        rep_ok = np.isfinite(P["snu423_replicate_offplate"][:, 0])
        Pw = {k: v for k, v in P.items() if k != "snu423_replicate_offplate"}
        evaluate("aP", ts, unitsw, Ywell[evw].astype(np.float64), Pw, info, degw[evw], None)
        # replicate-ceiling subset: wells with an off-plate SNU-423 replicate of the same drug x dose
        if rep_ok.sum() >= 5:
            sub = {k: v[rep_ok] for k, v in P.items()}
            evaluate("aP_repsubset", ts, unitsw[rep_ok], Ywell[evw][rep_ok].astype(np.float64), sub, info, degw[evw][rep_ok], None)
        np.savez_compressed(O / f"pred_aP_{ts}.npz", wells=evw, Y=Ywell[evw].astype(np.float32), **{k: v.astype(np.float32) for k, v in P.items()})
        log(ts, "aP done", sel[f"{ts}_aP"])

        # ======================================================== condition b (unseen line + unseen drug)
        trc = np.where(~cdf.test_drug.to_numpy())[0]
        tec = np.array([c for c in np.where(cdf.test_drug.to_numpy())[0] if cond_ok[c]])
        Mb = M  # training-line means; only rows of training drugs are used below
        glob_b = np.nanmean(M[trc], axis=0)
        dose_of = cdf.dose_uM.to_numpy(); moa_of = cdf.moa.to_numpy(); drug_c = cdf.drug.to_numpy()

        def group_means(Ysrc, train_idx, test_idx, gl):
            dm = {x: np.nanmean(Ysrc[[c for c in train_idx if dose_of[c] == x]], axis=0) for x in np.unique(dose_of)}
            pd_ = np.stack([dm[dose_of[c]] for c in test_idx])
            pm = []
            for c in test_idx:
                sel_ = [k for k in train_idx if moa_of[k] == moa_of[c] and dose_of[k] == dose_of[c] and moa_of[c] != "unclear"]
                pm.append(np.nanmean(Ysrc[sel_], axis=0) if sel_ else dm[dose_of[c]])
            return pd_, np.stack(pm), dm

        def fp_models(Ysrc, train_idx, test_idx, tag):
            """Tanimoto kernel ridge (x dose kernel) and Tanimoto kNN, hyperparameters by drug-grouped 5-fold CV
            inside train_idx (targets = Ysrc rows)."""
            Fd = fp.loc[drug_c].to_numpy()
            T_ = tanimoto(Fd, Fd)
            same_dose = (dose_of[:, None] == dose_of[None, :]).astype(float)
            tr_drugs = np.array(sorted(set(drug_c[train_idx])))
            fdr = np.array_split(np.random.default_rng(SEED + 1).permutation(tr_drugs), 5)
            res_krr, res_knn = {}, {}
            for f in fdr:
                fset = set(f)
                a_ = np.array([c for c in train_idx if drug_c[c] not in fset]); v_ = np.array([c for c in train_idx if drug_c[c] in fset])
                dm = {x: np.nanmean(Ysrc[[c for c in a_ if dose_of[c] == x]], axis=0) for x in np.unique(dose_of)}
                off_a = np.stack([dm[dose_of[c]] for c in a_]); off_v = np.stack([dm[dose_of[c]] for c in v_])
                for gam in (0.0, 0.5, 1.0):
                    Kf = T_ * (same_dose + gam * (1 - same_dose))
                    for lam in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
                        pr = krr_fit_predict(Kf[np.ix_(a_, a_)], Ysrc[a_] - off_a, Kf[np.ix_(v_, a_)], lam) + off_v
                        e = res_krr.setdefault((gam, lam), [0.0, 0]); e[0] += np.sum((pr - Ysrc[v_]) ** 2); e[1] += pr.size
                for k in (1, 3, 5, 10, 20):
                    pr = knn_fp(T_, a_, v_, Ysrc, k, dm)
                    e = res_knn.setdefault(k, [0.0, 0]); e[0] += np.sum((pr - Ysrc[v_]) ** 2); e[1] += pr.size
            gk = min(res_krr, key=lambda x: res_krr[x][0] / res_krr[x][1])
            kk = min(res_knn, key=lambda x: res_knn[x][0] / res_knn[x][1])
            dm = {x: np.nanmean(Ysrc[[c for c in train_idx if dose_of[c] == x]], axis=0) for x in np.unique(dose_of)}
            off_tr = np.stack([dm[dose_of[c]] for c in train_idx]); off_te = np.stack([dm[dose_of[c]] for c in test_idx])
            Kf = T_ * (same_dose + gk[0] * (1 - same_dose))
            p_krr = krr_fit_predict(Kf[np.ix_(train_idx, train_idx)], Ysrc[train_idx] - off_tr, Kf[np.ix_(test_idx, train_idx)], gk[1]) + off_te
            p_knn = knn_fp(T_, train_idx, test_idx, Ysrc, kk, dm)
            sel[tag] = dict(krr_gamma=gk[0], krr_lambda=gk[1], krr_cv_rmse=float(np.sqrt(res_krr[gk][0] / res_krr[gk][1])),
                            knn_k=int(kk), knn_cv_rmse=float(np.sqrt(res_knn[kk][0] / res_knn[kk][1])),
                            max_tanimoto_test_to_train_median=float(np.median(T_[np.ix_(test_idx, train_idx)].max(1))))
            return p_krr, p_knn

        def knn_fp(T_, a_, v_, Ysrc, k, dm):
            out = []
            for c in v_:
                cands = [x for x in a_ if dose_of[x] == dose_of[c]]
                if not cands:
                    out.append(dm[dose_of[c]]); continue
                cands = np.array(cands)
                o = cands[np.argsort(-T_[c, cands], kind="stable")][:k]
                out.append(np.nanmean(Ysrc[o], axis=0))
            return np.stack(out)

        pd_b, pm_b, _ = group_means(M, trc, tec, glob_b)
        p_krr_b, p_knn_b = fp_models(M, trc, tec, f"{ts}_b_fp")
        # plate-sharing diagnostic and same-well oracle
        pl_b, sw_b = [], []
        for c in tec:
            ws = [w for w in cond_wells[c] if snu_ok[w]]
            wp = np.where(np.isin(plate_of, plate_of[ws]) & ~wells.test_drug.to_numpy())[0]
            a_ = set_line_mean(Dp, Tr, wp); b_ = set_line_mean(Dp, Tr, np.array(ws))
            pl_b.append(a_ if a_ is not None else glob_b); sw_b.append(b_ if b_ is not None else glob_b)
        preds_b = {"no_change": np.zeros((len(tec), G)), "global_mean": np.broadcast_to(glob_b, (len(tec), G)).copy(),
                   "dose_mean": pd_b, "moa_dose_mean": pm_b, "fp_kernel_ridge": p_krr_b, "fp_knn": p_knn_b,
                   "plate_mean_traindrugs": np.stack(pl_b), "same_well_lines_mean": np.stack(sw_b)}
        info_b = {"no_change": "none", "global_mean": "train", "dose_mean": "train", "moa_dose_mean": "train", "fp_kernel_ridge": "train",
                  "fp_knn": "train", "plate_mean_traindrugs": "plate", "same_well_lines_mean": "W"}
        units_b = cdf.loc[tec].reset_index().merge(cstr, on="cid")
        units_b["unit"] = units_b.cid.map(lambda c: conds[c])
        units_b["plates"] = units_b.cid.map(lambda c: ",".join(sorted(wells.plate[[w for w in cond_wells[c] if snu_ok[w]]].unique())))
        evaluate("b", ts, units_b, Yc[tec], preds_b, info_b, degc[tec], None)
        np.savez_compressed(O / f"pred_b_{ts}.npz", cid=tec, Y=Yc[tec].astype(np.float32), **{k: v.astype(np.float32) for k, v in preds_b.items()})
        log(ts, "b done", sel[f"{ts}_b_fp"])

        # ======================================================== condition c (within SNU-423, unseen drug) - panel of this train set
        trc_s = np.array([c for c in trc if cond_ok[c]])
        glob_c = np.nanmean(Yc[trc_s], axis=0)
        pd_c, pm_c, _ = group_means(Yc, trc_s, tec, glob_c)
        p_krr_c, p_knn_c = fp_models(Yc, trc_s, tec, f"{ts}_c_fp")
        pl_c = []
        for c in tec:
            ws = [w for w in cond_wells[c] if snu_ok[w]]
            wp = [w for w in np.where(np.isin(plate_of, plate_of[ws]) & ~wells.test_drug.to_numpy())[0] if snu_ok[w]]
            pl_c.append(Ywell[wp].mean(0))
        preds_c = {"no_change": np.zeros((len(tec), G)), "global_mean_snu": np.broadcast_to(glob_c, (len(tec), G)).copy(),
                   "dose_mean_snu": pd_c, "moa_dose_mean_snu": pm_c, "fp_kernel_ridge_snu": p_krr_c, "fp_knn_snu": p_knn_c,
                   "plate_mean_traindrugs_snu": np.stack(pl_c)}
        info_c = {k: ("none" if k == "no_change" else "plate" if k.startswith("plate") else "train_snu") for k in preds_c}
        evaluate("c", ts, units_b, Yc[tec], preds_c, info_c, degc[tec], None)
        np.savez_compressed(O / f"pred_c_{ts}.npz", cid=tec, Y=Yc[tec].astype(np.float32), **{k: v.astype(np.float32) for k, v in preds_c.items()})
        log(ts, "c done", sel[f"{ts}_c_fp"])

        # unit-level descriptors of SNU truth for this panel
        sel[f"{ts}_truth"] = dict(n_cond_eval=int(cond_ok.sum()), n_wells_eval=int(snu_ok.sum()), n_test_conds_b=int(len(tec)),
                                  median_deg_per_cond=float(np.median(degc[cond_ok].sum(1))),
                                  median_deg_per_cond_detectable=float(np.median(degc[cond_ok & cstr.detectable.to_numpy()].sum(1))),
                                  median_deg_per_cond_notdetectable=float(np.median(degc[cond_ok & ~cstr.detectable.to_numpy()].sum(1))))
        del Dp, Dc

    sel["args"] = vars(args)
    sel["elapsed_s_eval"] = round(time.time() - t0, 1)
    (R / "r3_selection.json").write_text(json.dumps(sel, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    pd.DataFrame({"drug": sorted(test_drugs)}).to_csv(R / "r3_test_drugs.tsv", sep="\t", index=False)
    pd.DataFrame(dmso_ctl_rows).to_csv(R / "r3_dmso_well_control.tsv", sep="\t", index=False, float_format="%.5g")
    U = pd.concat(unit_rows, ignore_index=True)
    summarize(U, R, args, log)


def summarize(U, R, args, log):
    # ------------------------------------------------------------ summaries
    U["detectable"] = U.detectable.astype(bool); U["strong"] = U.strong.astype(bool)
    boot_rows = []
    U["stratum_strength"] = np.where(U.strong, "strong", np.where(U.detectable, "detectable_not_strong", "not_detectable"))
    keep_cols = ["condition", "train_set", "predictor", "info", "unit", "drug", "dose_uM", "moa", "moa-broad", "detectable", "strong",
                 "stratum_strength", "plates", "R_f4" if "R_f4" in U else "R_f4_max", "rmse", "pearson", "pearson_pertcentered",
                 "pearson_platecentered", "retrieval_rank",
                 "n_deg", "deg_auprc", "deg_prec_at_k", "deg_sign_agree"]
    if "R_f4_any" not in U:
        U["R_f4_any"] = U["R_f4"].fillna(U["R_f4_max"]) if "R_f4" in U else U["R_f4_max"]
    keep_cols = [c for c in keep_cols if c not in ("R_f4", "R_f4_max")] + ["R_f4_any"]
    U[keep_cols].to_csv(R / "r3_unit_metrics.tsv.gz", sep="\t", index=False, float_format="%.5g")
    mets = ["rmse", "pearson", "pearson_pertcentered", "pearson_platecentered", "retrieval_rank", "deg_auprc", "deg_prec_at_k", "deg_sign_agree"]
    strata = {"all": lambda d: d, "detectable": lambda d: d[d.detectable], "strong": lambda d: d[d.strong],
              "not_detectable": lambda d: d[~d.detectable]}
    rows = []
    for (cn, ts, pr), g in U.groupby(["condition", "train_set", "predictor"], sort=False):
        for sn, f in strata.items():
            x = f(g)
            r = dict(condition=cn, train_set=ts, stratum=sn, predictor=pr, info=g["info"].iloc[0], n_units=len(x),
                     n_drugs=x.drug.nunique())
            for m in mets:
                r[m] = x[m].mean()
            r["n_units_deg"] = int(x.deg_auprc.notna().sum())
            rows.append(r)
    Smry = pd.DataFrame(rows)
    Smry.to_csv(R / "r3_summary.tsv", sep="\t", index=False, float_format="%.5g")

    # drug-clustered paired bootstrap
    brng = np.random.default_rng(SEED + 2)
    for (cn, ts), g in U.groupby(["condition", "train_set"], sort=False):
        allg = g
        nonshare = Smry[(Smry.condition == cn) & (Smry.train_set == ts) & (Smry.stratum == "all") &
                        Smry["info"].isin(["none", "train", "train_snu"])]
        ref = nonshare.sort_values("rmse").predictor.iloc[0]
        for sn, f in strata.items():
            x = f(allg)
            if x.unit.nunique() < 5:
                continue
            piv = {m: x.pivot_table(index=["unit", "drug"], columns="predictor", values=m)
                   for m in ("rmse", "pearson_pertcentered", "pearson_platecentered", "pearson")}
            ud = pd.Index(sorted(set(x.drug)))
            Wb = np.stack([np.bincount(brng.integers(0, len(ud), len(ud)), minlength=len(ud)) for _ in range(args.nboot)])
            for m, pv in piv.items():
                dcode = ud.get_indexer(pv.index.get_level_values("drug"))
                for pr in pv.columns:
                    for base in ("no_change", ref):
                        if pr == base or base not in pv.columns:
                            continue
                        dlt = (pv[pr] - pv[base]).to_numpy()
                        ok = np.isfinite(dlt)
                        if ok.sum() < 5:
                            continue
                        sd = np.bincount(dcode[ok], weights=dlt[ok], minlength=len(ud)); nd = np.bincount(dcode[ok], minlength=len(ud))
                        bs = (Wb @ sd) / np.maximum(Wb @ nd, 1)
                        boot_rows.append(dict(condition=cn, train_set=ts, stratum=sn, metric=m, predictor=pr, baseline=base,
                                              ref_is_best_nonsharing=(base == ref), n_units=int(ok.sum()),
                                              mean_diff=float(dlt[ok].mean()), ci_lo=float(np.percentile(bs, 2.5)),
                                              ci_hi=float(np.percentile(bs, 97.5))))
    pd.DataFrame(boot_rows).to_csv(R / "r3_bootstrap.tsv", sep="\t", index=False, float_format="%.5g")

    # failure patterns: by MoA-broad and dose for key predictors (aW tr49, aP tr49, b, c)
    fp_rows = []
    for (cn, ts, pr), g in U[U.train_set == "tr49"].groupby(["condition", "train_set", "predictor"]):
        for col in ("moa-broad", "dose_uM", "stratum_strength"):
            for v, x in g.groupby(col):
                fp_rows.append(dict(condition=cn, predictor=pr, by=col, value=v, n=len(x), rmse=x.rmse.mean(),
                                    pearson=x.pearson.mean(), pearson_pertcentered=x.pearson_pertcentered.mean(),
                                    pearson_platecentered=x.pearson_platecentered.mean()))
    pd.DataFrame(fp_rows).to_csv(R / "r3_breakdown.tsv", sep="\t", index=False, float_format="%.5g")
    log("done")


if __name__ == "__main__":
    main()
