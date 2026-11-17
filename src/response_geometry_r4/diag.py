"""D1-D3 diagnostics on training-drug units only (no validation / test response is read)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import public, dump, HCC
from .data import Store, TrainStats, split_half_reliability

OUT = ("manifests", "response_geometry_r4")


def tanimoto(a, b):
    inter = a @ b.T
    na = a.sum(1)[:, None]; nb = b.sum(1)[None, :]
    return inter / np.maximum(na + nb - inter, 1)


def cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12); b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def main():
    S = Store(); st = TrainStats(S); u = S.units
    tr = S.train
    rng = np.random.RandomState(0)
    res = dict(teacher_explained_variance_64=st.explained)
    # ---------------- D1
    tru = u.iloc[tr]
    counts = tru[tru.is_hcc == 0].groupby("cell").size()
    cells = sorted(counts[counts >= 40].sort_values(ascending=False, kind="mergesort").index[:12]) + list(HCC)
    rows, bins_all = [], []
    edges = [0, 0.2, 0.4, 0.6, 1.01]
    for c in cells:
        idx = tr[u.cell.to_numpy()[tr] == c]
        if len(idx) > 600:
            idx = np.sort(rng.choice(idx, 600, replace=False))
        y = S.delta(idx) - st.cell_mean[S.cell_idx[idx]]
        T = tanimoto(S.fp(idx).astype(np.float32), S.fp(idx).astype(np.float32)); C = cos(y, y)
        plates = [set(p.split("|")) for p in u.plates.to_numpy()[idx]]
        iu = np.triu_indices(len(idx), 1)
        t, cc = T[iu], C[iu]
        share = np.array([len(plates[i] & plates[j]) > 0 for i, j in zip(*iu)])
        rho = spearmanr(t, cc).correlation
        row = dict(cell=c, units=len(idx), pairs=len(t), spearman_tanimoto_vs_cos=rho,
                   close_structure_opposite_response=int(((t >= 0.6) & (cc <= 0)).sum()), n_tanimoto_ge_0p6=int((t >= 0.6).sum()),
                   far_structure_similar_response=int(((t <= 0.2) & (cc >= 0.5)).sum()), n_cos_ge_0p5=int((cc >= 0.5).sum()),
                   frac_pairs_sharing_plate=float(share.mean()),
                   mean_cos_sharing_plate=float(cc[share].mean()) if share.any() else np.nan,
                   mean_cos_not_sharing_plate=float(cc[~share].mean()) if (~share).any() else np.nan)
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (t >= lo) & (t < hi)
            row[f"mean_cos_tanimoto_{lo}_{hi:.1f}"] = float(cc[s].mean()) if s.any() else np.nan
            row[f"n_tanimoto_{lo}_{hi:.1f}"] = int(s.sum())
        rows.append(row)
    d1 = pd.DataFrame(rows)
    d1.to_csv(public(*OUT, "d1_structure_vs_response.tsv"), sep="\t", index=False)
    res["D1"] = dict(cells=cells, median_spearman=float(d1.spearman_tanimoto_vs_cos.median()),
                     close_structure_opposite_response=int(d1.close_structure_opposite_response.sum()), n_tanimoto_ge_0p6=int(d1.n_tanimoto_ge_0p6.sum()),
                     far_structure_similar_response=int(d1.far_structure_similar_response.sum()), n_cos_ge_0p5=int(d1.n_cos_ge_0p5.sum()))
    # ---------------- D2
    rep, r, ha, hb = split_half_reliability(S, st)
    m = st.cell_mean[S.cell_idx[rep]]
    ua = (((ha - m) / st.scale - st.pca_mu) @ st.U); ub = (((hb - m) / st.scale - st.pca_mu) @ st.U)
    d2rows, overl, randexp = [], [], []
    same_half, cross_half = [], []
    for c in np.unique(S.cell_idx[rep]):
        s = np.where(S.cell_idx[rep] == c)[0]
        row = dict(cell=S.cells[c], replicated_units=len(s), median_split_half_pearson=float(np.median(r[s])))
        if len(s) >= 30:
            if len(s) > 800:
                s = np.sort(rng.choice(s, 800, replace=False))
            A = cos(ua[s], ua[s]); B = cos(ub[s], ub[s])
            np.fill_diagonal(A, -np.inf); np.fill_diagonal(B, -np.inf)
            na = np.argsort(-A, 1)[:, :10]; nb = np.argsort(-B, 1)[:, :10]
            ov = np.array([len(set(na[i]) & set(nb[i])) for i in range(len(s))])
            row.update(top10_overlap_mean=float(ov.mean()), top10_overlap_random=100.0 / (len(s) - 1))
            overl.append(ov.mean()); randexp.append(100.0 / (len(s) - 1))
            # shared-control check: pairs of units, similarity from the same half (often the same plates / DMSO pool)
            # versus from opposite halves (different plates, different DMSO pools)
            Aa = cos(ua[s], ua[s]); X = cos(ua[s], ub[s])
            iu = np.triu_indices(len(s), 1)
            row.update(mean_abs_cos_same_half=float(np.abs(Aa[iu]).mean()), mean_abs_cos_cross_half=float(np.abs(X[iu]).mean()))
            same_half.append(row["mean_abs_cos_same_half"]); cross_half.append(row["mean_abs_cos_cross_half"])
        d2rows.append(row)
    d2 = pd.DataFrame(d2rows)
    d2.to_csv(public(*OUT, "d2_replicate_stability.tsv"), sep="\t", index=False)
    res["D2"] = dict(replicated_training_units=int(len(rep)), training_units=int(len(tr)), frac_replicated=float(len(rep) / len(tr)),
                     split_half_pearson_quantiles={q: float(np.quantile(r, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
                     median_split_half_pearson=float(np.median(r)),
                     cells_with_overlap=len(overl), median_top10_overlap=float(np.median(overl)) if overl else None,
                     median_top10_overlap_random=float(np.median(randexp)) if randexp else None,
                     median_abs_cos_same_half=float(np.median(same_half)) if same_half else None,
                     median_abs_cos_cross_half=float(np.median(cross_half)) if cross_half else None)
    gate_fail = res["D2"]["median_split_half_pearson"] <= 0.1 and (res["D2"]["median_top10_overlap"] or 0) <= 2 * (res["D2"]["median_top10_overlap_random"] or 0)
    res["D2"]["gate_premise_failed"] = bool(gate_fail)
    # ---------------- D3
    sig_mean = np.sqrt(((ha - hb) ** 2).mean(1)) / 2.0                     # noise scale of a two-half mean
    null_c = pd.Series(sig_mean).groupby(S.cell_idx[rep]).median()
    yall = S.delta(tr) - st.cell_mean[S.cell_idx[tr]]
    rms = np.sqrt((yall ** 2).mean(1))
    gnull = float(np.median(sig_mean))
    nc = np.array([null_c.get(c, gnull) if (S.cell_idx[rep] == c).sum() >= 10 else gnull for c in S.cell_idx[tr]])
    ratio = rms / nc
    weak = ratio < 1.5
    rep_pos = {k: i for i, k in enumerate(rep)}
    rel_tr = np.array([r[rep_pos[k]] if k in rep_pos else np.nan for k in tr])
    terc = pd.qcut(ratio, 3, labels=["low", "mid", "high"])
    ut = st.teacher(S, tr)
    wsel = np.where(weak)[0]; ssel = np.where(~weak)[0]
    wsamp = rng.choice(wsel, min(2000, len(wsel)), replace=False) if len(wsel) else wsel
    ssamp = rng.choice(ssel, min(2000, len(ssel)), replace=False)
    def mean_offdiag(ix):
        if len(ix) < 2:
            return np.nan
        C = cos(ut[ix], ut[ix]); iu = np.triu_indices(len(ix), 1)
        return float(np.abs(C[iu]).mean())
    res["D3"] = dict(null_scale_median=gnull, frac_weak_ratio_lt_1p5=float(weak.mean()), ratio_quantiles={q: float(np.quantile(ratio, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
                     split_half_pearson_by_strength_tertile={str(k): float(np.nanmedian(rel_tr[terc == k])) for k in ["low", "mid", "high"]},
                     replicated_units_by_tertile={str(k): int(np.isfinite(rel_tr[terc == k]).sum()) for k in ["low", "mid", "high"]},
                     mean_abs_teacher_cos_weak_pairs=mean_offdiag(wsamp), mean_abs_teacher_cos_strong_pairs=mean_offdiag(ssamp))
    dump(res, public(*OUT, "diagnostics_d1_d3.json"))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
