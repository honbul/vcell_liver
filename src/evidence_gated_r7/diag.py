"""D1 on training drugs of source cells only: response similarity vs chemistry similarity vs target overlap (plate-aware)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import pman, dump
from .data import Store, TrainStats, Knowledge


def tanimoto(a):
    inter = a @ a.T
    n = a.sum(1)
    return inter / np.maximum(n[:, None] + n[None, :] - inter, 1)


def cos(y):
    y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    return y @ y.T


def main():
    S = Store(); st = TrainStats(S); K = Knowledge(len(S.groups)); u = S.units
    tr = S.train[S.source(S.train)]
    cells = u.iloc[tr].groupby("cell").size().sort_values(ascending=False, kind="mergesort").index[:12]
    tsets = K.seeds.groupby("group_idx").node.apply(frozenset).to_dict()
    rng = np.random.RandomState(0)
    rows, pool = [], []
    for c in sorted(cells):
        idx = tr[u.cell.to_numpy()[tr] == c]
        if len(idx) > 800:
            idx = np.sort(rng.choice(idx, 800, replace=False))
        y = S.delta(idx) - st.cell_mean[S.cell_idx[idx]]
        C = cos(y); T = tanimoto(S.fp(idx).astype(np.float32))
        g = S.group_idx[idx]
        ann = np.array([gi in tsets for gi in g])
        plates = [set(p.split("|")) for p in u.plates.to_numpy()[idx]]
        iu = np.triu_indices(len(idx), 1)
        both = ann[iu[0]] & ann[iu[1]]
        jac = np.full(len(iu[0]), np.nan)
        k = np.where(both)[0]
        jac[k] = [len(tsets[g[i]] & tsets[g[j]]) / len(tsets[g[i]] | tsets[g[j]]) for i, j in zip(iu[0][k], iu[1][k])]
        share = np.array([len(plates[i] & plates[j]) > 0 for i, j in zip(*iu)])
        cc, tt = C[iu], T[iu]
        lowT = tt < 0.3
        ov = both & (jac > 0)
        r = dict(cell=c, units=len(idx), frac_annotated=float(ann.mean()), pairs=int(len(cc)), pairs_both_annotated=int(both.sum()),
                 spearman_tanimoto=float(spearmanr(tt, cc).correlation),
                 spearman_jaccard_both_annotated=float(spearmanr(jac[both], cc[both]).correlation) if both.sum() > 10 else np.nan,
                 mean_cos_target_overlap=float(cc[ov].mean()) if ov.any() else np.nan, n_target_overlap=int(ov.sum()),
                 mean_cos_no_overlap_both_annotated=float(cc[both & (jac == 0)].mean()),
                 mean_cos_target_overlap_lowT=float(cc[ov & lowT].mean()) if (ov & lowT).any() else np.nan, n_target_overlap_lowT=int((ov & lowT).sum()),
                 mean_cos_no_overlap_lowT=float(cc[both & (jac == 0) & lowT].mean()),
                 mean_cos_target_overlap_lowT_diffplate=float(cc[ov & lowT & ~share].mean()) if (ov & lowT & ~share).any() else np.nan,
                 n_target_overlap_lowT_diffplate=int((ov & lowT & ~share).sum()),
                 mean_cos_no_overlap_lowT_diffplate=float(cc[both & (jac == 0) & lowT & ~share].mean()),
                 mean_cos_tanimoto_ge_0p6=float(cc[tt >= 0.6].mean()) if (tt >= 0.6).any() else np.nan, n_tanimoto_ge_0p6=int((tt >= 0.6).sum()),
                 mean_cos_shared_plate=float(cc[share].mean()), mean_cos_not_shared_plate=float(cc[~share].mean()))
        rows.append(r)
        print(r, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(pman("d1_chem_target_response.tsv"), sep="\t", index=False)
    summ = {k: float(d[k].median()) for k in d.columns if k not in ("cell",) and d[k].dtype != object}
    dump(dict(cells=list(d.cell), medians=summ,
              note="training drugs, source cells only; cosine of cell-centred delta; informative only (no gate)"), pman("d1_summary.json"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
