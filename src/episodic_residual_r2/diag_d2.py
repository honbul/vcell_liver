"""D2: is there structure in the offset-removed residuals? Source only.

(1) drug pairs inside one cell: correlation of centred residuals vs. fingerprint similarity, split by shared plate;
(2) replicate ceiling: the same drug measured on two plates of the same cell vs. different drugs on different plates.
Residual = unit (or plate-row) delta - leave-family-out f0, centred on the cell's mean residual over its eligible units.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch

from .common import public, dump, new_run_dir, run_record
from .data import Source
from .methods import tanimoto
from src.transigen_adaptation_b1.data import Data

TBINS = [0.0, 0.2, 0.4, 0.6, 1.0001]


def rowcorr(a, b):
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d2_v1")
    ap.add_argument("--max-units", type=int, default=600)
    a = ap.parse_args()
    t0 = time.time()
    out = new_run_dir("diag", a.run_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = Source(dev)
    rs = np.random.RandomState(20261117)
    pair_rows, rep_rows = [], []
    B = Data()
    n = B.n_src
    brow = B.rows.iloc[:n]
    ex = pd.read_csv("manifests/transigen_adaptation_b1/source_rows_excluded_degenerate.tsv", sep="\t").row_id.to_numpy()
    keep = np.ones(n, bool)
    keep[ex] = False
    gidx = brow.drug_group.map(src.groups.set_index("drug_group").group_idx).to_numpy()
    x1 = B.source_x1()
    for role in ("train", "val"):
        for ci in src.eval_cells(role):
            fam = src.cell_family[ci]
            regime = src.regime_of(ci)
            S, nt = src.f0_table(regime, fam if regime == "meta_train" else None)
            ui_all = src.eligible_units(ci, nt)
            f0_all, _ = src.f0_from(S, nt, src.u_group[ui_all])
            E_all = src.D[torch.tensor(ui_all, device=dev)] - f0_all
            mres = E_all.mean(0, keepdim=True)                       # the cell's common offset (diagnostic: uses all its units)
            offset_share = float((mres ** 2).mean() / (E_all ** 2).mean())
            sel = rs.choice(len(ui_all), min(a.max_units, len(ui_all)), replace=False)
            ui = ui_all[sel]
            R = E_all[torch.tensor(sel, device=dev)] - mres
            F0c = f0_all[torch.tensor(sel, device=dev)]
            F0c = F0c - f0_all.mean(0, keepdim=True)
            Rn = R - R.mean(1, keepdim=True)
            Rn = Rn / Rn.norm(dim=1, keepdim=True).clamp(min=1e-12)
            C = (Rn @ Rn.T).cpu().numpy()
            Fn = F0c - F0c.mean(1, keepdim=True)
            Fn = Fn / Fn.norm(dim=1, keepdim=True).clamp(min=1e-12)
            CF = (Fn @ Fn.T).cpu().numpy()
            fp = src.FP[torch.tensor(src.u_group[ui], device=dev)]
            T = tanimoto(fp, fp).cpu().numpy()
            pl = [set(p.split("|")) for p in src.units.plates.to_numpy()[ui]]
            iu = np.triu_indices(len(ui), 1)
            same = np.array([bool(pl[i] & pl[j]) for i, j in zip(*iu)])
            c, t, cf = C[iu], T[iu], CF[iu]
            for sp in (True, False):
                m = same == sp
                for lo, hi in zip(TBINS[:-1], TBINS[1:]):
                    mm = m & (t >= lo) & (t < hi)
                    if mm.sum() >= 5:
                        pair_rows.append(dict(role=role, cell=src.cell_name[ci], family=fam, same_plate=sp, kind="tanimoto", bin=f"[{lo:.1f},{min(hi,1):.1f})",
                                              n_pairs=int(mm.sum()), mean_residual_corr=float(c[mm].mean())))
                for lo, hi in ((-1.01, 0.0), (0.0, 0.3), (0.3, 0.6), (0.6, 1.01)):
                    mm = m & (cf >= lo) & (cf < hi)
                    if mm.sum() >= 5:
                        pair_rows.append(dict(role=role, cell=src.cell_name[ci], family=fam, same_plate=sp, kind="f0_corr", bin=f"[{max(lo,-1):.1f},{min(hi,1):.1f})",
                                              n_pairs=int(mm.sum()), mean_residual_corr=float(c[mm].mean())))
                if m.sum() >= 5:
                    pair_rows.append(dict(role=role, cell=src.cell_name[ci], family=fam, same_plate=sp, kind="all", bin="all", n_pairs=int(m.sum()),
                                          mean_residual_corr=float(c[m].mean())))
            # replicate ceiling from plate rows of this cell
            rm = keep & (brow.cell.to_numpy() == src.cell_name[ci]) & (nt[torch.tensor(np.nan_to_num(gidx, nan=0).astype(int), device=dev)].cpu().numpy() >= 1)
            rr = brow[rm]
            cnt = rr.groupby("pert_id").det_plate.nunique()
            multi = cnt[cnt >= 2].index
            row = dict(role=role, cell=src.cell_name[ci], family=fam, n_units=len(ui_all), offset_share_of_residual_ms=offset_share,
                       residual_rms=float(torch.sqrt((E_all ** 2).mean())), centred_residual_rms=float(torch.sqrt(((E_all - mres) ** 2).mean())),
                       n_replicated_conditions=int(len(multi)))
            if len(multi) >= 10:
                rr2 = rr[rr.pert_id.isin(multi)].sort_values(["pert_id", "det_plate"])
                two = rr2.groupby("pert_id").head(2)
                ids = two.row_id.to_numpy().reshape(-1, 2)
                if len(ids) > 400:
                    ids = ids[rs.choice(len(ids), 400, replace=False)]
                d = torch.tensor(np.asarray(x1[np.sort(ids.ravel())]) - np.asarray(B.x0[np.sort(ids.ravel())]), device=dev)
                order = np.argsort(np.argsort(ids.ravel()))
                d = d[torch.tensor(order, device=dev)].reshape(len(ids), 2, -1)
                g = torch.tensor(gidx[ids[:, 0]].astype(int), device=dev)
                f0r = S[g] / nt[g][:, None]
                r1, r2 = d[:, 0] - f0r - mres, d[:, 1] - f0r - mres
                same_drug = rowcorr(r1, r2)
                perm = torch.roll(torch.arange(len(ids), device=dev), 1)
                plates = two.det_plate.to_numpy().reshape(-1, 2) if len(two) == 2 * len(ids) else None
                diff_drug = rowcorr(r1, r2[perm])
                row.update(n_replicate_pairs=int(len(ids)), replicate_same_drug_corr=float(same_drug.mean()), replicate_diff_drug_corr=float(diff_drug.mean()),
                           raw_delta_same_drug_corr=float(rowcorr(d[:, 0], d[:, 1]).mean()), raw_delta_diff_drug_corr=float(rowcorr(d[:, 0], d[:, 1][perm]).mean()))
            rep_rows.append(row)
    pairs = pd.DataFrame(pair_rows)
    reps = pd.DataFrame(rep_rows)
    pairs.to_csv(os.path.join(out, "d2_pairs_by_cell.tsv"), sep="\t", index=False)
    reps.to_csv(public("manifests", "episodic_residual_r2", "d2_cells.tsv"), sep="\t", index=False, float_format="%.5g")
    fam = pairs.groupby(["role", "kind", "bin", "same_plate", "family"]).agg(c=("mean_residual_corr", "mean"), n_pairs=("n_pairs", "sum")).reset_index()
    summ = fam.groupby(["role", "kind", "bin", "same_plate"]).agg(mean_residual_corr=("c", "mean"), n_families=("family", "nunique"), n_pairs=("n_pairs", "sum")).reset_index()
    summ.to_csv(public("manifests", "episodic_residual_r2", "d2_pair_summary.tsv"), sep="\t", index=False, float_format="%.5g")
    rf = reps.groupby(["role", "family"]).mean(numeric_only=True).reset_index()
    cols = [c for c in ["offset_share_of_residual_ms", "residual_rms", "centred_residual_rms", "replicate_same_drug_corr", "replicate_diff_drug_corr",
                        "raw_delta_same_drug_corr", "raw_delta_diff_drug_corr"] if c in rf]
    rsum = rf.groupby("role")[cols].agg(["mean", "count"])
    rsum.columns = ["_".join(c) for c in rsum.columns]
    rsum.reset_index().to_csv(public("manifests", "episodic_residual_r2", "d2_replicate_summary.tsv"), sep="\t", index=False, float_format="%.5g")
    dump(run_record(a.run_id, "diag_d2", t0, n_cells=len(reps)), os.path.join(out, "result.json"))
    pd.set_option("display.width", 250)
    print(summ.to_string())
    print(rsum.T.to_string())


if __name__ == "__main__":
    main()
