"""R3 prep: per-well responses (Delta) for all 50 lines from the A1 pseudobulk (r3_pseudobulk_all.py).

Expression of a (well, line) group: E = log1p(mean over cells of CP10k-normalised counts) = log1p(norm/n_cells).
Plate-matched control of a line: the two DMSO_TF wells of that plate pooled (cells pooled).
Delta[line, well, gene] = E(well, line) - E(plate DMSO pool, line); NaN if the (line, well) group has
< --min-cells cells (training-line noise guard; SNU-423 truth applies its own >= 50 rule in r3_loco.py).

Candidate genes (fixed before any split; uses no SNU-423 or HepG2/C3A data): mean CP10k >= --min-cp10k in the
DMSO pool averaged over the 48 non-liver lines.

Output (assets/r3/prep): delta.npy float32 [50, n_wells, n_genes], ncells.npy int [50, n_wells],
  wells.tsv (treated samples: plate, drug, dose, moa, smiles), lines.tsv, genes.tsv,
  dmso_line_profile.npy [50, n_genes] (E of all DMSO cells of a line pooled over plates),
  dmso_wells.tsv + dmso_well_E.npy [n_dmso_wells, 50, n_genes] + dmso_ncells.npy, dmso_plate_E.npy [14, 50, G].
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

LIVER = {"CVCL_0366": "SNU-423", "CVCL_1098": "HepG2/C3A"}


def parse_dose(s: str) -> tuple[str, float]:
    v = ast.literal_eval(s)
    assert len(v) == 1, s
    return str(v[0][0]).strip(), float(v[0][1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--tahoe", default="/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M")
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--min-cp10k", type=float, default=0.1)
    ap.add_argument("--f4-npz", default="assets/f4/snu423_pseudobulk.npz")
    args = ap.parse_args()
    t0 = time.time()
    P, O = Path(args.pb_dir), Path(args.out_dir)
    O.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(P / "groups.tsv", sep="\t")
    genes = pd.read_csv(P / "genes.tsv", sep="\t")  # token_id 3..62712
    lines = sorted(gt.cell_line_id.unique())
    li = {l: i for i, l in enumerate(lines)}
    nL = len(lines)
    gt["is_dmso"] = gt.drug.str.startswith("DMSO")
    dd = gt.drop_duplicates("sample")[["sample", "plate", "drug", "drugname_drugconc"]].copy()
    dd["dose_uM"] = [parse_dose(s)[1] for s in dd.drugname_drugconc]
    dm = pd.read_parquet(Path(args.tahoe) / "metadata/drug_metadata.parquet")
    plates = sorted(gt.plate.unique())

    # pass 1: DMSO per (plate, line) and per DMSO well
    dmso_w = dd[dd.drug.str.startswith("DMSO")].sort_values(["plate", "sample"]).reset_index(drop=True)
    dmso_norm = np.zeros((len(dmso_w), nL, 62713), np.float64)
    dmso_n = np.zeros((len(dmso_w), nL), np.int64)
    for p in plates:
        A = np.load(P / f"plate_{p}_norm.npy", mmap_mode="r")
        g = gt[gt.plate == p]
        for k, s in dmso_w[dmso_w.plate == p]["sample"].items():
            rows = g[g["sample"] == s].sort_values("cell_line_id")
            assert list(rows.cell_line_id) == lines
            dmso_norm[k] = A[rows.row.to_numpy()]
            dmso_n[k] = rows.n_cells.to_numpy()
    line_pool = dmso_norm.sum(0) / np.maximum(dmso_n.sum(0), 1)[:, None]  # mean CP10k per line
    nonliver = np.array([l not in LIVER for l in lines])
    cp = line_pool[nonliver].mean(0)
    cand = np.where((cp >= args.min_cp10k) & (np.arange(62713) >= 3))[0]
    print(f"candidate genes {len(cand)} (mean CP10k >= {args.min_cp10k} in 48 non-liver DMSO) {time.time()-t0:.0f}s", flush=True)
    G = len(cand)
    dmso_well_E = np.log1p(dmso_norm[:, :, cand] / np.maximum(dmso_n, 1)[:, :, None]).astype(np.float32)
    plate_E = np.zeros((len(plates), nL, G), np.float32)
    plate_n = np.zeros((len(plates), nL), np.int64)
    for i, p in enumerate(plates):
        k = dmso_w.index[dmso_w.plate == p]
        s = dmso_norm[k][:, :, cand].sum(0)
        n = dmso_n[k].sum(0)
        plate_E[i] = np.log1p(s / np.maximum(n, 1)[:, None])
        plate_n[i] = n
    line_prof = np.log1p(line_pool[:, cand]).astype(np.float32)
    del dmso_norm

    # pass 2: treated wells
    wells = dd[~dd.drug.str.startswith("DMSO")].sort_values(["plate", "sample"]).reset_index(drop=True)
    nW = len(wells)
    D = np.full((nL, nW, G), np.nan, np.float32)
    NC = np.zeros((nL, nW), np.int64)
    pidx = {p: i for i, p in enumerate(plates)}
    for p in plates:
        A = np.load(P / f"plate_{p}_norm.npy", mmap_mode="r")
        g = gt[gt.plate == p]
        for k, s in wells[wells.plate == p]["sample"].items():
            rows = g[g["sample"] == s].sort_values("cell_line_id")
            n = rows.n_cells.to_numpy()
            E = np.log1p(np.asarray(A[rows.row.to_numpy()][:, cand], np.float64) / np.maximum(n, 1)[:, None])
            d = (E - plate_E[pidx[p]]).astype(np.float32)
            d[n < args.min_cells] = np.nan
            D[:, k] = d
            NC[:, k] = n
        print(f"{p} done {time.time()-t0:.0f}s", flush=True)

    dmd = dm.drop_duplicates("drug").set_index("drug")
    wells = wells.merge(dm.drop_duplicates("drug")[["drug", "moa-fine", "moa-broad", "targets", "canonical_smiles", "pubchem_cid"]],
                        on="drug", how="left")
    wells["drug_name_conc"] = [parse_dose(s)[0] for s in wells.drugname_drugconc]
    np.save(O / "delta.npy", D)
    np.save(O / "ncells.npy", NC)
    np.save(O / "dmso_line_profile.npy", line_prof)
    np.save(O / "dmso_well_E.npy", dmso_well_E)
    np.save(O / "dmso_ncells.npy", dmso_n)
    np.save(O / "dmso_plate_E.npy", plate_E)
    np.save(O / "dmso_plate_ncells.npy", plate_n)
    wells.to_csv(O / "wells.tsv", sep="\t", index=False)
    dmso_w.to_csv(O / "dmso_wells.tsv", sep="\t", index=False)
    cl = gt.drop_duplicates("cell_line_id").set_index("cell_line_id")
    ldf = pd.DataFrame({"cell_line_id": lines, "cell_name": [cl.at[l, "cell_name"] for l in lines],
                        "organ": [cl.at[l, "Organ"] for l in lines],
                        "dmso_cells": dmso_n.sum(0), "median_cells_per_treated_well": np.median(NC, 1),
                        "frac_wells_lt_min_cells": (NC < args.min_cells).mean(1)})
    ldf.to_csv(O / "lines.tsv", sep="\t", index=False)
    genes.set_index("token_id").loc[cand].reset_index().assign(dmso_cp10k_nonliver=cp[cand]).to_csv(O / "genes.tsv", sep="\t", index=False)
    R = Path(args.report_dir)
    R.mkdir(parents=True, exist_ok=True)
    # provenance cross-check: SNU-423 raw per-well sums vs F4's independent build from the by_cell_line h5ad
    xcheck = {}
    if args.f4_npz and Path(args.f4_npz).exists():
        import io
        z = np.load(args.f4_npz, allow_pickle=True)
        fg = pd.read_json(io.StringIO(str(z["groups"])), orient="records")
        fs = fg[fg.kind == "sample"].set_index("group")
        F = z["sums"]
        nmis, maxdiff, ncmp, cellmis = 0, 0.0, 0, 0
        for p in plates:
            A = np.load(P / f"plate_{p}_raw.npy", mmap_mode="r")
            g = gt[(gt.plate == p) & (gt.cell_line_id == "CVCL_0366")]
            for _, r in g.iterrows():
                if r["sample"] not in fs.index:
                    nmis += 1; continue
                a = np.asarray(A[r.row], np.float64); b = F[fs.index.get_loc(r["sample"])].astype(np.float64)
                maxdiff = max(maxdiff, float(np.abs(a[: len(b)] - b).max()))
                cellmis += int(r.n_cells != fs.at[r["sample"], "n_cells"])
                ncmp += 1
        xcheck = dict(f4_npz=args.f4_npz, snu423_wells_compared=ncmp, missing_in_f4=nmis, max_abs_count_diff=maxdiff,
                      n_cells_mismatch=cellmis)
        print("F4 cross-check", xcheck, flush=True)
    ldf.to_csv(R / "line_summary.tsv", sep="\t", index=False, float_format="%.4g")
    meta = dict(n_lines=nL, n_treated_wells=nW, n_dmso_wells=len(dmso_w), n_genes=G, min_cells=args.min_cells,
                min_cp10k=args.min_cp10k, n_drug_dose=int(wells.groupby(["drug", "dose_uM"]).ngroups),
                frac_line_well_lt_min_cells=float((NC < args.min_cells).mean()), f4_crosscheck=xcheck,
                elapsed_s=round(time.time() - t0, 1))
    (O / "prep_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
