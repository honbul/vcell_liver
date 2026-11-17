"""Build the R8 index store. Source rows only for values; HCC rows contribute metadata (plate, well counts) and no expression.

Design: no large array is duplicated. Unit arrays are the predecessor study's (R2 `units_v1`, hash-checked) subset to the
kept cells; plate-row deltas are read from study B's processed arrays by `row_id`. This module writes index tables:

  cells.tsv        kept source cells (XC mixed pools removed): family, meta_role, inner cross-validation fold
  units.tsv        (cell, drug_group) units of kept cells with the index into the R2 unit arrays
  unit_rows.tsv    unit -> plate rows with the weight of each row in the unit mean
  rows.tsv         kept source plate rows with well-level provenance (RNA plate, bead batch, replicate tag)
  plates.tsv       (det_plate, cell): control well count, control dispersion, positional split-half check
  groups.tsv       drug groups (identical indices to R2)
  target_rows_meta.tsv   HCC plate rows: metadata only
"""
import argparse, json, os, re, time
import numpy as np
import pandas as pd

from .common import MAN, R2_MAN, B_MAN, TARGETS, proc, public, sha256, dump, env
from src.transigen_adaptation_b1.build_dataset import read_gctx_columns

POOL_FAMILY = "XC_POOLS"
N_FOLDS = 3
FOLD_SEED = 20261117


def replicate_tag(plate):
    """X-tag of a detection plate name (same rule as study R5, build.py at cb06046)."""
    m = re.search(r"_X(\d+)", plate)
    return f"X{m.group(1)}" if m else "X?"


def pert_plate(plate):
    return plate.split("_")[0]


def row_weights(rows):
    """Weight of each plate row in its unit mean: unit = equal mean over conditions (pert_id) of equal mean over plates."""
    r = rows[["cell", "drug_group", "pert_id"]]
    n_plates = r.groupby(["cell", "pert_id"]).pert_id.transform("size")
    n_cond = r.groupby(["cell", "drug_group"]).pert_id.transform("nunique")
    return 1.0 / (n_plates.to_numpy() * n_cond.to_numpy())


def half_ratio(x, a):
    """Between-half squared difference of means over its expectation under independent wells. a: boolean half membership."""
    n1, n2 = int(a.sum()), int((~a).sum())
    if n1 < 3 or n2 < 3:
        return np.nan
    d2 = ((x[a].mean(0) - x[~a].mean(0)) ** 2).mean()
    s2 = (((x[a] - x[a].mean(0)) ** 2).sum(0) + ((x[~a] - x[~a].mean(0)) ** 2).sum(0)) / (n1 + n2 - 2)
    return float(d2 / (s2.mean() * (1 / n1 + 1 / n2)))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    t0 = time.time()
    out = proc()
    assert not os.path.exists(os.path.join(out, "units.tsv")), "store already built; refusing to overwrite"
    os.makedirs(out, exist_ok=True)
    tb, r2p, raw = env("TB_PROC"), env("R2_PROC"), env("TB_RAW")
    bman = json.load(open(os.path.join(B_MAN, "build_manifest.json")))
    r2man = json.load(open(os.path.join(R2_MAN, "data_contract.json")))
    for f in ("rows.tsv", "drugs.tsv", "ecfp4.npy", "x0_all.npy", "x1_source.npy"):
        assert sha256(os.path.join(tb, f)) == bman["files"][f]["sha256"], f
    for f, e in r2man["files"].items():
        assert sha256(os.path.join(r2p, f)) == e["sha256"], f

    # ---- cells and families (R2 family table; XC mixed pools leave the main analysis)
    fam = pd.read_csv(os.path.join(R2_MAN, "source_families.tsv"), sep="\t")
    xc = fam[fam.family == POOL_FAMILY]
    assert len(xc) == 29 and xc.cell.str.startswith("XC.").all() and not fam.cell[fam.family != POOL_FAMILY].str.startswith("XC.").any()
    kept = fam[fam.family != POOL_FAMILY].sort_values("cell").reset_index(drop=True)
    cells = kept[["cell", "family", "meta_role", "cell_type", "family_reason"]].copy()
    cells.insert(0, "cell_idx", np.arange(len(cells)))
    tr_fams = sorted(set(cells.family[cells.meta_role == "train"]))
    perm = np.random.RandomState(FOLD_SEED).permutation(len(tr_fams))
    fold_of = {tr_fams[j]: int(i % N_FOLDS) for i, j in enumerate(perm)}
    cells["cv_fold"] = cells.family.map(lambda f: fold_of.get(f, -1))
    assert cells.groupby("family").meta_role.nunique().max() == 1

    # ---- plate rows
    rows_all = pd.read_csv(os.path.join(tb, "rows.tsv"), sep="\t")
    deg = set(pd.read_csv(os.path.join(B_MAN, "source_rows_excluded_degenerate.tsv"), sep="\t").row_id)
    src_all = rows_all[(rows_all.role == "source") & ~rows_all.row_id.isin(deg)]
    n_xc_rows = int(src_all.cell.isin(set(xc.cell)).sum())
    rows = src_all[src_all.cell.isin(set(cells.cell))].reset_index(drop=True)
    assert set(rows.cell) == set(cells.cell)
    trows = rows_all[rows_all.role == "target"].reset_index(drop=True)
    assert set(trows.cell) == set(TARGETS)

    # ---- well-level provenance (metadata only)
    inst = pd.read_csv(os.path.join(raw, "instinfo_beta.txt"), sep="\t", low_memory=False,
                       usecols=["sample_id", "pert_type", "cmap_name", "pert_id", "pert_time", "pert_time_unit", "nearest_dose", "pert_dose_unit",
                                "qc_pass", "det_plate", "det_well", "rna_plate", "rna_well", "bead_batch", "cell_iname"])
    t = inst[(inst.pert_type == "trt_cp") & (inst.pert_time == 24) & (inst.pert_time_unit == "h") & (inst.nearest_dose == 10)
             & (inst.pert_dose_unit == "uM") & (inst.qc_pass == 1)]
    keys = set(zip(rows_all.cell, rows_all.pert_id, rows_all.det_plate))
    t = t[[k in keys for k in zip(t.cell_iname, t.pert_id, t.det_plate)]]
    join = lambda s: "|".join(sorted(set(map(str, s))))
    tw = t.groupby(["cell_iname", "pert_id", "det_plate"]).agg(n_wells=("sample_id", "size"), det_wells=("det_well", join), rna_plates=("rna_plate", join),
                                                               rna_wells=("rna_well", join), bead_batch=("bead_batch", join)).reset_index()
    tw = tw.rename(columns={"cell_iname": "cell"})

    def add_meta(r):
        r = r.merge(tw, on=["cell", "pert_id", "det_plate"], how="left", validate="one_to_one")
        assert (r.n_wells == r.n_trt_wells).all(), "treated well count differs from study B"
        r["rep_tag"] = r.det_plate.map(replicate_tag)
        r["pert_plate"] = r.det_plate.map(pert_plate)
        return r.drop(columns=["n_wells"])

    rows, trows = add_meta(rows), add_meta(trows)
    c = inst[(inst.pert_type == "ctl_vehicle") & (inst.cmap_name == "DMSO") & (inst.pert_time == 24) & (inst.pert_time_unit == "h") & (inst.qc_pass == 1)]
    pkeys = set(zip(rows.det_plate, rows.cell)) | set(zip(trows.det_plate, trows.cell))
    c = c[[k in pkeys for k in zip(c.det_plate, c.cell_iname)]].reset_index(drop=True)
    del inst

    # ---- plates: control wells. Expression of SOURCE control wells only is read (dispersion and positional check).
    is_src_plate = np.array([k in set(zip(rows.det_plate, rows.cell)) for k in zip(c.det_plate, c.cell_iname)])
    cs = c[is_src_plate].reset_index(drop=True)
    genes = pd.read_csv(os.path.join(tb, "genes.tsv"), sep="\t", dtype=str)
    gid = genes.gene_id.astype(str).tolist() if "gene_id" in genes else genes.iloc[:, 0].astype(str).tolist()
    print(f"reading {len(cs)} source control wells", flush=True)
    xc_ = read_gctx_columns(os.path.join(raw, "level3_beta_ctl_n188708x12328.gctx"), cs.sample_id.tolist(), gid)
    x0 = np.load(os.path.join(tb, "x0_all.npy"), mmap_mode="r")
    wrow = cs.det_well.str[0].map(lambda ch: ord(ch) - ord("A")).to_numpy()
    wcol = cs.det_well.str[1:].astype(int).to_numpy()
    prec = []
    first_row = rows.groupby(["det_plate", "cell"]).row_id.first()
    for (plate, cell), idx in cs.groupby(["det_plate", "cell_iname"], sort=True).indices.items():
        x = xc_[idx]
        assert np.allclose(x.mean(0), x0[first_row[(plate, cell)]], atol=1e-4), ("X0 round trip failed", plate)
        prec.append(dict(det_plate=plate, cell=cell, n_ctl=len(idx), ctl_sd=float(np.sqrt(x.var(0, ddof=1).mean())) if len(idx) > 1 else np.nan,
                         ctl_rna_plates=join(cs.rna_plate.iloc[idx]), ctl_bead_batch=join(cs.bead_batch.iloc[idx]),
                         ctl_n_plate_rows=int(len(set(wrow[idx]))), ctl_n_plate_cols=int(len(set(wcol[idx]))),
                         half_ratio_rows=half_ratio(x, wrow[idx] < 8), half_ratio_cols=half_ratio(x, wcol[idx] <= 12),
                         half_ratio_random=half_ratio(x, np.random.RandomState(len(prec)).permutation(len(idx)) < len(idx) // 2)))
    del xc_
    plates = pd.DataFrame(prec)
    plates.insert(0, "plate_idx", np.arange(len(plates)))
    plates["rep_tag"] = plates.det_plate.map(replicate_tag)
    plates["pert_plate"] = plates.det_plate.map(pert_plate)
    rows = rows.merge(plates[["det_plate", "cell", "plate_idx", "n_ctl", "ctl_sd"]], on=["det_plate", "cell"], how="left", validate="many_to_one")
    assert (rows.n_ctl == rows.n_ctl_wells).all(), "control well count differs from study B"
    tplates = c[~is_src_plate].groupby(["det_plate", "cell_iname"]).agg(n_ctl=("sample_id", "size"), ctl_rna_plates=("rna_plate", join),
                                                                        ctl_bead_batch=("bead_batch", join)).reset_index().rename(columns={"cell_iname": "cell"})
    trows = trows.merge(tplates, on=["det_plate", "cell"], how="left", validate="many_to_one")
    assert (trows.n_ctl == trows.n_ctl_wells).all()

    # cross-check against the R5 store (read-only): same control counts and dispersion
    r5 = pd.read_csv(os.path.join(env("R5_PROC"), "rows.tsv"), sep="\t", usecols=["row_id", "n_ctl", "ctl_sd", "rep_tag"]).set_index("row_id")
    chk = r5.reindex(rows.row_id)
    assert (chk.n_ctl.to_numpy() == rows.n_ctl.to_numpy()).all() and (chk.rep_tag.to_numpy() == rows.rep_tag.to_numpy()).all()
    assert np.allclose(chk.ctl_sd.to_numpy(), rows.ctl_sd.to_numpy(), atol=1e-4, equal_nan=True)

    # ---- units: R2 unit arrays subset to kept cells
    u2 = pd.read_csv(os.path.join(r2p, "units.tsv"), sep="\t")
    g2 = pd.read_csv(os.path.join(r2p, "groups.tsv"), sep="\t")
    n_xc_units = int(u2.cell.isin(set(xc.cell)).sum())
    units = u2[u2.cell.isin(set(cells.cell))].rename(columns={"unit_idx": "r2_unit_idx"}).reset_index(drop=True)
    units["cell_idx"] = units.cell.map(cells.set_index("cell").cell_idx)
    units.insert(0, "unit_idx", np.arange(len(units)))
    rows["w"] = row_weights(rows)
    ukey = pd.Series(units.unit_idx.to_numpy(), index=pd.MultiIndex.from_frame(units[["cell", "drug_group"]]))
    rows["unit_idx"] = ukey.reindex(pd.MultiIndex.from_frame(rows[["cell", "drug_group"]])).to_numpy()
    assert rows.unit_idx.notna().all() and np.allclose(rows.groupby("unit_idx").w.sum(), 1.0)
    assert (rows.groupby("unit_idx").size().reindex(units.unit_idx).to_numpy() == units.n_rows.to_numpy()).all()
    # value round trip on a sample of units: weighted row deltas reproduce the R2 unit delta
    ud = np.load(os.path.join(r2p, "unit_delta.npy"), mmap_mode="r")
    x1 = np.load(os.path.join(tb, "x1_source.npy"), mmap_mode="r")
    for uid in np.random.RandomState(0).choice(len(units), 300, replace=False):
        rr = rows[rows.unit_idx == uid]
        d = ((np.asarray(x1[rr.row_id.to_numpy()]) - np.asarray(x0[rr.row_id.to_numpy()])) * rr.w.to_numpy()[:, None]).sum(0)
        assert np.allclose(d, ud[units.r2_unit_idx[uid]], atol=2e-4), "unit delta round trip failed"
    rows["unit_idx"] = rows.unit_idx.astype(int)
    units["n_plates"] = rows.groupby("unit_idx").plate_idx.nunique().reindex(units.unit_idx).to_numpy()
    units["n_trt_wells"] = rows.groupby("unit_idx").n_trt_wells.sum().reindex(units.unit_idx).to_numpy()
    cells["n_units"] = cells.cell_idx.map(units.groupby("cell_idx").size()).fillna(0).astype(int)
    cells["n_rows"] = cells.cell.map(rows.groupby("cell").size()).fillna(0).astype(int)
    cells["n_plates"] = cells.cell.map(plates.groupby("cell").size()).fillna(0).astype(int)

    cells.to_csv(os.path.join(out, "cells.tsv"), sep="\t", index=False)
    units.to_csv(os.path.join(out, "units.tsv"), sep="\t", index=False)
    rows[["unit_idx", "row_id", "plate_idx", "w", "n_trt_wells"]].to_csv(os.path.join(out, "unit_rows.tsv"), sep="\t", index=False)
    rows.to_csv(os.path.join(out, "rows.tsv"), sep="\t", index=False)
    plates.to_csv(os.path.join(out, "plates.tsv"), sep="\t", index=False)
    g2.to_csv(os.path.join(out, "groups.tsv"), sep="\t", index=False)
    trows.to_csv(os.path.join(out, "target_rows_meta.tsv"), sep="\t", index=False)

    # ---- public tables
    ref = cells.groupby("family").agg(cells=("cell", lambda s: "|".join(sorted(s))), n_cells=("cell", "size"), meta_role=("meta_role", "first"),
                                      cv_fold=("cv_fold", "first"), n_units=("n_units", "sum")).reset_index()
    ref["in_source_training"] = (ref.meta_role == "train").astype(int)
    ref["in_hcc_reference"] = 1
    ref.to_csv(public("manifests", "replicate_calibrated_r8", "reference_families.tsv"), sep="\t", index=False)
    xc_tab = xc[["cell", "rows", "conditions", "old_role", "meta_role"]].copy()
    xc_tab["n_units_r2"] = xc_tab.cell.map(u2.groupby("cell").size()).fillna(0).astype(int)
    xc_tab.to_csv(public("manifests", "replicate_calibrated_r8", "excluded_xc_pools.tsv"), sep="\t", index=False)
    files = ["cells.tsv", "units.tsv", "unit_rows.tsv", "rows.tsv", "plates.tsv", "groups.tsv", "target_rows_meta.tsv"]
    man = dict(track="replicate_calibrated_r8", seconds=round(time.time() - t0, 1), commit=os.environ.get("R8_COMMIT", ""),
               platform="CMap LINCS 2020 Level 3, 10 uM, 24 h, 978 landmark genes (study B processed arrays, unchanged)",
               inputs=dict(study_b_processed=dict(dir=tb, files={f: bman["files"][f] for f in ("rows.tsv", "drugs.tsv", "ecfp4.npy", "x0_all.npy", "x1_source.npy")},
                                                  build_manifest_sha256=sha256(os.path.join(B_MAN, "build_manifest.json"))),
                           r2_units=dict(dir=r2p, files=r2man["files"], origin_commit="187ed0311692ba63f8d546ca3e2ba1ff63948160"),
                           r2_family_table=dict(path=os.path.join(R2_MAN, "source_families.tsv"), sha256=sha256(os.path.join(R2_MAN, "source_families.tsv"))),
                           r5_rows_cross_check=dict(path=os.path.join(env("R5_PROC"), "rows.tsv"), sha256=sha256(os.path.join(env("R5_PROC"), "rows.tsv")),
                                                    origin_commit="cb060460057a9e659ff08667c4aa8c05dae71842", use="read-only cross-check of n_ctl, ctl_sd, rep_tag"),
                           instinfo=dict(path=os.path.join(raw, "instinfo_beta.txt"), bytes=os.path.getsize(os.path.join(raw, "instinfo_beta.txt")),
                                         sha256=sha256(os.path.join(raw, "instinfo_beta.txt"))),
                           control_level3=dict(path=os.path.join(raw, "level3_beta_ctl_n188708x12328.gctx"),
                                               bytes=os.path.getsize(os.path.join(raw, "level3_beta_ctl_n188708x12328.gctx")), sha256="not recomputed (18.6 GB, read-only)")),
               excluded=dict(xc_mixed_pools=dict(cells=int(len(xc)), plate_rows=n_xc_rows, units=n_xc_units, reason="composition not documented in the release metadata"),
                             degenerate_rows=len(deg)),
               source=dict(cells=int(len(cells)), families=int(cells.family.nunique()), meta_train_cells=int((cells.meta_role == "train").sum()),
                           meta_train_families=int(cells.family[cells.meta_role == "train"].nunique()), meta_val_cells=sorted(cells.cell[cells.meta_role == "val"]),
                           meta_val_families=int(cells.family[cells.meta_role == "val"].nunique()), plate_rows=int(len(rows)), units=int(len(units)),
                           plates=int(len(plates)), units_multi_plate=int((units.n_plates > 1).sum()), cv_folds=N_FOLDS, cv_fold_seed=FOLD_SEED,
                           cv_fold_families={str(k): int(v) for k, v in pd.Series(fold_of).value_counts().sort_index().items()}),
               target=dict(lines=list(TARGETS), expression_values_read=False, plate_rows={t_: int((trows.cell == t_).sum()) for t_ in TARGETS},
                           plates={t_: sorted(trows.det_plate[trows.cell == t_].unique()) for t_ in TARGETS}),
               files={f: dict(sha256=sha256(os.path.join(out, f)), bytes=os.path.getsize(os.path.join(out, f))) for f in files}, out_dir=out)
    dump(man, public("manifests", "replicate_calibrated_r8", "store_manifest.json"))
    print(json.dumps({k: v for k, v in man.items() if k != "inputs"}, indent=1))


if __name__ == "__main__":
    main()
