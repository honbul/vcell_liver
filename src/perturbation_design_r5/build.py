"""Build the R5 store from study B's processed rows (read-only) plus plate DMSO dispersion from the Level 3 control file.

Outputs (assets/processed/store_v1):
  rows.tsv                 kept rows (degenerate source rows removed) with representative flag, replicate tag, plate control features
  well_delta_source.npy    X1 - X0 of every kept source row (float32)
  well_delta_target.npy    same for target rows -- SEALED: only oracle.py / evaluate.py in hcc mode open it
  units_source.tsv/.npy    (cell, drug_group) unit deltas of source cells
  units_target.tsv/.npy    unit deltas of HCC lines -- SEALED
  famgroup.tsv/.npy        per (family, drug_group): equal mean over the family's cells of the unit delta
  groups.tsv, group_fp.npy representative pert_id and ECFP4 bits per drug group
  bases.npz                f0_repr PCA (meta-train f0 table) and the z whitening (meta-train within-cell-centred residuals)
"""
import argparse, json, os, re, time
import h5py
import numpy as np
import pandas as pd

from .common import env, proc, protocol, sha_hex, sha_file, dump_json, MAN, B_MAN, TARGETS, public, commit


def replicate_tag(plate):
    m = re.search(r"_X(\d+)", plate)
    return f"X{m.group(1)}" if m else "X?"


def control_features(rows, raw, meta_dir):
    """Per (det_plate, cell): n qc-pass 24 h DMSO wells (must equal study B's n_ctl_wells) and ctl_sd."""
    inst = pd.read_csv(os.path.join(raw, "instinfo_beta.txt"), sep="\t", low_memory=False,
                       usecols=["sample_id", "pert_type", "cmap_name", "pert_time", "pert_time_unit", "qc_pass", "det_plate", "cell_iname"])
    c = inst[(inst.pert_type == "ctl_vehicle") & (inst.cmap_name == "DMSO") & (inst.pert_time == 24)
             & (inst.pert_time_unit == "h") & (inst.qc_pass == 1)]
    keys = set(zip(rows.det_plate, rows.cell))
    c = c[[k in keys for k in zip(c.det_plate, c.cell_iname)]].reset_index(drop=True)
    genes = pd.read_csv(os.path.join(env("TB_PROC"), "genes.tsv"), sep="\t")
    gid = genes.gene_id.astype(str).tolist() if "gene_id" in genes else genes.iloc[:, 0].astype(str).tolist()
    from src.transigen_adaptation_b1.build_dataset import read_gctx_columns   # study B reader, unchanged
    print(f"reading {len(c)} control wells", flush=True)
    xc = read_gctx_columns(os.path.join(raw, "level3_beta_ctl_n188708x12328.gctx"), c.sample_id.tolist(), gid)
    out = []
    for (plate, cell), idx in c.groupby(["det_plate", "cell_iname"], sort=True).indices.items():
        x = xc[idx]
        sd = float(np.sqrt(x.var(0, ddof=1).mean())) if len(idx) > 1 else np.nan
        out.append(dict(det_plate=plate, cell=cell, n_ctl=len(idx), ctl_sd=sd))
    return pd.DataFrame(out), xc, c


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    t0 = time.time()
    pr = protocol()
    tb = env("TB_PROC")
    out = proc()
    os.makedirs(out, exist_ok=True)
    rows = pd.read_csv(os.path.join(tb, "rows.tsv"), sep="\t")
    deg = pd.read_csv(os.path.join(B_MAN, "source_rows_excluded_degenerate.tsv"), sep="\t")
    assert (rows.loc[deg.row_id, "det_plate"].to_numpy() == deg.det_plate.to_numpy()).all()
    keep = ~rows.row_id.isin(set(deg.row_id))
    n_src_all = int((rows.role == "source").sum())

    # ---- metadata decisions first (no values read yet)
    rows = rows[keep].reset_index(drop=True)
    rows["rep_tag"] = rows.det_plate.map(replicate_tag)
    rows["row_hash"] = [sha_hex(f"{c}|{g}|{p}|{d}") for c, g, p, d in zip(rows.cell, rows.drug_group, rows.pert_id, rows.det_plate)]
    rows["is_rep"] = 0
    for (cell, grp), idx in rows.groupby(["cell", "drug_group"], sort=False).indices.items():
        sub = rows.iloc[idx]
        cand = sub[sub.n_trt_wells == 1]
        if len(cand) == 0:
            cand = sub
        rows.loc[cand.row_hash.idxmin(), "is_rep"] = 1
    # stress-test representative: among rows of the first replicate tag of the unit
    rows["is_rep_stress"] = 0
    rows["first_tag"] = rows.groupby(["cell", "drug_group"]).rep_tag.transform("min")
    for (cell, grp), idx in rows.groupby(["cell", "drug_group"], sort=False).indices.items():
        sub = rows.iloc[idx]
        sub = sub[sub.rep_tag == sub.first_tag]
        cand = sub[sub.n_trt_wells == 1]
        if len(cand) == 0:
            cand = sub
        rows.loc[cand.row_hash.idxmin(), "is_rep_stress"] = 1

    fam = pd.read_csv(os.path.join(MAN, "source_families_from_r2.tsv"), sep="\t")
    famof = dict(zip(fam.cell, fam.family))
    roleof = dict(zip(fam.cell, fam.meta_role))
    src = rows.role == "source"
    assert set(rows.cell[src]) == set(fam.cell), "family table does not match the source cells"
    rows["family"] = rows.cell.map(lambda c: famof.get(c, c))
    rows["meta_role"] = rows.cell.map(lambda c: roleof.get(c, "hcc"))
    assert set(rows.cell[~src]) == set(TARGETS)

    # ---- plate control dispersion (control wells only)
    cf, xc, cmeta = control_features(rows, env("TB_RAW"), None)
    rows = rows.merge(cf, on=["det_plate", "cell"], how="left", validate="many_to_one")
    assert (rows.n_ctl == rows.n_ctl_wells).all(), "control well count differs from study B"
    # X0 round trip: the plate DMSO mean of the wells read here equals study B's X0
    x0 = np.load(os.path.join(tb, "x0_all.npy"), mmap_mode="r")
    gi = cmeta.groupby(["det_plate", "cell_iname"]).indices
    chk = rows.sample(200, random_state=0)
    for _, r in chk.iterrows():
        m = xc[gi[(r.det_plate, r.cell)]].mean(0)
        assert np.allclose(m, x0[r.row_id], atol=1e-4), ("X0 round trip failed", r.det_plate)
    del xc

    # ---- values
    x1s = np.load(os.path.join(tb, "x1_source.npy"), mmap_mode="r")
    sid = rows.row_id[src].to_numpy()
    wd_s = (np.asarray(x1s[sid]) - np.asarray(x0[sid])).astype(np.float32)
    x1t = np.load(os.path.join(tb, "x1_target.npy"), mmap_mode="r")
    tid = rows.row_id[~src].to_numpy()
    wd_t = (np.asarray(x1t[tid - n_src_all]) - np.asarray(x0[tid])).astype(np.float32)
    assert np.isfinite(wd_s).all() and np.isfinite(wd_t).all()
    rows["store_idx"] = -1
    rows.loc[src, "store_idx"] = np.arange(src.sum())
    rows.loc[~src, "store_idx"] = np.arange((~src).sum())
    np.save(os.path.join(out, "well_delta_source.npy"), wd_s)
    np.save(os.path.join(out, "well_delta_target.npy"), wd_t)

    def units(sub, wd):
        recs, vals = [], []
        for (cell, grp), idx in sub.groupby(["cell", "drug_group"], sort=True).indices.items():
            s = sub.iloc[idx]
            cond = []
            for pid, jdx in s.groupby("pert_id", sort=True).indices.items():
                cond.append(wd[s.store_idx.to_numpy()[jdx]].mean(0))
            vals.append(np.mean(cond, 0))
            recs.append(dict(cell=cell, drug_group=grp, family=s.family.iloc[0], meta_role=s.meta_role.iloc[0],
                             n_conditions=len(cond), n_rows=len(s), n_plates=s.det_plate.nunique()))
        u = pd.DataFrame(recs)
        u.insert(0, "unit_idx", np.arange(len(u)))
        return u, np.stack(vals).astype(np.float32)

    us, uds = units(rows[src].reset_index(drop=True), wd_s)
    ut, udt = units(rows[~src].reset_index(drop=True), wd_t)
    us.to_csv(os.path.join(out, "units_source.tsv"), sep="\t", index=False)
    np.save(os.path.join(out, "unit_delta_source.npy"), uds)
    ut.to_csv(os.path.join(out, "units_target.tsv"), sep="\t", index=False)
    np.save(os.path.join(out, "unit_delta_target.npy"), udt)

    # family x group means (source only)
    fg, fgv = [], []
    for (f, g), idx in us.groupby(["family", "drug_group"], sort=True).indices.items():
        fg.append(dict(family=f, drug_group=g, meta_role=us.meta_role.iloc[idx[0]], n_cells=len(idx)))
        fgv.append(uds[idx].mean(0))
    fg = pd.DataFrame(fg)
    fg.insert(0, "fg_idx", np.arange(len(fg)))
    fg.to_csv(os.path.join(out, "famgroup.tsv"), sep="\t", index=False)
    np.save(os.path.join(out, "famgroup.npy"), np.stack(fgv).astype(np.float32))

    # groups: representative pert_id (lexicographically first) and its ECFP4
    drugs = pd.read_csv(os.path.join(tb, "drugs.tsv"), sep="\t")
    ecfp = np.load(os.path.join(tb, "ecfp4.npy"))
    used = rows.groupby("drug_group").pert_id.min().reset_index().rename(columns={"pert_id": "rep_pert_id"})
    d_idx = dict(zip(drugs.pert_id, drugs.drug_idx))
    used["rep_drug_idx"] = used.rep_pert_id.map(d_idx).astype(int)
    used.insert(0, "group_idx", np.arange(len(used)))
    used.to_csv(os.path.join(out, "groups.tsv"), sep="\t", index=False)
    np.save(os.path.join(out, "group_fp.npy"), np.asarray(ecfp[used.rep_drug_idx.to_numpy()]).astype(bool))

    rows.drop(columns=["first_tag"]).to_csv(os.path.join(out, "rows.tsv"), sep="\t", index=False)

    # ---- bases fitted on meta-train only
    from .data import Store, f0_table
    st = Store(out, allow_target=False)
    tr_fams = sorted(set(fg.family[fg.meta_role == "train"]))
    allowed = set(tr_fams)
    G, F0, ncell = f0_table(st, allowed)                         # meta-train f0 table over groups
    mu = F0.mean(0)
    _, s, vt = np.linalg.svd(F0 - mu, full_matrices=False)
    P = vt[:32].T.astype(np.float32)
    # residuals of meta-train units against leave-own-family-out f0 (meta-train families only)
    ut_tr = st.units[st.units.meta_role == "train"]
    res = []
    for f, idx in ut_tr.groupby("family").indices.items():
        sub = ut_tr.iloc[idx]
        g2, F0f, _ = f0_table(st, allowed - {f})
        pos = pd.Series(np.arange(len(g2)), index=g2)
        ok = sub.drug_group.isin(pos.index).to_numpy()
        sub = sub[ok]
        r = st.unit_delta[sub.unit_idx.to_numpy()] - F0f[pos[sub.drug_group].to_numpy()]
        for c, jdx in sub.groupby("cell").indices.items():   # within-cell centring
            rr = r[jdx]
            res.append(rr - rr.mean(0))
    res = np.concatenate(res)
    _, s2, vt2 = np.linalg.svd(res - res.mean(0), full_matrices=False)
    lam = (s2[:32] ** 2) / (len(res) - 1)
    U = vt2[:32].T.astype(np.float32)
    zr = (res @ U) / np.sqrt(lam)
    f0r = (F0 - mu) @ P
    feat = np.concatenate([f0r, np.log1p(ncell)[:, None]], 1)
    np.savez(os.path.join(out, "bases.npz"), f0_mu=mu.astype(np.float32), f0_P=P, f0_sv=s[:32].astype(np.float32),
             feat_mean=feat.mean(0).astype(np.float32), feat_sd=(feat.std(0) + 1e-6).astype(np.float32),
             z_U=U, z_lam=lam.astype(np.float32), z_res_mean=res.mean(0).astype(np.float32),
             explained_res=np.float32((s2[:32] ** 2).sum() / (s2 ** 2).sum()), explained_f0=np.float32((s[:32] ** 2).sum() / (s ** 2).sum()))

    files = {fn: dict(sha256=sha_file(os.path.join(out, fn)), bytes=os.path.getsize(os.path.join(out, fn))) for fn in sorted(os.listdir(out))}
    summary = dict(
        commit=commit(), seconds=round(time.time() - t0, 1),
        rows_kept=int(len(rows)), source_rows=int(src.sum()), target_rows=int((~src).sum()),
        degenerate_removed=int((~keep).sum()), source_units=int(len(us)), target_units=int(len(ut)),
        famgroups=int(len(fg)), groups=int(len(used)), control_plates=int(len(cf)),
        rep_rows_multiwell=int(((rows.is_rep == 1) & (rows.n_trt_wells > 1)).sum()),
        target_units_by_line=ut.groupby("cell").size().to_dict(),
        target_rows_by_line=rows[~src].groupby("cell").size().to_dict(),
        z_explained_variance_32=float((s2[:32] ** 2).sum() / (s2 ** 2).sum()),
        f0_explained_variance_32=float((s[:32] ** 2).sum() / (s ** 2).sum()),
        ctl_sd_source=rows.loc[src, "ctl_sd"].describe().round(4).to_dict(),
        ctl_sd_by_target_plate=rows[~src].groupby(["cell", "det_plate"]).ctl_sd.first().round(4).astype(str).to_dict(),
        files=files)
    dump_json(summary, public("manifests", "perturbation_design_r5", "store_manifest.json"))
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=1, default=str))


if __name__ == "__main__":
    main()
