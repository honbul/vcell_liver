"""Build plate-level (X0, X1) rows from CMap LINCS 2020 Level 3, following configs/.../protocol.yaml.

One row = (cell, pert_id, det_plate): X0 is the mean of that plate's qc-pass DMSO wells of the same
cell, X1 the mean of the qc-pass treated wells. Target X1 goes to its own file, which only the
guarded loader in data.py opens. HUH751 (a Huh-7 clone) is dropped entirely.
"""
import argparse, json, os
import h5py
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

from .tsg_adapter import sha256

RDLogger.DisableLog("rdApp.*")
LINCS_META = "/mnt/S1/sdata/pubdata/LINCS/LINCS2020"


def read_gctx_columns(path, sample_ids, gene_ids, span=4096):
    """Rows of `sample_ids` x `gene_ids` from a GCTX (HDF5) file, in the requested order."""
    with h5py.File(path, "r") as f:
        cid = f["/0/META/COL/id"][:].astype(str)
        rid = f["/0/META/ROW/id"][:].astype(str)
        m = f["/0/DATA/0/matrix"]
        assert m.shape == (len(cid), len(rid)), (m.shape, len(cid), len(rid))
        cpos = pd.Series(np.arange(len(cid)), index=cid)
        assert cpos.index.is_unique, "duplicated sample ids in the matrix"
        idx = cpos.reindex(sample_ids).to_numpy()
        assert not np.isnan(idx).any(), "some requested wells are not in the matrix"
        idx = idx.astype(np.int64)
        gser = pd.Series(np.arange(len(rid)), index=rid)
        gpos = gser.reindex(gene_ids).to_numpy()
        assert not np.isnan(gpos).any(), "some landmark genes are not in the matrix"
        gpos = gpos.astype(np.int64)
        g_lo, g_hi = gpos.min(), gpos.max() + 1
        order = np.argsort(idx, kind="stable")
        sidx = idx[order]
        out = np.empty((len(idx), len(gene_ids)), dtype=np.float32)
        start = 0
        while start < len(sidx):
            lo = sidx[start]
            end = int(np.searchsorted(sidx, lo + span, side="left"))
            hi = sidx[end - 1] + 1
            block = m[lo:hi, g_lo:g_hi]
            out[order[start:end]] = block[sidx[start:end] - lo][:, gpos - g_lo]
            start = end
        # round-trip: the ids at the positions we read are the ids we asked for
        assert (cid[idx] == np.asarray(sample_ids)).all()
        assert (rid[gpos] == np.asarray(gene_ids)).all()
    return out


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def drug_table(pert_ids, inst_names):
    comp = pd.read_csv(os.path.join(LINCS_META, "compoundinfo_beta.txt"), sep="\t", dtype=str).fillna("")
    comp = comp[comp.pert_id.isin(pert_ids)]
    rows, fps = [], {}
    for pid in sorted(pert_ids):
        c = comp[comp.pert_id == pid]
        smiles = sorted({s for s in c.canonical_smiles if s and s.lower() not in ("restricted", "nan", "-666")})
        mol, used = None, ""
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is not None and mol.GetNumAtoms() > 0:
                used = s
                break
            mol = None
        names = sorted({n.lower() for n in c.cmap_name if n} | {n.lower() for n in inst_names.get(pid, []) if n})
        inchis = sorted({k for k in c.inchi_key if k})
        if mol is None:
            rows.append(dict(pert_id=pid, cmap_name=";".join(names), smiles_used="", rdkit_smiles="", inchi_keys=";".join(inchis),
                             n_distinct_smiles=len(smiles), has_structure=0))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps[pid] = arr
        rows.append(dict(pert_id=pid, cmap_name=";".join(names), smiles_used=used, rdkit_smiles=Chem.MolToSmiles(mol),
                         inchi_keys=";".join(inchis), n_distinct_smiles=len(smiles), has_structure=1))
    d = pd.DataFrame(rows)
    uf = UnionFind()
    for r in d[d.has_structure == 1].itertuples():
        uf.find(r.pert_id)
        for n in r.cmap_name.split(";"):
            if n:
                uf.union(r.pert_id, "name:" + n)
        for k in r.inchi_keys.split(";"):
            if k:
                uf.union(r.pert_id, "inchi:" + k)
        uf.union(r.pert_id, "smi:" + r.rdkit_smiles)
        uf.union(r.pert_id, "fp:" + fps[r.pert_id].tobytes().hex())
    roots = {r.pert_id: uf.find(r.pert_id) for r in d[d.has_structure == 1].itertuples()}
    # group id = smallest pert_id in the component
    comp_min = {}
    for pid, root in roots.items():
        comp_min[root] = min(comp_min.get(root, pid), pid)
    d["drug_group"] = [("G:" + comp_min[roots[p]]) if p in roots else "" for p in d.pert_id]
    return d, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/transigen_adaptation_b1/protocol.yaml")
    ap.add_argument("--manifest-dir", default="manifests/transigen_adaptation_b1")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    assert not os.path.exists(os.path.join(a.out, "build_manifest.json")), "dataset already built; refusing to overwrite"
    os.makedirs(a.out, exist_ok=True)
    targets = sorted(cfg["targets"])
    excluded = set(cfg["excluded_from_source_all_conditions"])
    assert set(targets) <= excluded

    genes = pd.read_csv(os.path.join(LINCS_META, "geneinfo_beta.txt"), sep="\t", dtype=str)
    lm = genes[genes.feature_space == "landmark"].reset_index(drop=True)
    assert len(lm) == 978 and lm.gene_id.is_unique

    inst = pd.read_csv(os.path.join(a.raw, "instinfo_beta.txt"), sep="\t", low_memory=False)
    assert inst.sample_id.is_unique, "an instance appears twice in the release"
    attr = []

    def note(step, df):
        row = dict(step=step, wells=len(df), cells=df.cell_iname.nunique(), pert_ids=df.pert_id.nunique())
        for t in targets + ["HUH751"]:
            s = df[df.cell_iname == t]
            row[f"{t}_wells"] = len(s)
            row[f"{t}_pert_ids"] = s.pert_id.nunique()
        attr.append(row)

    t = inst[inst.pert_type == "trt_cp"]
    note("trt_cp", t)
    t = t[(t.pert_time == 24) & (t.pert_time_unit == "h")]
    note("24 h", t)
    t = t[(t.nearest_dose == 10) & (t.pert_dose_unit == "uM")]
    note("nearest_dose 10 uM", t)
    t = t[t.qc_pass == 1]
    note("qc_pass", t)
    c = inst[(inst.pert_type == "ctl_vehicle") & (inst.cmap_name == "DMSO") & (inst.pert_time == 24)
             & (inst.pert_time_unit == "h") & (inst.qc_pass == 1)]
    ckeys = set(zip(c.det_plate, c.cell_iname))
    t = t[[k in ckeys for k in zip(t.det_plate, t.cell_iname)]]
    note("same-plate same-cell qc-pass DMSO", t)

    inst_names = t.groupby("pert_id").cmap_name.agg(lambda s: sorted(set(map(str, s)))).to_dict()
    drugs, fps = drug_table(set(t.pert_id), inst_names)
    ok = set(drugs.pert_id[drugs.has_structure == 1])
    t = t[t.pert_id.isin(ok)]
    note("parsable structure", t)
    t = t[t.cell_iname != "HUH751"]
    note("drop HUH751 (Huh-7 clone, neither source nor target)", t)
    pd.DataFrame(attr).to_csv(os.path.join(a.manifest_dir, "attrition.tsv"), sep="\t", index=False)

    c = c[[k in set(zip(t.det_plate, t.cell_iname)) for k in zip(c.det_plate, c.cell_iname)]]
    t = t.assign(role=np.where(t.cell_iname.isin(targets), "target", "source"))
    assert not (set(t.cell_iname[t.role == "source"]) & excluded)

    # ---- expression
    gene_ids = lm.gene_id.tolist()
    xt = read_gctx_columns(os.path.join(a.raw, "level3_beta_trt_cp_n1805898x12328.gctx"), t.sample_id.tolist(), gene_ids)
    xc = read_gctx_columns(os.path.join(a.raw, "level3_beta_ctl_n188708x12328.gctx"), c.sample_id.tolist(), gene_ids)
    assert np.isfinite(xt).all() and np.isfinite(xc).all()

    t = t.reset_index(drop=True)
    c = c.reset_index(drop=True)
    cg = c.groupby(["det_plate", "cell_iname"], sort=True).indices
    plate_x0 = {k: xc[v].mean(0) for k, v in cg.items()}
    plate_nctl = {k: len(v) for k, v in cg.items()}
    tg = t.groupby(["role", "cell_iname", "pert_id", "det_plate"], sort=True).indices
    keys = sorted(tg, key=lambda k: (k[0] != "source", k[1], k[2], k[3]))   # source rows first
    grp = drugs.set_index("pert_id").drug_group.to_dict()
    rows, x0, x1 = [], [], []
    for role, cell, pid, plate in keys:
        v = tg[(role, cell, pid, plate)]
        x1.append(xt[v].mean(0))
        x0.append(plate_x0[(plate, cell)])
        rows.append(dict(role=role, cell=cell, pert_id=pid, drug_group=grp[pid], det_plate=plate,
                         n_trt_wells=len(v), n_ctl_wells=plate_nctl[(plate, cell)]))
    rows = pd.DataFrame(rows)
    rows.insert(0, "row_id", np.arange(len(rows)))
    x0 = np.stack(x0).astype(np.float32)
    x1 = np.stack(x1).astype(np.float32)
    n_src = int((rows.role == "source").sum())
    assert (rows.role.iloc[:n_src] == "source").all() and (rows.role.iloc[n_src:] == "target").all()

    used = drugs[drugs.pert_id.isin(set(rows.pert_id))].reset_index(drop=True)
    used.insert(0, "drug_idx", np.arange(len(used)))
    fp = np.stack([fps[p] for p in used.pert_id])
    rows["drug_idx"] = rows.pert_id.map(used.set_index("pert_id").drug_idx).astype(int)
    # round-trip of the id dictionary used by training and evaluation alike
    assert (used.pert_id.to_numpy()[rows.drug_idx.to_numpy()] == rows.pert_id.to_numpy()).all()

    np.save(os.path.join(a.out, "x0_all.npy"), x0)
    np.save(os.path.join(a.out, "x1_source.npy"), x1[:n_src])
    np.save(os.path.join(a.out, "x1_target.npy"), x1[n_src:])
    np.save(os.path.join(a.out, "ecfp4.npy"), fp)
    rows.to_csv(os.path.join(a.out, "rows.tsv"), sep="\t", index=False)
    used.to_csv(os.path.join(a.out, "drugs.tsv"), sep="\t", index=False)
    lm[["gene_id", "gene_symbol", "ensembl_id"]].to_csv(os.path.join(a.out, "genes.tsv"), sep="\t", index=False)
    drugs[drugs.has_structure == 0].to_csv(os.path.join(a.manifest_dir, "dropped_no_structure.tsv"), sep="\t", index=False)
    lm[["gene_id", "gene_symbol", "ensembl_id"]].to_csv(os.path.join(a.manifest_dir, "genes_landmark_978.tsv"), sep="\t", index=False)
    rows[rows.role == "target"].to_csv(os.path.join(a.manifest_dir, "target_rows.tsv"), sep="\t", index=False)
    src = rows[rows.role == "source"]
    sc = src.groupby("cell").agg(rows=("row_id", "size"), conditions=("pert_id", "nunique"), plates=("det_plate", "nunique")).reset_index()
    sc["liver_non_hcc"] = sc.cell.isin(cfg["source_liver_non_hcc_flagged"]).astype(int)
    sc.to_csv(os.path.join(a.manifest_dir, "source_cells.tsv"), sep="\t", index=False)

    files = ["x0_all.npy", "x1_source.npy", "x1_target.npy", "ecfp4.npy", "rows.tsv", "drugs.tsv", "genes.tsv"]
    man = dict(protocol_variant=cfg["protocol_variant"], n_rows=len(rows), n_source_rows=n_src, n_target_rows=len(rows) - n_src,
               n_source_cells=int(src.cell.nunique()), n_source_conditions=int(src.groupby(["cell", "pert_id"]).ngroups),
               n_drugs=len(used), n_drug_groups=int(used.drug_group.nunique()),
               value_range_x0=[float(x0.min()), float(x0.max())], value_range_x1=[float(x1.min()), float(x1.max())],
               value_mean_x0=float(x0.mean()), value_mean_x1=float(x1.mean()),
               targets={t_: dict(rows=int((rows.cell == t_).sum()), conditions=int(rows[rows.cell == t_].pert_id.nunique()),
                                 drug_groups=int(rows[rows.cell == t_].drug_group.nunique()),
                                 plates=int(rows[rows.cell == t_].det_plate.nunique())) for t_ in targets},
               files={f: dict(sha256=sha256(os.path.join(a.out, f)), bytes=os.path.getsize(os.path.join(a.out, f))) for f in files},
               out_dir=a.out)
    json.dump(man, open(os.path.join(a.manifest_dir, "build_manifest.json"), "w"), indent=1)
    json.dump(man, open(os.path.join(a.out, "build_manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
