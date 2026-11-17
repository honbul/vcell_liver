"""XPert comparator inputs on the R4 split (run with the XPert env, XP_PY).

* drug features aligned to the R4 group index: UniMol atom features and the released HG drug embedding of the XPert drug
  index, linked by structure (full InChIKey of the group's pert_ids vs InChIKey of XPert's SMILES; first-block fallback flagged).
  Groups without a link get UniMol features generated with XPert's own routine and a zero HG embedding (flag hg_missing).
* h5ad with training units ('train'), validation units ('valid') and a copy of the validation units ('test') for the
  upstream script's final report. Test units are never written. X = X0 + delta (treated profile, the upstream target),
  obsm['X_ctl'] = X0.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

from .common import ext, tb, pman, dump, sha256
from .data import Store

WORK = os.path.join(os.environ["R7_ASSETS"], "xpert_work")


def find(name):
    hits = glob.glob(ext("xpert_zenodo", "**", name), recursive=True)
    assert hits, name
    return sorted(hits, key=len)[0]


def main():
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    import anndata as ad
    S = Store()
    groups = S.groups
    drugs = pd.read_csv(tb("drugs.tsv"), sep="\t")
    smi = np.load(find("all_drugs_idx2smi_8981.npy"), allow_pickle=True).item()
    xk = {}
    for i, s in smi.items():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            xk.setdefault(Chem.MolToInchiKey(m), i)
    xb = {}
    for k, i in xk.items():
        xb.setdefault(k[:14], i)
    link, how = {}, {}
    for g, d in drugs.groupby("drug_group"):
        keys = [k for k in d.inchi_keys.fillna("") if k]
        hit = next((xk[k] for k in keys if k in xk), None)
        if hit is not None:
            link[g], how[g] = hit, "exact"; continue
        hit = next((xb[k[:14]] for k in keys if k[:14] in xb), None)
        if hit is not None:
            link[g], how[g] = hit, "connect"
    ng = len(groups)
    uni_all = np.load(find("all_drugs_unimol_arr.npy"), mmap_mode="r")
    hg_all = np.load(os.path.join(os.environ["R7_ASSETS"], "external", "XPert", "HG_data", "saved_embedding", "HG_drug_embeddings.npy"))
    F = np.zeros((ng,) + uni_all.shape[1:], np.float32)
    H = np.zeros((ng, hg_all.shape[1]), np.float32)
    flag = []
    missing = {}
    for gi, g in zip(groups.group_idx, groups.drug_group):
        if g in link:
            F[gi] = uni_all[link[g]]; H[gi] = hg_all[link[g]]
            flag.append(dict(group_idx=gi, drug_group=g, xpert_idx=link[g], link=how[g], hg_missing=0, unimol_generated=0))
        else:
            rep = drugs[drugs.pert_id == groups.rep_pert_id[gi]].iloc[0]
            missing[gi] = rep.rdkit_smiles if isinstance(rep.rdkit_smiles, str) else rep.smiles_used
            flag.append(dict(group_idx=gi, drug_group=g, xpert_idx=-1, link="none", hg_missing=1, unimol_generated=1))
    if missing:
        sys.path.insert(0, os.path.join(os.environ["R7_ASSETS"], "external", "XPert"))
        from utils import generate_unimol_feat
        feats = generate_unimol_feat(missing, max_atoms=F.shape[1])
        for gi, v in feats.items():
            F[gi] = v
    fl = pd.DataFrame(flag)
    os.makedirs(WORK, exist_ok=True)
    np.save(os.path.join(WORK, "r7_drug_unimol.npy"), F)
    np.save(os.path.join(WORK, "r7_drug_hg.npy"), H)
    fl.to_csv(os.path.join(WORK, "r7_drug_link.tsv"), sep="\t", index=False)

    # h5ad: train + valid + valid copy as 'test'
    tr, va = S.train, S.val
    idx = np.concatenate([tr, va, va])
    lab = np.array(["train"] * len(tr) + ["valid"] * len(va) + ["test"] * len(va))
    x0 = S.x0[idx].astype(np.float32)
    d = np.concatenate([S.delta(tr), S.delta(va), S.delta(va)]).astype(np.float32)
    u = S.units.iloc[idx]
    obs = pd.DataFrame(dict(pert_id=u.drug_group.to_numpy(), pert_idx=u.group_idx.to_numpy(), cell_iname=u.cell.to_numpy(),
                            cell_idx=u.cell_idx.to_numpy(), tissue_idx=0, pert_dose=10.0, pert_time=24.0, r7_split=lab, unit_idx=idx))
    obs.index = [f"r{i}" for i in range(len(obs))]
    A = ad.AnnData(X=x0 + d, obs=obs)
    A.obsm["X_ctl"] = x0
    pdir = os.path.join(WORK, "processed_data"); os.makedirs(pdir, exist_ok=True)
    # PPI gene vectors are stored in XPert's gene order (var of its example h5ad): reorder to the R4 gene axis
    xdir = os.path.join(os.environ["R7_ASSETS"], "external", "XPert", "processed_data")
    ex = ad.read_h5ad(os.path.join(xdir, "l1000_sdst_example.h5ad"), backed="r")
    xv = [str(v) for v in ex.var_names]
    gt = pd.read_csv(tb("genes.tsv"), sep="\t", dtype=str)
    key = "gene_id" if set(xv) <= set(gt.gene_id) else "gene_symbol"
    pos = {v: i for i, v in enumerate(xv)}
    order = np.array([pos[g] for g in gt[key]])
    assert len(set(order)) == 978
    ppi = np.load(os.path.join(xdir, "PPI_gene_vector_128d.npy"), allow_pickle=True)
    np.save(os.path.join(pdir, "r7_PPI_gene_vector_128d.npy"), np.asarray(ppi)[order])
    A.var_names = list(gt.gene_symbol)
    A.write_h5ad(os.path.join(pdir, "l1000_r7sdst.h5ad"))
    rec = dict(groups=ng, linked_exact=int((fl.link == "exact").sum()), linked_connect=int((fl.link == "connect").sum()), unlinked=int((fl.link == "none").sum()),
               unlinked_by_role={r: int(v) for r, v in fl.merge(S.units[["group_idx", "role"]].drop_duplicates(), on="group_idx").query("link == 'none'").role.value_counts().items()},
               h5ad_rows=dict(train=int(len(tr)), valid=int(len(va)), test_copy_of_valid=int(len(va))), test_units_written=0,
               ppi_reorder_key=key, ppi_identity_order=bool((order == np.arange(978)).all()),
               value_range=[float((x0 + d).min()), float((x0 + d).max())], unimol_shape=list(F.shape), hg_shape=list(H.shape),
               sha256={f: sha256(os.path.join(WORK, f)) for f in ["r7_drug_unimol.npy", "r7_drug_hg.npy", "r7_drug_link.tsv"]})
    dump(rec, pman("xpert_inputs.json"))
    fl.to_csv(pman("xpert_drug_link.tsv"), sep="\t", index=False)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
