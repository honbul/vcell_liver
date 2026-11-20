"""Provenance check of the released XPert HG inputs (metadata only, no RNA): are the drug-drug (DDS) edge weights explained by
chemical structure? Compares DDS weights with ECFP4 Tanimoto of the same XPert drug pairs and with random pairs; inspects DTI size."""
import glob, json, os
import numpy as np

from .common import ext, pman, dump


def main():
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    hg = os.path.join(os.environ["R7_EXT"], "xpert_zenodo", "xpert_zip_unz", "XPert", "HG_data")
    smi = np.load(sorted(glob.glob(ext("xpert_zenodo", "**", "all_drugs_idx2smi_8981.npy"), recursive=True), key=len)[0], allow_pickle=True).item()
    n = len(smi)
    fps = {}
    for i, s in smi.items():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps[i] = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    d = np.load(os.path.join(hg, "DDS_edge_index_all_287834_pairs_weighted.npy"), allow_pickle=True).item()
    ei, w = np.asarray(d["edge_index"]), np.asarray(d["edge_attr"], float)
    rng = np.random.RandomState(0)
    k = rng.choice(ei.shape[1], min(20000, ei.shape[1]), replace=False)
    t_e = np.array([DataStructs.TanimotoSimilarity(fps[a], fps[b]) if a in fps and b in fps else np.nan for a, b in ei[:, k].T])
    ra, rb = rng.randint(0, n, 20000), rng.randint(0, n, 20000)
    t_r = np.array([DataStructs.TanimotoSimilarity(fps[a], fps[b]) if a in fps and b in fps and a != b else np.nan for a, b in zip(ra, rb)])
    ok = np.isfinite(t_e)
    from scipy.stats import spearmanr, pearsonr
    dti = np.load(os.path.join(hg, "DTI_edge_index_all_12890_pairs.npy"), allow_pickle=True).item()
    res = dict(dds_edges=int(ei.shape[1]), dds_index_range=[int(ei.min()), int(ei.max())], xpert_drugs=n,
               dds_weight_quantiles={q: float(np.quantile(w, q)) for q in [0, 0.1, 0.5, 0.9, 1]},
               sampled_edges=int(ok.sum()),
               tanimoto_on_dds_edges_median=float(np.nanmedian(t_e)), tanimoto_random_pairs_median=float(np.nanmedian(t_r)),
               frac_dds_edges_tanimoto_ge_0p3=float(np.nanmean(t_e >= 0.3)), frac_random_tanimoto_ge_0p3=float(np.nanmean(t_r >= 0.3)),
               spearman_weight_vs_tanimoto=float(spearmanr(w[k][ok], t_e[ok]).correlation),
               pearson_weight_vs_tanimoto=float(pearsonr(w[k][ok], t_e[ok])[0]),
               dti_edges=int(np.asarray(dti["edge_index"]).shape[1]),
               note="metadata only; ECFP4 (Morgan r2, 2048) Tanimoto computed here; DDS definition is not documented in the XPert repository")
    dump(res, pman("xpert_hg_provenance_check.json"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
