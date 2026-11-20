"""Second provenance check of the released XPert HG drug embedding (metadata only, no RNA).

If the embedding was trained on the default relations (PPI / DTI / DDS), its neighbourhood should be explained by those
relations. We test: for a sample of XPert drugs, are the nearest neighbours in HG space (a) DDS-linked, (b) chemically
similar, (c) target-sharing (DTI)? A high share means the embedding is consistent with structure / target pretraining.
This is evidence, not proof: it cannot exclude an unlisted relation that happens to align with structure.
"""
import glob, json, os
import numpy as np

from .common import ext, pman, dump

K = 10


def main():
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    root = os.path.join(os.environ["R7_EXT"], "xpert_zenodo", "xpert_zip_unz", "XPert", "HG_data")
    E = np.load(os.path.join(root, "saved_embedding", "HG_drug_embeddings.npy"), allow_pickle=True).astype(np.float32)
    dds = np.load(os.path.join(root, "DDS_edge_index_all_287834_pairs_weighted.npy"), allow_pickle=True).item()
    dti = np.load(os.path.join(root, "DTI_edge_index_all_12890_pairs.npy"), allow_pickle=True).item()
    smi = np.load(sorted(glob.glob(ext("xpert_zenodo", "**", "all_drugs_idx2smi_8981.npy"), recursive=True), key=len)[0], allow_pickle=True).item()
    n = E.shape[0]
    dnb, tset = {}, {}
    ei = np.asarray(dds["edge_index"])
    for a, b in ei.T:
        dnb.setdefault(int(a), set()).add(int(b)); dnb.setdefault(int(b), set()).add(int(a))
    ti = np.asarray(dti["edge_index"])
    for a, b in ti.T:
        tset.setdefault(int(a), set()).add(int(b))
    fps = {}
    for i, s in smi.items():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps[i] = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    rng = np.random.RandomState(0)
    idx = rng.choice(n, 1500, replace=False)
    dds_hit, tan_nb, tan_rand, dti_hit, dti_rand, zero = [], [], [], [], [], 0
    for i in idx:
        if not np.any(E[i]):
            zero += 1; continue
        sim = En @ En[i]; sim[i] = -np.inf
        nb = np.argpartition(-sim, K)[:K]
        nb = nb[np.argsort(-sim[nb])]
        dn = dnb.get(int(i), set())
        dds_hit.append(np.mean([int(j) in dn for j in nb]))
        if i in fps:
            tan_nb += [DataStructs.TanimotoSimilarity(fps[i], fps[int(j)]) for j in nb if int(j) in fps]
            r = rng.randint(0, n, K)
            tan_rand += [DataStructs.TanimotoSimilarity(fps[i], fps[int(j)]) for j in r if int(j) in fps and int(j) != i]
        ts = tset.get(int(i), set())
        if ts:
            dti_hit.append(np.mean([bool(tset.get(int(j), set()) & ts) for j in nb]))
            dti_rand.append(np.mean([bool(tset.get(int(j), set()) & ts) for j in rng.randint(0, n, K)]))
    res = dict(embedding_shape=list(E.shape), sampled=len(idx), all_zero_rows_sampled=zero,
               mean_frac_top10_hg_neighbours_dds_linked=float(np.mean(dds_hit)),
               dds_edge_density=float(2 * ei.shape[1] / (n * (n - 1))),
               median_tanimoto_top10_hg_neighbours=float(np.median(tan_nb)), median_tanimoto_random=float(np.median(tan_rand)),
               mean_frac_top10_sharing_a_dti_target=float(np.mean(dti_hit)) if dti_hit else None,
               mean_frac_random_sharing_a_dti_target=float(np.mean(dti_rand)) if dti_rand else None,
               drugs_with_dti=len(tset),
               interpretation="high DDS / structure / DTI agreement is consistent with pretraining on the default relations",
               caveat="evidence only; cannot exclude an undocumented relation that aligns with structure. Pretraining logs are not published.")
    dump(res, pman("xpert_hg_embedding_check.json"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
