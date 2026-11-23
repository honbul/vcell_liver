"""Fit-time normalization and feature parity across every H3-v2 family."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh
from sklearn.decomposition import PCA

from nextgen_h2.assets import AUXILIARY, COLD, HALLMARK, KEY, LIMORE, PRIOR, TARGET, canonical_registry, parquet_lock
from nextgen_h2.model import molecular_features
from nextgen_h2.run import META, read_json
from vc_build.external import SOURCE, native_sensitivity
from vc_build.io import sha256
from vc_build.structured_kernel import fit_cell_kernels, ranked_expression
from vc_build.v0plus_cv import molecular_inputs, panel_targets
from vc_build.v0plus_neural import load_auxiliary

from .data import ASSETS, CORE, FM, INPUT_SHA, TARGETS, register


def load_registered(path):
    receipt = read_json(ASSETS / "data_receipt.json")
    return parquet_lock(ASSETS / path, receipt["files"][path])


def checked_embedding(name, ids):
    root = FM / name
    report = read_json(root / "receipt.json")
    path = root / "vectors.npz"
    if sha256(path) != report["output"]["sha256"]:
        raise ValueError(f"Invalid public chemistry embedding: {name}")
    with np.load(path, allow_pickle=False) as data:
        vectors = pd.DataFrame(data["vectors"], index=data["id"].astype(str)).reindex(ids)
    array = vectors.to_numpy(np.float32)
    return array, np.isfinite(array).all(axis=1), report["output"]["sha256"]


def normalize_fit(x, fit, floor=.05):
    ref = np.asarray(x[fit], dtype=np.float64)
    if not np.isfinite(ref).all() or len(ref) == 0:
        raise ValueError("Normalization requires finite fitting covariates")
    center, scale = ref.mean(0).astype(np.float32), ref.std(0).clip(min=floor).astype(np.float32)
    return ((x - center) / scale).clip(-8., 8.).astype(np.float32), (center, scale)


def active_scores(raw, present, fitted):
    """The stored value for an absent public view stays NaN until the branch is masked."""
    out, stamp = normalize_fit(raw, fitted[present[fitted]])
    if np.isfinite(out[~present]).any():
        raise ValueError("An unobserved pharmacology profile gained numeric features")
    return out, stamp


def geometrical_projection(train, all_samples):
    """Inductive H2 linear and radial RNA-kernel statistics with no held covariance."""
    factor = train.astype(np.float64)
    gram = factor @ factor.T
    values, vectors = eigh(gram, check_finite=False, driver="evd")
    use = values > max(1., values[-1]) * 2e-9
    x = (vectors[:, use] / np.sqrt(values[use])).T @ factor
    coords = (all_samples @ x.T).astype(np.float32)
    test = np.sum(all_samples * all_samples, axis=1, keepdims=True, dtype=np.float64)
    if coords.shape[1] < 5:
        raise ValueError("Unable to recover the fitted RNA span")
    # RBF query distance includes energy orthogonal to the fit span.
    return coords, test.astype(np.float32), x.astype(np.float32)


@dataclass
class SourceFeatures:
    registry: pd.DataFrame
    full_rna: pd.DataFrame
    legacy_rna: pd.DataFrame
    chem: pd.DataFrame
    targets: dict
    lincs: pd.DataFrame
    lincs_path: pd.DataFrame
    weights: pd.DataFrame
    coverage: pd.DataFrame
    edges: pd.DataFrame
    aux: dict

    @classmethod
    def read(cls, extra=None):
        registry = register()
        full = parquet_lock(CORE / "rna.parquet", INPUT_SHA[CORE / "rna.parquet"]).sort_index(axis=1)
        registration = read_json(COLD / "receipt.json")
        legacy = parquet_lock(COLD/"rna.parquet", registration["files"]["rna.parquet"])
        if extra is not None:
            seen = canonical_registry(extra)
            registry = pd.concat([registry, seen.loc[~seen.index.isin(registry.index)]], verify_integrity=True)
            limore = read_json(LIMORE / "receipt.json")["artifacts"]
            limore_rna = parquet_lock(LIMORE/"test_inputs/rna.parquet", limore["test_inputs/rna.parquet"]["sha256"])
            legacy = pd.concat([legacy, limore_rna.reindex(columns=legacy.columns)], verify_integrity=True)
            appendix = pd.concat([limore_rna, legacy.loc[~legacy.index.isin(full.index.union(limore_rna.index))]],
                                 verify_integrity=True)
            full = pd.concat([full, appendix.reindex(columns=full.columns)], verify_integrity=True)
        ann = read_json(TARGETS / "receipt.json")
        if sha256(TARGETS / "targets.json") != ann["targets_sha256"]:
            raise ValueError("ChEMBL exact-parent target registration changed")
        biology, _, _ = molecular_inputs(PRIOR, HALLMARK, TARGET)
        auxiliary, _ = load_auxiliary(AUXILIARY)
        weights = load_registered("pathway_weights.parquet")
        return cls(registry, full, legacy, molecular_features(registry), read_json(TARGETS/"targets.json"),
                   load_registered("lincs_modz24.parquet"), load_registered("paths_modz24.parquet"),
                   weights, load_registered("coverage.parquet"), biology, auxiliary)


def build_atlas(data, fit, query):
    cells = pd.Index(sorted(set(fit.cell_id) | set(query.cell_id)))
    drugs = pd.Index(sorted(set(fit[KEY]) | set(query[KEY])))
    ci = cells.get_indexer(sorted(fit.cell_id.unique()))
    di = drugs.get_indexer(sorted(fit[KEY].unique()))
    all_fit_cells = cells[ci]
    full = data.full_rna.loc[cells]
    full_cols = full.columns
    if full.notna().sum(axis=1).min() < 10000:
        raise ValueError("Incomplete basal RNA profile")
    # Rank only measured genes on each query before training-median imputation of platform-missing genes.
    rank = full.rank(axis=1, pct=True) - .5
    rna_median = rank.iloc[ci].median()
    if rna_median.isna().any():
        raise ValueError("Full RNA fitting axis contains an unmeasured gene")
    rank = rank.fillna(rna_median).to_numpy(np.float32)
    full_z, full_scale = normalize_fit(rank, ci)
    width = min(512, len(ci)-1)
    pca = PCA(n_components=width, whiten=True, svd_solver="randomized", random_state=3389,
              iterated_power=3).fit(full_z[ci])
    pca_out = pca.transform(full_z).clip(-8.,8.).astype(np.float32)
    # Missing RNA padding only aligns a fixed maximum neural interface width on smaller folds.
    pca_out = np.pad(pca_out, ((0, 0), (0, 512-width)))
    variability = np.argsort(-np.std(rank[ci],axis=0),kind="stable")[:512]
    known = sorted({g for name in drugs[di] for g in data.targets.get(name, []) if g in full_cols})
    selected = np.array(list(dict.fromkeys([*full_cols.get_indexer(known), *variability])), dtype=int)
    selected_rna = full_z[:,selected]

    lrna = data.legacy_rna.reindex(cells)
    lfit = lrna.loc[all_fit_cells]
    legacy_median = lfit.median()
    if legacy_median.isna().any():
        raise ValueError("H2 legacy RNA has missing fitting axes")
    basis, _ = fit_cell_kernels(lfit.fillna(legacy_median), data.edges, data.aux, device=torch.device("cpu"))
    h2_rna, activity, proxy = [t.numpy() for t in basis.vectors(lrna.fillna(legacy_median))]
    geom, energy, axes = geometrical_projection(h2_rna[ci], h2_rna)
    g_scale = np.sqrt(basis.reference_scale * h2_rna.shape[1])
    geom, energy = geom/g_scale, energy/(g_scale*g_scale)

    path_columns = data.weights.columns
    pidx = full_cols.get_indexer(data.weights.index)
    ok = pidx >= 0
    pathway, pw_scale = normalize_fit(rank[:,pidx[ok]] @ data.weights.to_numpy(np.float32)[ok], ci)
    chem = data.chem.loc[drugs].to_numpy(np.float32)
    ecpf, props = chem[:, :2048], chem[:, 2048:]
    props, chem_scale = normalize_fit(props, di)
    gene_book = sorted(set().union(*data.targets.values()))
    gene_index = {symbol: i for i,symbol in enumerate(gene_book)}
    target_vec = np.zeros((len(drugs),len(gene_book)),dtype=np.float32)
    match = np.zeros((len(cells),len(drugs),5),dtype=np.float32)
    for j, d in enumerate(drugs):
        annotated = data.targets.get(d, [])
        target_vec[j,[gene_index[g] for g in annotated]] = 1
        ix = full_cols.get_indexer(annotated)
        available = ix[ix >= 0]
        if len(available):
            x = full_z[:, available]
            match[:,j] = np.stack([x.mean(1),x.max(1),x.std(1),np.broadcast_to(len(available),len(cells)),
                                    np.broadcast_to(len(available)/len(annotated),len(cells))],axis=1)
    perturb = data.lincs.reindex(drugs).to_numpy(np.float32)
    path = data.lincs_path.reindex(index=drugs, columns=path_columns).to_numpy(np.float32)
    observed = data.coverage.reindex(drugs).lincs_modz24.eq(True).to_numpy()
    for array in (perturb,path):
        # The H2 publication registers exact profiles in the coverage table.
        if not np.array_equal(np.isfinite(array).all(1), observed):
            raise ValueError("LINCS measurement and explicit availability disagree")
    perturb, lincs_scale = active_scores(perturb,observed,di)
    path, lp_scale = active_scores(path,observed,di)

    # Standalone models share this existing fit-only anchor, including Tanimoto support for cold drugs.
    _, anchor, aligned, transforms = panel_targets(fit)
    close = ecpf @ ecpf[di].T
    similarity = close / (ecpf.sum(1)[:,None]+ecpf[di].sum(1)[None,:]-close).clip(min=1)
    nearest = np.argsort(-similarity, axis=1,kind="stable")[:,:5]
    yref = anchor.loc[drugs[di]].to_numpy()
    taper = np.take_along_axis(similarity,nearest,axis=1)**4
    den = taper.sum(1)
    mu = np.divide((yref[nearest]*taper).sum(1),den,where=den>0,out=np.full(len(drugs),yref.mean()))
    mu[di] = yref
    if full_cols.size != 17871 or lfit.shape[1] != 14421 or perturb.shape[1] != 978 or pathway.shape[1] != 44:
        raise ValueError("Feature-parity dimensions differ from the immutable inputs")
    # Export the preprocessing parameters to replay an isolated query after training.
    schema = {"full_genes": full_cols.tolist(), "legacy_genes": lrna.columns.tolist(), "full_rank_median":rna_median,
              "full_scale":full_scale, "pca":pca, "pca_width":width, "selected":selected,
              "legacy_median":legacy_median, "basis":basis, "rna_axes":axes, "rnascale":g_scale,
              "pathway_weights": data.weights, "path_scale":pw_scale, "chem_scale":chem_scale,
              "lincs_scale":lincs_scale, "lincs_path_scale":lp_scale, "targets":gene_book,
              "molecules":data.registry.loc[drugs].smiles, "transforms":transforms}
    # Rows built by panel_targets retain auditable original native identity.
    labels = aligned.set_index("row_id").loc[fit.row_id,"model_value"].to_numpy(np.float32)
    scale = max(.05,float(np.std(labels-mu[drugs.get_indexer(fit[KEY])])))
    coverage = {
        "source_rna_width":full_cols.size,"pca":width,"variable_target_genes":len(selected),
        "h2_rna_width":lfit.shape[1],"h2_fit_span":geom.shape[1],"pathway_tf":activity.shape[1],
        "rna_inferred_auxiliary":proxy.shape[1],"fingerprint":ecpf.shape[1],"rdkit":props.shape[1],
        "chembl_gene_axis":len(gene_book),"lincs":perturb.shape[1],"lincs_available":int(observed.sum()),
        "context_pathways":len(path_columns),"direct_target_summary":match.shape[-1],
        "masked_profiles_all_nan":bool(np.isnan(perturb[~observed]).all()),
        "fit_cells":len(ci),"fit_drugs":len(di),"fit_only_centering":True,
    }
    atlas = {"rna": np.concatenate([pca_out, selected_rna, geom, energy],axis=1), "activity":activity,
             "proxy":proxy, "paths":pathway, "bits":ecpf, "props":props, "targets":target_vec,
             "lincs":perturb, "perturb_paths":path, "lincs_mask":observed, "target_match":match,
             "mu":mu.astype(np.float32), "labels":labels, "scale":scale}
    return atlas,schema,cells,drugs,coverage


def positions(rows,cells,drugs):
    a,b=cells.get_indexer(rows.cell_id),drugs.get_indexer(rows[KEY])
    if min(a.min(),b.min()) < 0:
        raise ValueError("Incomplete feature support")
    return a,b


def cell_cross(atlas, a, b):
    """User-requested direct target expression and signed pathway interactions for observed pairs."""
    products = atlas["paths"][a] * atlas["perturb_paths"][b]
    valid = atlas["lincs_mask"][b]
    mean = np.mean(products,axis=1)
    high = np.max(np.abs(products),axis=1)
    length = np.linalg.norm(atlas["paths"][a],axis=1)*np.linalg.norm(atlas["perturb_paths"][b],axis=1)
    cosine = products.sum(1) / np.clip(length,1e-8,None)
    sign_agreement = (products>0).mean(1).astype(np.float32)
    sign_agreement[~valid] = np.nan
    return np.concatenate([atlas["target_match"][a,b], products,
                            np.column_stack([mean,high,cosine,sign_agreement])],axis=1)


def tabular(atlas,a,b,*,base=None,out=None,batch=16384):
    blocks_cell=("rna","activity","proxy","paths")
    blocks_drug=("bits","props","targets","lincs","perturb_paths")
    ncol=sum(atlas[k].shape[1] for k in blocks_cell+blocks_drug)+2+53+int(base is not None)
    X=np.empty((len(a),ncol),dtype=np.float32) if out is None else np.memmap(out,mode="w+",shape=(len(a),ncol),dtype=np.float32)
    for i in range(0,len(a),batch):
        part=slice(i,i+batch)
        ai,bi=a[part],b[part]
        x=[atlas[k][ai] for k in blocks_cell]+[atlas[k][bi] for k in blocks_drug]
        x.extend([atlas["lincs_mask"][bi,None],atlas["mu"][bi,None],cell_cross(atlas,ai,bi)])
        if base is not None:
            x.append(base[part,None])
        X[part]=np.concatenate(x,axis=1)
    return X
