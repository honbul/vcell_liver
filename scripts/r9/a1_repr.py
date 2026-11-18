"""Fixed PCA representations per outer fold (G1 recipe: CP10k+log1p, top-2000 train-variance HVG, train scaler, PCA50,
then train StandardScaler on PCs). HVG candidates are restricted to gene symbols also present in GSE151530 (schema only),
fixed before any modelling so that the development and external pipelines are identical.

Usage: a1_repr.py <fold>   fold in {HCC01..HCC10, ALL}; R9_TASK indexes the list when fold == 'task'.
For each fold builds: 'full' (all training cells) and 'excl_<t>' for t in SPARSE_TARGETS (training cells of t removed
from HVG/scaler/PCA fitting, section 7). ALL = all 10 GSE149614 donors (external A4 pipeline; also transforms GSE151530 HCC).
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from r9common import *

SPARSE_TARGETS = ['B', 'Endothelial', 'Fibroblast']
FOLDS = ['HCC01', 'HCC02', 'HCC03', 'HCC04', 'HCC05', 'HCC06', 'HCC07', 'HCC08', 'HCC09', 'HCC10', 'ALL']
fold = sys.argv[1]
if fold == 'task':
    fold = FOLDS[int(os.environ['R9_TASK'])]
t = Timer()
out = CACHE / 'repr'
out.mkdir(parents=True, exist_ok=True)


def lognorm(X):
    lib = np.asarray(X.sum(1)).ravel()
    X = X.multiply((1e4 / np.maximum(lib, 1))[:, None]).tocsr().astype(np.float32)
    X.data = np.log1p(X.data)
    return X


a = ad.read_h5ad(PROCESSED149614)
obs = pd.read_csv(CACHE / 'gse149614_obs.tsv.gz', sep='\t', index_col=0)
assert (obs.index == a.obs_names).all()
ext = ad.read_h5ad(CACHE / 'gse151530_hcc_counts.h5ad')
sym = a.var.gene_symbol.astype(str).values
ext_sym = ext.var.gene_symbol.astype(str).values
# unique-symbol intersection; duplicated symbols in either cohort are excluded (no arbitrary choice)
s149 = pd.Series(sym).value_counts()
s151 = pd.Series(ext_sym).value_counts()
shared = sorted(set(s149[s149 == 1].index) & set(s151[s151 == 1].index))
col149 = pd.Series(np.arange(len(sym)), index=sym)[shared].values
X = lognorm(a.layers['counts'][:, col149] if 'counts' in a.layers else a.X[:, col149])
patient = obs.patient.values
broad = obs.broad.values

variants = [('full', None)] + [(f'excl_{tt.replace("/", "")}', tt) for tt in SPARSE_TARGETS]
for name, target in variants:
    path = out / f'{fold}__{name}.npz'
    if path.exists():
        print('exists', path)
        continue
    train = (patient != fold) if fold != 'ALL' else np.ones(len(patient), bool)
    fitmask = train & (broad != target) if target else train
    Xf = X[fitmask]
    mu = np.asarray(Xf.mean(0)).ravel()
    var = np.asarray(Xf.power(2).mean(0)).ravel() - mu ** 2
    hvg = np.argsort(-var, kind='stable')[:2000]
    dense = X[:, hvg].toarray()
    sc = StandardScaler().fit(dense[fitmask])
    pca = PCA(n_components=50, random_state=42, svd_solver='randomized').fit(sc.transform(dense[fitmask]))
    Z = pca.transform(sc.transform(dense))
    sc2 = StandardScaler().fit(Z[fitmask])
    Z = sc2.transform(Z).astype(np.float32)
    res = dict(Z=Z, cell_ids=obs.index.values.astype(str), hvg_symbols=np.array(shared)[hvg],
               fitmask=fitmask, evr=pca.explained_variance_ratio_)
    if fold == 'ALL':
        Xe = lognorm(ext.X[:, pd.Series(np.arange(len(ext_sym)), index=ext_sym)[np.array(shared)[hvg]].values].tocsr())
        Ze = sc2.transform(pca.transform(sc.transform(Xe.toarray()))).astype(np.float32)
        res.update(Z_ext=Ze, ext_cell_ids=ext.obs_names.values.astype(str))
    np.savez(path, **res)
    print('wrote', path, Z.shape, 'fit cells', int(fitmask.sum()), 'evr50', float(pca.explained_variance_ratio_.sum()))
write_json(out / f'{fold}__meta.json', dict(fold=fold, n_shared_unique_symbols=len(shared), commit=git_commit(),
                                            resources=t.finish()))
print('done', fold, t.finish())
