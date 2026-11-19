"""R9-C1 step 1 (private plan 2026-11-19 section 6.1.1-6.1.2): rebuild the v1 ALL-donor source transforms, check them
against the stored v1 npz with the committed tolerance T1, and build the corrected external representation.

v1 (a1_repr.py): source = CP10k+log1p over the full shared unique-symbol list, then HVG; external = HVG columns first,
then CP10k+log1p (library = HVG counts only) -> different denominators. Corrected external = align external raw counts to
the same shared list/order, CP10k+log1p over it, then the source HVG columns -> source scaler -> PCA50 -> PC scaler.
v1 saved no scaler/PCA objects, so the transform is refitted with the identical code (random_state 42, stable HVG sort)
and verified against the stored source Z and the stored (v1-method) external Z before use.

Outputs (never overwriting v1): assets/cache/r9_repr_external_norm_v2/ALL__<variant>.npz (+ bundle_meta.json, compat.json)
reports/r9_rare_state/a4_external_norm_v2/{transform_replay,implementation_checks,input_change_summary,input_change_per_pc}.tsv
"""
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from r9common import *

SPEC = json.loads((ROOT / 'configs/r9/a4_external_norm_v2.json').read_text())
T1 = SPEC['tolerances']['representation_T1']
V1 = CACHE / 'repr_v1'
OUT = CACHE / 'r9_repr_external_norm_v2'
REP = REPORT / 'a4_external_norm_v2'
VARIANTS = [('full', None), ('excl_B', 'B'), ('excl_Endothelial', 'Endothelial'), ('excl_Fibroblast', 'Fibroblast')]
t = Timer()


def lognorm(X):
    """Identical to v1 a1_repr.lognorm (CP10k over the columns of X, log1p)."""
    lib = np.asarray(X.sum(1)).ravel()
    X = X.multiply((1e4 / np.maximum(lib, 1))[:, None]).tocsr().astype(np.float32)
    X.data = np.log1p(X.data)
    return X


def shared_unique(sym_a, sym_b):
    sa, sb = pd.Series(sym_a).value_counts(), pd.Series(sym_b).value_counts()
    return sorted(set(sa[sa == 1].index) & set(sb[sb == 1].index))


def ext_v1method(Xe, ext_sym, hvg_symbols):
    """v1 external path (the bug): HVG columns first, library over HVG counts only."""
    cols = pd.Series(np.arange(len(ext_sym)), index=ext_sym)[list(hvg_symbols)].values
    return lognorm(Xe[:, cols].tocsr())


def ext_corrected(Xe, ext_sym, shared, hvg_idx):
    """Corrected external path: full shared list (source order) -> CP10k+log1p -> HVG columns."""
    cols = pd.Series(np.arange(len(ext_sym)), index=ext_sym)[list(shared)].values
    return lognorm(Xe[:, cols].tocsr())[:, hvg_idx]


a = ad.read_h5ad(PROCESSED149614)
obs = pd.read_csv(CACHE / 'gse149614_obs.tsv.gz', sep='\t', index_col=0)
assert (obs.index == a.obs_names).all()
ext = ad.read_h5ad(CACHE / 'gse151530_hcc_counts.h5ad')
sym = a.var.gene_symbol.astype(str).values
ext_sym = ext.var.gene_symbol.astype(str).values
shared = shared_unique(sym, ext_sym)
v1meta = json.loads((V1 / 'ALL__meta.json').read_text())
col149 = pd.Series(np.arange(len(sym)), index=sym)[shared].values
counts = a.layers['counts'] if 'counts' in a.layers else a.X
counts = counts.tocsr()
X = lognorm(counts[:, col149])
patient = obs.patient.values
broad = obs.broad.values
Xe = ext.X.tocsr()
lib_shared_ext = np.asarray(Xe[:, pd.Series(np.arange(len(ext_sym)), index=ext_sym)[shared].values].sum(1)).ravel()

replay_rows, change_rows, pc_rows, checks = [], [], [], []
compat = True
for name, target in VARIANTS:
    R = np.load(V1 / f'ALL__{name}.npz', allow_pickle=True)
    fitmask = (broad != target) if target else np.ones(len(patient), bool)
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

    def project(M):
        return sc2.transform(pca.transform(sc.transform(M.toarray() if hasattr(M, 'toarray') else M))).astype(np.float32)

    hvg_sym = np.array(shared)[hvg]
    Ze_v1m = project(ext_v1method(Xe, ext_sym, hvg_sym))
    Ze_cor = project(ext_corrected(Xe, ext_sym, shared, hvg))
    exact = dict(n_shared=len(shared) == v1meta['n_shared_unique_symbols'],
                 hvg_symbols=bool(np.array_equal(hvg_sym, R['hvg_symbols'])),
                 fitmask=bool(np.array_equal(fitmask, R['fitmask'])),
                 cell_ids=bool(np.array_equal(obs.index.values.astype(str), R['cell_ids'])),
                 ext_cell_ids=bool(np.array_equal(ext.obs_names.values.astype(str), R['ext_cell_ids'])))
    dz = np.abs(Z - R['Z'])
    pcr = np.array([np.corrcoef(Z[:, i], R['Z'][:, i])[0, 1] for i in range(50)])
    dze = np.abs(Ze_v1m - R['Z_ext'])
    ok = (all(exact.values()) and dz.max() <= T1['source_Z_max_abs'] and pcr.min() >= T1['source_Z_min_per_pc_pearson']
          and dze.max() <= T1['ext_Z_v1method_max_abs'])
    compat &= bool(ok)
    replay_rows.append(dict(variant=f'ALL__{name}', n_shared=len(shared), v1_n_shared=v1meta['n_shared_unique_symbols'],
                            **{f'exact_{k}': v for k, v in exact.items()}, source_Z_max_abs=float(dz.max()),
                            source_Z_mean_abs=float(dz.mean()), source_Z_min_pc_pearson=float(pcr.min()),
                            ext_Z_v1method_max_abs=float(dze.max()), ext_Z_v1method_mean_abs=float(dze.mean()),
                            evr50_rebuilt=float(pca.explained_variance_ratio_.sum()), evr50_v1=float(R['evr'].sum()),
                            T1_pass=bool(ok)))
    # input change (descriptive; no label use)
    lib_hvg_ext = np.asarray(Xe[:, pd.Series(np.arange(len(ext_sym)), index=ext_sym)[list(hvg_sym)].values].sum(1)).ravel()
    lib_hvg_src = np.asarray(counts[:, col149[hvg]].sum(1)).ravel()
    lib_shared_src = np.asarray(counts[:, col149].sum(1)).ravel()
    re_, rs_ = lib_hvg_ext / np.maximum(lib_shared_ext, 1), lib_hvg_src / np.maximum(lib_shared_src, 1)
    shift = np.linalg.norm(Ze_cor - R['Z_ext'], axis=1)
    q = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
    for nm, v in (('ext_hvg_over_shared_library_ratio', re_), ('source_hvg_over_shared_library_ratio', rs_),
                  ('ext_pca_shift_norm_corrected_vs_v1', shift), ('ext_norm_v1', np.linalg.norm(R['Z_ext'], axis=1)),
                  ('ext_norm_corrected', np.linalg.norm(Ze_cor, axis=1)), ('source_norm', np.linalg.norm(R['Z'], axis=1))):
        change_rows.append(dict(variant=f'ALL__{name}', quantity=nm, mean=float(v.mean()),
                                **{f'q{int(x * 100):02d}': float(np.quantile(v, x)) for x in q}))
    for i in range(50):
        pc_rows.append(dict(variant=f'ALL__{name}', pc=i + 1, source_mean=float(R['Z'][:, i].mean()), source_sd=float(R['Z'][:, i].std()),
                            ext_v1_mean=float(R['Z_ext'][:, i].mean()), ext_v1_sd=float(R['Z_ext'][:, i].std()),
                            ext_corrected_mean=float(Ze_cor[:, i].mean()), ext_corrected_sd=float(Ze_cor[:, i].std()),
                            mean_shift=float((Ze_cor[:, i] - R['Z_ext'][:, i]).mean()),
                            rebuilt_source_pearson=float(pcr[i])))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'ALL__{name}.npz'
    if path.exists():
        raise SystemExit(f'immutable output exists: {path}')
    np.savez(path, shared_symbols=np.array(shared), hvg_idx=hvg, hvg_symbols=hvg_sym, sc_mean=sc.mean_, sc_scale=sc.scale_,
             pca_components=pca.components_, pca_mean=pca.mean_, pca_evr=pca.explained_variance_ratio_,
             sc2_mean=sc2.mean_, sc2_scale=sc2.scale_, fitmask=fitmask, cell_ids=obs.index.values.astype(str),
             ext_cell_ids=ext.obs_names.values.astype(str), Z_source_rebuilt=Z, Z_ext_v1method_rebuilt=Ze_v1m,
             Z_ext_corrected=Ze_cor)

    # ---------------- implementation checks (variant 'full' only; transform-independent) ----------------
    if name == 'full':
        rng = np.random.default_rng(0)
        rows = np.sort(rng.choice(counts.shape[0], 2000, replace=False))
        # (1) same raw rows via the external path (permuted gene order + 3 non-shared extra genes) == source HVG values
        perm = rng.permutation(counts.shape[1])
        extra = rng.integers(0, 50, (len(rows), 3)).astype(np.float32)
        import scipy.sparse as sp
        fake = sp.hstack([counts[rows][:, perm], sp.csr_matrix(extra)]).tocsr().astype(np.float32)
        fake_sym = np.concatenate([sym[perm], ['FAKE_A', 'FAKE_B', 'FAKE_C']])
        via_ext = ext_corrected(fake, fake_sym, shared, hvg).toarray()
        via_src = X[rows][:, hvg].toarray()
        checks.append(dict(check='source rows through corrected external path (permuted genes, +3 non-shared genes) == source HVG values',
                           value=float(np.abs(via_ext - via_src).max()), passed=bool(np.abs(via_ext - via_src).max() <= 1e-6)))
        v1m = ext_v1method(fake, fake_sym, hvg_sym).toarray()
        checks.append(dict(check='same rows through the v1 external path differ from source values (bug reproduced)',
                           value=float(np.abs(v1m - via_src).max()), passed=bool(np.abs(v1m - via_src).max() > 1e-3)))
        # (2) adding counts to a shared non-HVG gene lowers every nonzero corrected HVG value; v1 path unchanged
        nonhvg = np.setdiff1d(np.arange(len(shared)), hvg)[0]
        g = np.where(fake_sym == shared[nonhvg])[0][0]
        bumped = fake[:5].tolil()
        bumped[:, g] = bumped[:, g].toarray() + 100
        bumped = bumped.tocsr()
        b_cor = ext_corrected(bumped, fake_sym, shared, hvg).toarray()
        o_cor = via_ext[:5]
        nz = o_cor > 0
        checks.append(dict(check='+100 counts on a shared non-HVG gene lowers all nonzero corrected HVG values',
                           value=float((b_cor[nz] < o_cor[nz]).mean()), passed=bool((b_cor[nz] < o_cor[nz]).all())))
        b_v1 = ext_v1method(bumped, fake_sym, hvg_sym).toarray()
        checks.append(dict(check='same bump leaves the v1 (HVG-only library) values unchanged',
                           value=float(np.abs(b_v1 - v1m[:5]).max()), passed=bool(np.abs(b_v1 - v1m[:5]).max() == 0)))
        # (3) zero-library cell
        zero = sp.csr_matrix((1, fake.shape[1]), dtype=np.float32)
        zc = ext_corrected(zero, fake_sym, shared, hvg).toarray()
        checks.append(dict(check='zero-library cell -> all-zero, finite HVG values', value=float(np.abs(zc).max()),
                           passed=bool(np.isfinite(zc).all() and np.abs(zc).max() == 0)))
        # (4) duplicate exclusion and order
        dup = pd.Series(ext_sym).value_counts()
        dup = set(dup[dup > 1].index)
        checks.append(dict(check='duplicated external symbols excluded from the shared list', value=len(dup & set(shared)),
                           passed=len(dup & set(shared)) == 0))
        checks.append(dict(check='n duplicated external symbols (info)', value=len(dup), passed=True))
        checks.append(dict(check='shared list sorted and unique; length equals v1 ALL__meta n_shared_unique_symbols',
                           value=len(shared), passed=bool(shared == sorted(set(shared)) and len(shared) == v1meta['n_shared_unique_symbols'])))
        checks.append(dict(check='symbol intersection incl. duplicates (v1 RESULTS 18,624; info)',
                           value=len(set(sym) & set(ext_sym)), passed=True))
        checks.append(dict(check='external zero-library cells over shared genes (info)', value=int((lib_shared_ext == 0).sum()), passed=True))
    print('variant', name, replay_rows[-1], flush=True)

write_tsv(pd.DataFrame(replay_rows), REP / 'transform_replay.tsv')
write_tsv(pd.DataFrame(checks), REP / 'implementation_checks.tsv')
write_tsv(pd.DataFrame(change_rows), REP / 'input_change_summary.tsv')
write_tsv(pd.DataFrame(pc_rows), REP / 'input_change_per_pc.tsv')
import numpy, scipy, sklearn
write_json(OUT / 'compat.json', dict(T1_pass=bool(compat), path='compatible' if compat else 'rebuilt', tolerances=T1,
                                     commit=git_commit()))
write_json(OUT / 'bundle_meta.json', dict(n_shared_unique_symbols=len(shared), source_h5ad=str(PROCESSED149614),
           source_sha256_expected=PROCESSED149614_SHA, ext_h5ad_sha256=sha(CACHE / 'gse151530_hcc_counts.h5ad'),
           versions=dict(numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
           threads=os.environ.get('OMP_NUM_THREADS'), random_state=42, commit=git_commit(), resources=t.finish(),
           files={f'ALL__{n}.npz': sha(OUT / f'ALL__{n}.npz') for n, _ in VARIANTS}))
print('T1 pass', compat, 'checks', pd.DataFrame(checks).passed.all(), t.finish())
