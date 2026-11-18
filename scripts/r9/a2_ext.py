"""External prototype models under the R9 contract: same source cells/labels, same HVG gene set as the PCA pipeline of
that fold (raw counts of those genes), same test donor. Full-pipeline comparison, NOT a same-embedding head comparison.

  scpoli     : scArches 0.6.1 scPoli; condition = sample; reference training then the official query-surgery step
               (unsupervised adaptation on the test donor's RNA) -> TRANSDUCTIVE condition. Class scores =
               softmax(-distance to labelled prototypes) (classify(get_prob=True)).
  protocloud : ProtoCloud (Ding-Group, commit 90c2678), official defaults (6 prototypes/class, 100 epochs, internal
               10% random validation split, rare-type augmentation); inductive. Class scores = softmax(logits).

Usage: a2_ext.py <config.json> <method>   (R9_TASK indexes conditions x test donors x seeds)
Run with the r9ext env (R9_PY_OVERRIDE).
"""
import json
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch

from r9common import *
from r9tasks import load_obs, rep_name, select

cfg = json.loads(open(sys.argv[1]).read())
method = sys.argv[2]
run_id = cfg['run_id'] + f'_{method}'
tasks = [(c, d, s) for c in cfg['conditions'] for d in cfg['test_donors'] for s in cfg['seeds']]
cond, test, seed = tasks[int(os.environ['R9_TASK'])]
t0 = Timer()
rd = new_run_dir(run_id)
tag = f'{cond["name"]}__{test}__s{seed}'
if (rd / f'{tag}.npz').exists():
    raise SystemExit(f'immutable part exists: {tag}')
obs = load_obs()
LABELS, yi, src, te, design = select(obs, cond, test, seed)
R = np.load(CACHE / 'repr' / f'{test}__{rep_name(cond)}.npz', allow_pickle=True)
hvg = list(R['hvg_symbols'])
a = ad.read_h5ad(PROCESSED149614, backed='r')
sym = a.var.gene_symbol.astype(str).values
col = pd.Series(np.arange(len(sym)), index=sym)[hvg].values
rows = np.sort(np.concatenate([src, te]))
X = a.layers['counts'][rows][:, col] if 'counts' in a.layers else a.X[rows][:, col]
X = X.tocsr().astype(np.float32)
pos = {r: i for i, r in enumerate(rows)}
Xs, Xt = X[[pos[i] for i in src]], X[[pos[i] for i in te]]
present = np.unique(yi[src])
lab_src = np.array([LABELS[k] for k in yi[src]])
C = len(LABELS)
torch.manual_seed(seed)
np.random.seed(seed)
tm = time.time()
if method == 'scpoli':
    import scipy.sparse as _sp
    if not hasattr(_sp.csr_array, 'getnnz'):
        # scanpy 1.10.4 neighbors() calls csr_array.getnnz, removed in SciPy 1.14; restore the alias (no numeric change)
        _sp.csr_array.getnnz = lambda self, axis=None: self._getnnz(axis=axis)
    from scarches.models.scpoli import scPoli
    ref = ad.AnnData(Xs.copy(), obs=pd.DataFrame(dict(sample=obs['sample'].values[src].astype(str), label=lab_src),
                                                index=obs.index.values[src]))
    ref.var_names = hvg
    qry = ad.AnnData(Xt.copy(), obs=pd.DataFrame(dict(sample=obs['sample'].values[te].astype(str), label='unknown'),
                                                index=obs.index.values[te]))
    qry.var_names = hvg
    model = scPoli(adata=ref, condition_keys='sample', cell_type_keys='label', embedding_dims=5, latent_dim=10,
                   recon_loss='nb')
    model.train(n_epochs=50, pretraining_epochs=40, eta=5, early_stopping_kwargs=dict(
        early_stopping_metric='val_prototype_loss', mode='min', threshold=0, patience=20, reduce_lr=True,
        lr_patience=13, lr_factor=0.1), seed=seed)
    q = scPoli.load_query_data(adata=qry, reference_model=model, labeled_indices=[])
    q.train(n_epochs=50, pretraining_epochs=40, eta=10, seed=seed)
    qd = qry.copy()
    qd.X = np.asarray(qry.X.toarray())   # scArches 0.6.1 classify() uses csr.A, removed in SciPy 1.14: pass dense input
    res = q.classify(qd, scale_uncertainties=False, get_prob=True)['label']
    inv = {v: k for k, v in q.model.cell_type_encoder.items()}
    P = np.zeros((len(te), C))
    for j in range(res['weighted_distances'].shape[1]):
        P[:, LABELS.index(inv[j])] = res['weighted_distances'][:, j]
    info = dict(transductive=True, query_adaptation='scPoli load_query_data + unsupervised query training (eta=10)')
elif method == 'protocloud':
    from ProtoCloud.api import ProtoCloudModel
    ref = ad.AnnData(Xs.copy(), obs=pd.DataFrame(dict(celltype=lab_src), index=obs.index.values[src]))
    ref.var['gene_name'] = hvg
    ref.var_names = hvg
    ref.layers['counts'] = ref.X.copy()
    m = ProtoCloudModel()
    m.fit_model(ref, celltype_col='celltype', count_layer='counts', seed=seed)
    net = m._model.eval()
    dev = next(net.parameters()).device
    Xq = torch.as_tensor(Xt.toarray(), dtype=torch.float32)
    logits = []
    with torch.no_grad():
        for b in torch.split(Xq, 1000):
            logits.append(net.get_pred(b.to(dev))[0].cpu())
    L = torch.softmax(torch.cat(logits), 1).numpy()
    names = m._cell_encoder.inverse_transform(np.arange(L.shape[1]))
    P = np.zeros((len(te), C))
    for j, n in enumerate(names):
        P[:, LABELS.index(n)] = L[:, j]
    info = dict(transductive=False, defaults='num_prototypes_per_class=6 epochs=100 lr=1e-3 data_balance=True test_ratio=0.1')
else:
    raise ValueError(method)
fit_seconds = time.time() - tm
logp = np.log(np.clip(P, 1e-12, 1)).astype(np.float32)
logp[:, [k for k in range(C) if k not in present]] = -np.inf
np.savez_compressed(rd / f'{tag}.npz', cell_ids=obs.index.values[te].astype(str), truth=yi[te], models=np.array([method]),
                    labels=np.array(LABELS), train_present=present, **{f'logp__{method}': logp})
write_json(rd / f'{tag}.run.json', dict(run_id=run_id, condition=cond, test=test, seed=seed, method=method, design=design,
           n_train=int(len(src)), n_genes=len(hvg), gene_set=f'HVG of {test}__{rep_name(cond)}', labels=LABELS,
           commit=git_commit(), model_cpu_seconds={method: round(fit_seconds, 1)}, wall_fit_seconds=fit_seconds,
           gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
           peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None,
           **info, resources=t0.finish()))
print('done', tag, method, t0.finish(), flush=True)
