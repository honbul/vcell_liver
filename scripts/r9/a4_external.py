"""A4: one evaluation of the frozen R9 version(s) on the separate cohort (GSE151530, HCC samples only).

Training: all 10 GSE149614 donors (representation 'ALL__full' or 'ALL__excl_<t>' for sparse conditions; HVG/scaler/PCA
fitted on GSE149614 only, GSE151530 cells only transformed). Source calibration: 3 donor-group OOF fits on GSE149614
-> temperature and three-level thresholds (EPS 0.02), identical for every model. Nothing is fitted on GSE151530.
Usage: a4_external.py <freeze_config.json>   (R9_TASK indexes conditions x seeds)
"""
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

from r9common import *
from r9arms import *
from r9tasks import load_obs, rep_name, select

cfg = json.loads(open(sys.argv[1]).read())
run_id, models = cfg['run_id'], cfg['models']
tasks = [(c, s) for c in cfg['conditions'] for s in cfg['seeds']]
cond, seed = tasks[int(os.environ['R9_TASK'])]
t0 = Timer()
rd = new_run_dir(run_id)
tag = f'{cond["name"]}__s{seed}'
if (rd / f'{tag}.npz').exists():
    raise SystemExit(f'immutable part exists: {tag}')
obs = load_obs()
R = np.load(CACHE / 'repr' / f'ALL__{rep_name(cond)}.npz', allow_pickle=True)
Z, Ze = R['Z'], R['Z_ext']
ext = ad.read_h5ad(CACHE / 'gse151530_hcc_counts.h5ad', backed='r')
assert (R['ext_cell_ids'] == ext.obs_names.values).all()
donor = obs.patient.values
LABELS, yi, src, te, design = select(obs, cond, 'ALL', seed)
assert len(te) == 0 and len(LABELS) == len(BROAD)
C = len(LABELS)


def fit_predict(tr_idx, Xte):
    present = np.unique(yi[tr_idx])
    remap = {k: i for i, k in enumerate(present)}
    dn, dc = np.unique(donor[tr_idx], return_inverse=True)
    out = run_models(Z[tr_idx], np.array([remap[k] for k in yi[tr_idx]]), dc, Xte, len(present), seed, models=models)
    res = {}
    for m, r in out.items():
        lp = np.full((len(Xte), C), -np.inf)
        lp[:, present] = r['logp'][0]
        res[m] = (lp, r['logdens'][0] if r['logdens'] is not None else None, r)
    return res


# source OOF for calibration (3 donor groups, deterministic)
sdon = sorted(np.unique(donor[src]))
groups = [sdon[g::3] for g in range(3)]
oof = {m: np.zeros((len(src), C)) for m in models}
for gd in groups:
    hold = np.isin(donor[src], gd)
    res = fit_predict(src[~hold], Z[src[hold]])
    for m in models:
        oof[m][hold] = res[m][0]
final = fit_predict(src, Ze)
save = dict(ext_cell_ids=ext.obs_names.values.astype(str), labels=np.array(LABELS), src_truth=yi[src])
rows, protos = [], []
for m in models:
    save[f'ext_logp__{m}'] = final[m][0].astype(np.float32)
    save[f'oof_logp__{m}'] = oof[m].astype(np.float32)
    if final[m][1] is not None:
        save[f'ext_logdens__{m}'] = final[m][1].astype(np.float32)
    al = final[m][2].get('alloc')
    if al is not None and m in ('A00', 'A01', 'P1_donor'):
        dn = np.unique(donor[src])
        for row in describe_alloc(al, dn):
            protos.append(dict(condition=cond['name'], seed=seed, allocation={'A00': 'fixed', 'A01': 'donor_supported', 'P1_donor': 'single'}[m],
                               class_label=LABELS[row.pop('class_idx')], **row))
np.savez_compressed(rd / f'{tag}.npz', **save)
write_tsv(pd.DataFrame(protos), rd / f'{tag}__prototypes.tsv')
write_json(rd / f'{tag}.run.json', dict(run_id=run_id, condition=cond, seed=seed, design=design, groups=groups,
           models=models, n_train=int(len(src)), repr=f'ALL__{rep_name(cond)}', freeze_config=sys.argv[1],
           commit=git_commit(), resources=t0.finish()))
print('done', tag, t0.finish(), flush=True)
