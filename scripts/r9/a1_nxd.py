"""A1: N x D stress (artificially sparsified broad label). Same rare-cell count N, varying number of supporting source
donors D; non-target training pool and the outer test donor fixed. Representation: PCA fitted without any target-class
cells (section 7), so the experiment reduces both labelled and unlabelled target RNA.

Usage: a1_nxd.py <run_id>   (R9_TASK in 0..29 -> (target, outer test donor))
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from r9common import *
from r9model import *

TARGETS = ['B', 'Endothelial', 'Fibroblast']
DONORS = ['HCC01', 'HCC02', 'HCC03', 'HCC04', 'HCC05', 'HCC06', 'HCC07', 'HCC08', 'HCC09', 'HCC10']
NS, DS, ROT = [32, 64, 128], [1, 2, 4], 5
POOL_MIN = 128

run_id = sys.argv[1]
task = int(os.environ['R9_TASK'])
target, test = TARGETS[task // 10], DONORS[task % 10]
t0 = Timer()
rd = new_run_dir(run_id)
part = rd / f'{target.replace("/", "")}__{test}.tsv'
if part.exists():
    raise SystemExit(f'immutable part exists: {part}')

obs = pd.read_csv(CACHE / 'gse149614_obs.tsv.gz', sep='\t', index_col=0)
R = np.load(CACHE / 'repr' / f'{test}__excl_{target.replace("/", "")}.npz', allow_pickle=True)
assert (R['cell_ids'] == obs.index.values).all()
Z = R['Z']
C = len(BROAD)
yi = obs.broad.map({k: i for i, k in enumerate(BROAD)}).values
donor = obs.patient.values
ti = BROAD.index(target)
src = donor != test
te = donor == test
nontarget = np.where(src & (yi != ti))[0]
tcells = {d: np.where(src & (donor == d) & (yi == ti))[0] for d in DONORS if d != test}
pool = sorted([d for d, ix in tcells.items() if len(ix) >= POOL_MIN])
Zte, yte = Z[te], yi[te]
npos = int((yte == ti).sum())


def evaluate(logp, head, N, D, r, chosen, ncells):
    p = np.exp(logp)
    pred = p.argmax(1)
    pos = yte == ti
    conf = np.bincount(pred[pos], minlength=C) / max(pos.sum(), 1)
    row = dict(target=target, test_donor=test, head=head, N=N, D=D, rotation=r, donors=','.join(chosen),
               n_train_target=ncells, n_test_pos=npos, n_test=len(yte),
               auprc=auprc(pos, p[:, ti]),
               recall=float((pred[pos] == ti).mean()) if pos.any() else np.nan,
               precision=float((yte[pred == ti] == ti).mean()) if (pred == ti).any() else np.nan,
               false_pos=int(((pred == ti) & ~pos).sum()),
               brier_target=float(((p[:, ti] - pos) ** 2).mean()),
               macro_f1_all=f1_score(yte, pred, labels=[k for k in range(C) if (yte == k).any()], average='macro', zero_division=0))
    for k in range(C):
        if k != ti:
            row[f'to_{BROAD[k]}'] = float(conf[k]) if pos.any() else np.nan
    return row


rows = []
designs = [('full', 0, 0, sorted(tcells), np.concatenate(list(tcells.values())))]
for r in range(ROT):
    rng = np.random.default_rng([task, r, 9])
    order = list(rng.permutation(pool))
    perm = {d: rng.permutation(tcells[d]) for d in pool}
    for D in DS:
        chosen = order[:D]
        for N in NS:
            idx = np.concatenate([perm[d][:N // D] for d in chosen])
            designs.append(('grid', N, D, chosen, idx, r))
for des in designs:
    kind, N, D, chosen, idx = des[:5]
    r = des[5] if len(des) > 5 else -1
    tr = np.concatenate([nontarget, idx])
    Ztr, ytr, dtr = Z[tr], yi[tr], donor[tr]
    w = loss_weights(ytr, dtr, 'class', C)
    lr = fit_lr(Ztr, ytr, w, seed=0)
    lp = np.full((len(Zte), C), -np.inf)
    lp[:, lr.classes_] = np.log(np.clip(lr.predict_proba(Zte), 1e-12, 1))
    rows.append(evaluate(lp, 'LR_class', N, D, r, chosen, len(idx)))
    al = alloc_fixed(Ztr, ytr, dtr, C, 1, seed=0)
    ph = ProtoHead(comps_from_alloc(al, w), C).fit(Ztr, ytr, w)
    rows.append(evaluate(ph.predict(Zte)[0], 'P1_class', N, D, r, chosen, len(idx)))
    print(kind, N, D, r, rows[-2]['auprc'], rows[-1]['auprc'], flush=True)
df = pd.DataFrame(rows)
df['pool_donors'] = ','.join(pool)
write_tsv(df, part)
write_json(rd / f'{target.replace("/", "")}__{test}.run.json', dict(run_id=run_id, task=task, target=target, test=test,
           pool=pool, commit=git_commit(), repr=f'{test}__excl_{target}', resources=t0.finish()))
print('done', t0.finish())
