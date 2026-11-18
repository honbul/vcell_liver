"""A2: same-total-prototype 2x2 (A00/A10/A01/A11) plus section-8 baselines, outer leave-one-patient-out on GSE149614.

Usage: a2_run.py <config.json>   (R9_TASK indexes conditions x test donors x seeds)
Config keys: run_id, conditions [{name, target|null, N, D}], test_donors, seeds, models.
Condition 'nat': all training cells, full representation. Sparse condition: target class restricted to N cells from D
source donors (pool = source donors with >= 128 target cells; N/D cells each), PCA without target cells.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from r9common import *
from r9arms import *

cfg = json.loads(open(sys.argv[1]).read())
run_id = cfg['run_id']
tasks = [(c, d, s) for c in cfg['conditions'] for d in cfg['test_donors'] for s in cfg['seeds']]
cond, test, seed = tasks[int(os.environ['R9_TASK'])]
t0 = Timer()
rd = new_run_dir(run_id)
tag = f'{cond["name"]}__{test}__s{seed}'
if (rd / f'{tag}.npz').exists():
    raise SystemExit(f'immutable part exists: {tag}')

obs = pd.read_csv(CACHE / 'gse149614_obs.tsv.gz', sep='\t', index_col=0)
target = cond.get('target')
rep = 'full' if not target else f'excl_{target.replace("/", "")}'
R = np.load(CACHE / 'repr' / f'{test}__{rep}.npz', allow_pickle=True)
assert (R['cell_ids'] == obs.index.values).all()
Z = R['Z']
C = len(BROAD)
yi = obs.broad.map({k: i for i, k in enumerate(BROAD)}).values
donor = obs.patient.values
src = np.where(donor != test)[0]
te = np.where(donor == test)[0]
design = {}
if target:
    ti = BROAD.index(target)
    tc = {d: src[(donor[src] == d) & (yi[src] == ti)] for d in np.unique(donor[src])}
    pool = sorted(d for d, ix in tc.items() if len(ix) >= 128)
    rng = np.random.default_rng([BROAD.index(target), int(test[3:]), seed, cond['N'], cond['D']])
    chosen = sorted(rng.choice(pool, cond['D'], replace=False))
    idx = np.concatenate([rng.choice(tc[d], cond['N'] // cond['D'], replace=False) for d in chosen])
    src = np.concatenate([src[yi[src] != ti], idx])
    design = dict(target=target, N=cond['N'], D=cond['D'], pool=pool, chosen=chosen)
dnames, dcode = np.unique(donor[src], return_inverse=True)
out = run_models(Z[src], yi[src], dcode, Z[te], C, seed, models=cfg['models'])
save = dict(cell_ids=obs.index.values[te].astype(str), truth=yi[te], models=np.array(list(out)))
rows = []
for m, r in out.items():
    save[f'logp__{m}'] = r['logp'][0].astype(np.float32)
    if r['logdens'] is not None:
        save[f'logdens__{m}'] = r['logdens'][0].astype(np.float32)
    if r['alloc'] is not None and m in ('A00', 'A01', 'P1_class', 'TL_donor', 'RA_donor'):
        for row in describe_alloc(r['alloc'], dnames):
            rows.append(dict(condition=cond['name'], test_donor=test, seed=seed,
                             allocation={'A00': 'fixed', 'A01': 'donor_supported', 'P1_class': 'single',
                                         'TL_donor': 'training_loss', 'RA_donor': 'random'}[m],
                             class_label=BROAD[row.pop('class_idx')], **row))
        if m in ('A01', 'TL_donor', 'RA_donor'):
            write_json(rd / f'{tag}__alloclog_{m}.json', r['alloc'].log)
np.savez_compressed(rd / f'{tag}.npz', **save)
write_tsv(pd.DataFrame(rows), rd / f'{tag}__prototypes.tsv')
write_json(rd / f'{tag}.run.json', dict(run_id=run_id, condition=cond, test=test, seed=seed, repr=rep, design=design,
           n_train=int(len(src)), commit=git_commit(), models=cfg['models'], resources=t0.finish()))
print('done', tag, t0.finish(), flush=True)
