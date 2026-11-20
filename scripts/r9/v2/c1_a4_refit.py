"""R9-C1 step 2: refit the 7 final A4 heads of one (condition, seed) task (v1 saved no head checkpoints) with the same
training cells/order/seed/weights/settings as a4_external_v1, predict the Replay input and the Corrected input with the
SAME fitted head, save checkpoints, and run the committed prediction-replay check T2 against the stored v1 predictions.

Path (compat.json from c1_transform.py): 'compatible' -> source Z, Replay input = stored v1 files;
'rebuilt' -> rebuilt source Z / rebuilt v1-method external input. Corrected input always from the v2 bundle.
With --oof the script instead builds the 3-donor-group source OOF (calibration) for the task into run
a4_external_norm_v2_oof (only used when the calibration policy requires it).
Usage: c1_a4_refit.py [--oof]   (R9_TASK indexes conditions x seeds as in v1)
"""
import json
import sys

import numpy as np
import pandas as pd

from r9common import *
from r9tasks import load_obs, select
from r9v2_heads import run_models_ckpt

SPEC = json.loads((ROOT / 'configs/r9/a4_external_norm_v2.json').read_text())
T2 = SPEC['tolerances']['prediction_replay_T2']
OOF = '--oof' in sys.argv
models = SPEC['models']
tasks = [(c, s) for c in SPEC['conditions'] for s in SPEC['seeds']]
cond, seed = tasks[int(os.environ['R9_TASK'])]
tag = f'{cond["name"]}__s{seed}'
t0 = Timer()
rd = new_run_dir(SPEC['run_id'] + ('_oof' if OOF else ''))
if (rd / f'{tag}.npz').exists():
    raise SystemExit(f'immutable part exists: {tag}')
compat = json.loads((CACHE / 'r9_repr_external_norm_v2' / 'compat.json').read_text())
rep = 'full' if not cond.get('target') else f'excl_{cond["target"]}'
V1R = np.load(CACHE / 'repr_v1' / f'ALL__{rep}.npz', allow_pickle=True)
B = np.load(CACHE / 'r9_repr_external_norm_v2' / f'ALL__{rep}.npz', allow_pickle=True)
assert (V1R['ext_cell_ids'] == B['ext_cell_ids']).all()
if compat['path'] == 'compatible':
    Z, Z_rep = V1R['Z'], V1R['Z_ext']
else:
    Z, Z_rep = B['Z_source_rebuilt'], B['Z_ext_v1method_rebuilt']
Z_cor = B['Z_ext_corrected']
obs = load_obs()
donor = obs.patient.values
LABELS, yi, src, te, design = select(obs, cond, 'ALL', seed)
assert len(te) == 0 and len(LABELS) == len(BROAD)
C = len(LABELS)
v1 = np.load(RUNS / 'a4_external_v1' / f'{tag}.npz', allow_pickle=True)
v1meta = json.loads((RUNS / 'a4_external_v1' / f'{tag}.run.json').read_text())
assert v1meta['design'] == json.loads(json.dumps(design, default=str)) and v1meta['n_train'] == len(src)
assert np.array_equal(v1['src_truth'], yi[src])


def fit(tr_idx, Xte, Xextra=None):
    present = np.unique(yi[tr_idx])
    remap = {k: i for i, k in enumerate(present)}
    dn, dc = np.unique(donor[tr_idx], return_inverse=True)
    out = run_models_ckpt(Z[tr_idx], np.array([remap[k] for k in yi[tr_idx]]), dc, Xte, len(present), seed,
                          models=models, Zextra=Xextra)
    res = {}
    for m, r in out.items():
        lps = []
        for lp_ in r['logp']:
            lp = np.full((len(lp_), C), -np.inf)
            lp[:, present] = lp_
            lps.append(lp)
        res[m] = (lps, r)
    return res


save = dict(ext_cell_ids=B['ext_cell_ids'], labels=np.array(LABELS), src_truth=yi[src])
meta = dict(run_id=rd.name, condition=cond, seed=seed, design=design, n_train=int(len(src)), repr=f'ALL__{rep}',
            path=compat['path'], spec='configs/r9/a4_external_norm_v2.json', commit=git_commit(), models=models)
if OOF:
    sdon = sorted(np.unique(donor[src]))
    groups = [sdon[g::3] for g in range(3)]
    assert groups == v1meta['groups']
    oof = {m: np.zeros((len(src), C)) for m in models}
    for gd in groups:
        hold = np.isin(donor[src], gd)
        res = fit(src[~hold], Z[src[hold]])
        for m in models:
            oof[m][hold] = res[m][0][0]
    for m in models:
        save[f'oof_logp__{m}'] = oof[m].astype(np.float32)
    meta['groups'] = groups
else:
    res = fit(src, Z_rep, Z_cor)
    rep_rows = []
    ck = {}
    for m in models:
        (lp_rep, lp_cor), r = res[m]
        save[f'ext_logp_replay__{m}'] = lp_rep.astype(np.float32)
        save[f'ext_logp_corrected__{m}'] = lp_cor.astype(np.float32)
        if r['logdens'] is not None:
            save[f'ext_logdens_replay__{m}'] = r['logdens'][0].astype(np.float32)
            save[f'ext_logdens_corrected__{m}'] = r['logdens'][1].astype(np.float32)
        for k, v in r['params'].items():
            ck[f'{m}__{k}'] = np.asarray(v)
        p_new = np.exp(lp_rep.astype(np.float32).astype(np.float64))
        p_old = np.exp(v1[f'ext_logp__{m}'].astype(np.float64))
        dp = np.abs(p_new - p_old).max(1)
        agree = float((p_new.argmax(1) == p_old.argmax(1)).mean())
        p99 = float(np.quantile(dp, 0.99))
        rep_rows.append(dict(condition=cond['name'], seed=seed, model=m, path=compat['path'], argmax_agreement=agree,
                             p99_max_abs_dprob=p99, max_abs_dprob=float(dp.max()), mean_max_abs_dprob=float(dp.mean()),
                             T2_pass=bool(agree >= T2['min_argmax_agreement'] and p99 <= T2['max_p99_of_cellwise_max_abs_dprob']),
                             n_proto=int(r['alloc'].count()) if r.get('alloc') is not None else np.nan,
                             cpu_seconds=round(r['cpu_seconds'], 2)))
    np.savez_compressed(rd / f'{tag}__checkpoints.npz', **ck)
    write_tsv(pd.DataFrame(rep_rows), rd / f'{tag}__replay_check.tsv')
    meta['replay_check'] = rep_rows
    meta['task_T2_pass'] = bool(all(r['T2_pass'] for r in rep_rows))
np.savez_compressed(rd / f'{tag}.npz', **save)
meta['resources'] = t0.finish()
write_json(rd / f'{tag}.run.json', meta)
print('done', tag, meta.get('task_T2_pass'), meta['resources'], flush=True)
