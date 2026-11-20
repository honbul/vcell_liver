"""R9-E1/E2 runner (configs/r9/prototype_diagnostics_v1.json): one (sparse condition, outer test donor, seed) task.

E1: P1/P3 x {current, fixedvar} + P3_pooledvar on shared fixed allocations.
E2: donor-supported greedy with candidate logging -> A11_refit (K=3C with fallback), supported_stop (K*), matched_kstar.
Saves held-out logp for each arm, E2 memberships (for the conditional variance rerun), candidate log, diagnostics.
Usage: a2_diagnostics.py [--e2-var fixedvar|pooledvar]   (R9_TASK indexes conditions x test donors x seeds)
  --e2-var: conditional rerun of the three E2 heads with frozen per-component / pooled class variance from the saved
            memberships (run prototype_diagnostics_v1_e2<var>); only when the committed E1 condition is met.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from r9common import *
from r9model import ProtoHead, alloc_fixed, comps_from_alloc, loss_weights
from r9tasks import load_obs, select
from r9v2_heads import (FrozenVarProtoHead, alloc_fixed_counts, alloc_from_members, alloc_greedy_ds_logged,
                        proto_diagnostics)

SPEC = json.loads((ROOT / 'configs/r9/prototype_diagnostics_v1.json').read_text())
SC = SPEC['source_contract']
E2VAR = sys.argv[sys.argv.index('--e2-var') + 1] if '--e2-var' in sys.argv else None
tasks = [(c, d, s) for c in SC['conditions'] for d in SC['test_donors'] for s in SC['seeds']]
cond, test, seed = tasks[int(os.environ['R9_TASK'])]
tag = f'{cond["name"]}__{test}__s{seed}'
t0 = Timer()
run_id = SPEC['run_id'] + (f'_e2{E2VAR}' if E2VAR else '')
rd = new_run_dir(run_id)
if (rd / f'{tag}.npz').exists():
    raise SystemExit(f'immutable part exists: {tag}')
obs = load_obs()
rep = f'excl_{cond["target"]}'
R = np.load(CACHE / 'repr_v1' / f'{test}__{rep}.npz', allow_pickle=True)
assert (R['cell_ids'] == obs.index.values).all()
Z = R['Z']
donor = obs.patient.values
LABELS, yi, src, te, design = select(obs, cond, test, seed)
v1meta = json.loads((RUNS / 'a2_main_v1' / f'{tag}.run.json').read_text())
assert v1meta['design'] == json.loads(json.dumps(design)) and v1meta['n_train'] == len(src)
C = len(LABELS)
present = np.unique(yi[src])
assert len(present) == C
dnames, dcode = np.unique(donor[src], return_inverse=True)
ytr = yi[src]
Ztr, Zte, yte = Z[src], Z[te], yi[te]
w = loss_weights(ytr, dcode, 'donor', C)
tk = LABELS.index(cond['target'])
K = 3 * C


def fit_head(comps, frozen):
    torch.manual_seed(seed)
    tc = time.process_time()
    h = (FrozenVarProtoHead if frozen else ProtoHead)(comps, C).fit(Ztr, ytr, w)
    return h, time.process_time() - tc


def labels_of(al):
    lab = np.full(len(ytr), -1, np.int32)
    for j, (k, idx) in enumerate(al.members):
        lab[idx] = j
    assert (lab >= 0).all()
    return lab, np.array([k for k, _ in al.members], np.int32)


def members_from(lab, ks):
    return [(int(k), np.where(lab == j)[0]) for j, k in enumerate(ks)]


heads, allocs, cpu = {}, {}, {}
info = dict(kstar_eq_K=None)
if E2VAR is None:
    # ---------------- E1 ----------------
    tc = time.process_time()
    al1 = alloc_fixed(Ztr, ytr, dcode, C, 1, seed)
    al3 = alloc_fixed(Ztr, ytr, dcode, C, 3, seed)
    cpu['alloc_fixed'] = time.process_time() - tc
    c1, c3 = comps_from_alloc(al1, w), comps_from_alloc(al3, w)
    classvar = {c['k']: c['var'] for c in c1}
    c3p = [dict(c, var=classvar[c['k']]) for c in c3]
    for name, comps, frozen in (('P1_current', c1, False), ('P3_current', c3, False), ('P1_fixedvar', c1, True),
                                ('P3_fixedvar', c3, True), ('P3_pooledvar', c3p, True)):
        heads[name], cpu[name] = fit_head(comps, frozen)
    # ---------------- E2 ----------------
    tc = time.process_time()
    alK, snap, clog = alloc_greedy_ds_logged(Ztr, ytr, dcode, C, K, seed)
    cpu['alloc_ds_greedy'] = time.process_time() - tc
    al_ss = alK if snap is None else alloc_from_members(Ztr, ytr, dcode, C, snap)
    counts = np.bincount([k for k, _ in al_ss.members], minlength=C)
    al_mk = alloc_fixed_counts(Ztr, ytr, dcode, C, counts, seed)
    info.update(kstar=int(al_ss.count()), kstar_eq_K=snap is None, kstar_counts=counts.tolist(),
                first_fallback_step=next((r['step'] for r in alK.log if r['fallback']), None),
                n_fallback_steps=int(sum(r['fallback'] for r in alK.log)), K_final=int(alK.count()),
                target_kstar=int(counts[tk]), target_K_final=int(sum(k == tk for k, _ in alK.members)))
    write_json(rd / f'{tag}__candlog.json', dict(steps=alK.log, candidates=clog))
    allocs = dict(A11_refit=alK, supported_stop=al_ss, matched_kstar=al_mk)
    for name, al in allocs.items():
        heads[name], cpu[name] = fit_head(comps_from_alloc(al, w), False)
else:
    # ---------------- conditional E2 variance rerun from saved memberships ----------------
    M = np.load(RUNS / SPEC['run_id'] / f'{tag}__members.npz')
    for name in ('A11_refit', 'supported_stop', 'matched_kstar'):
        al = alloc_from_members(Ztr, ytr, dcode, C, members_from(M[f'{name}__lab'], M[f'{name}__k']))
        comps = comps_from_alloc(al, w)
        if E2VAR == 'pooledvar':
            al1 = alloc_fixed(Ztr, ytr, dcode, C, 1, seed)
            classvar = {c['k']: c['var'] for c in comps_from_alloc(al1, w)}
            comps = [dict(c, var=classvar[c['k']]) for c in comps]
        heads[f'{name}_{E2VAR}'], cpu[f'{name}_{E2VAR}'] = fit_head(comps, True)

# ---------------- predictions, scores, diagnostics ----------------
save = dict(cell_ids=obs.index.values[te].astype(str), truth=yte, labels=np.array(LABELS))
rows = []
v1 = np.load(RUNS / 'a2_main_v1' / f'{tag}.npz', allow_pickle=True)
assert np.array_equal(v1['cell_ids'], save['cell_ids']) and np.array_equal(v1['truth'], yte)
REF = {'P1_current': 'P1_donor', 'P3_current': 'A10', 'A11_refit': 'A11'}


def score(lp):
    p = np.exp(lp)
    pred = lp.argmax(1)
    pos = yte == tk
    return dict(target_auprc=auprc(pos, p[:, tk]) if pos.sum() >= 10 else np.nan, n_target_pos=int(pos.sum()),
                target_recall=float((pred[pos] == tk).mean()) if pos.any() else np.nan,
                target_false_pos=int(((pred == tk) & ~pos).sum()),
                nll=float(-np.clip(lp[np.arange(len(yte)), yte], -50, 0).mean()))


for name, h in heads.items():
    lp, _ = h.predict(Zte)
    save[f'logp__{name}'] = lp.astype(np.float32)
    row = dict(condition=cond['name'], test_donor=test, seed=seed, arm=name, source='v2_fit', cpu_seconds=round(cpu[name], 2),
               **score(lp), **proto_diagnostics(h, Ztr, ytr, dcode, w, Zte, yte, C, target_k=tk))
    if name in REF:
        pv = np.exp(v1[f'logp__{REF[name]}'].astype(np.float64))
        pn = np.exp(lp.astype(np.float32).astype(np.float64))
        row.update(v1_ref=REF[name], v1_argmax_agreement=float((pv.argmax(1) == pn.argmax(1)).mean()),
                   v1_max_abs_dprob=float(np.abs(pv - pn).max()))
    rows.append(row)
if E2VAR is None:
    for m in ('LR_class', 'LR_donor', 'MLP_donor', 'P1_donor', 'A10', 'A11'):
        rows.append(dict(condition=cond['name'], test_donor=test, seed=seed, arm=f'v1_{m}', source='a2_main_v1_saved',
                         **score(v1[f'logp__{m}'].astype(np.float64))))
    mem = {}
    for name, al in allocs.items():
        mem[f'{name}__lab'], mem[f'{name}__k'] = labels_of(al)
    np.savez_compressed(rd / f'{tag}__members.npz', **mem)
for r in rows:
    r.update({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in info.items()})
np.savez_compressed(rd / f'{tag}.npz', **save)
write_tsv(pd.DataFrame(rows), rd / f'{tag}__rows.tsv')
write_json(rd / f'{tag}.run.json', dict(run_id=run_id, condition=cond, test=test, seed=seed, repr=f'{test}__{rep}', design=design,
           n_train=int(len(src)), spec='configs/r9/prototype_diagnostics_v1.json', e2_var=E2VAR, info=info,
           cpu_seconds={k: round(v, 2) for k, v in cpu.items()}, commit=git_commit(), resources=t0.finish()))
print('done', tag, info, {k: round(v, 1) for k, v in cpu.items()}, t0.finish(), flush=True)
