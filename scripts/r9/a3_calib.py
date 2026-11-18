"""A3: identical source calibration opportunity for every model, three-level output and pseudo-unknown abstention.

mode 'calib'  (task = condition x test donor; seed 0):
   source donors split into 3 donor groups (deterministic); out-of-fold (OOF) source predictions -> scalar temperature
   (NLL grid) and thresholds: tau_fine so that OOF accepted fine-level error <= EPS; among the rest tau_parent so that
   OOF parent-level error <= EPS; else abstain. Applied to the outer model already trained in A2 (same seed-0 run).
mode 'unknown' (task = test donor x held-out broad label u; nat condition, seed 0):
   label u removed from the source. Inner OOF folds additionally hide one pseudo-unknown label u'_g (rotating) whose
   held-out cells act as source pseudo-unknowns; the abstention threshold maximises Youden J on that source task.
   Outer model trained on source-without-u is evaluated on the test donor: u cells are the unknowns.
Usage: a3_calib.py <config.json>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from r9common import *
from r9arms import *
from r9tasks import load_obs, rep_name, select

EPS = 0.02
cfg = json.loads(open(sys.argv[1]).read())
run_id, mode, models = cfg['run_id'], cfg['mode'], cfg['models']
if mode == 'calib':
    tasks = [(c, d) for c in cfg['conditions'] for d in cfg['test_donors']]
    cond, test = tasks[int(os.environ['R9_TASK'])]
    u = None
else:
    tasks = [(d, u) for d in cfg['test_donors'] for u in BROAD]
    test, u = tasks[int(os.environ['R9_TASK'])]
    cond = dict(name='nat')
seed = 0
t0 = Timer()
rd = new_run_dir(run_id)
tag = f'{cond["name"]}__{test}' + (f'__unk_{u.replace("/", "")}' if u else '')
if (rd / f'{tag}.npz').exists():
    raise SystemExit(f'immutable part exists: {tag}')
obs = load_obs()
R = np.load(CACHE / 'repr' / f'{test}__{rep_name(cond)}.npz', allow_pickle=True)
Z = R['Z']
donor = obs.patient.values
LABELS, yi, src, te, design = select(obs, cond, test, seed)
C = len(LABELS)
if u is not None:
    src = src[yi[src] != LABELS.index(u)]
sdon = sorted(np.unique(donor[src]))
groups = [sdon[g::3] for g in range(3)]


def fit_predict(tr_idx, te_idx, ms):
    present = np.unique(yi[tr_idx])
    remap = {k: i for i, k in enumerate(present)}
    dn, dc = np.unique(donor[tr_idx], return_inverse=True)
    out = run_models(Z[tr_idx], np.array([remap[k] for k in yi[tr_idx]]), dc, Z[te_idx], len(present), seed, models=ms)
    res = {}
    for m, r in out.items():
        lp = np.full((len(te_idx), C), -np.inf)
        lp[:, present] = r['logp'][0]
        res[m] = (lp, r['logdens'][0] if r['logdens'] is not None else None)
    return res


oof = {m: [np.zeros((len(src), C)), np.full(len(src), np.nan)] for m in models}
pseudo = np.zeros(len(src), bool)
remaining = [l for l in LABELS if l != u]
for g, gd in enumerate(groups):
    hold = np.isin(donor[src], gd)
    tr = src[~hold]
    if u is not None:
        up = LABELS.index(remaining[g % len(remaining)])
        tr = tr[yi[tr] != up]
        pseudo[hold] = yi[src[hold]] == up
    res = fit_predict(tr, src[hold], models)
    for m in models:
        oof[m][0][hold] = res[m][0]
        if res[m][1] is not None:
            oof[m][1][hold] = res[m][1]
outer = fit_predict(src, te, models) if u is not None else None


def temp_scale(lp, T):
    x = lp / T
    x = np.where(np.isfinite(x), x, -np.inf)
    return x - np.logaddexp.reduce(x, axis=1, keepdims=True)


def fit_T(lp, y):
    grid = np.exp(np.linspace(np.log(0.25), np.log(8), 61))
    nll = [-temp_scale(lp, T)[np.arange(len(y)), y].clip(-50).mean() for T in grid]
    return float(grid[int(np.argmin(nll))])


parent_of = {k: PARENT.get(l, cond.get('lineage', 'lineage')) for k, l in enumerate(LABELS)}
parents = sorted(set(parent_of.values()))
pmat = np.array([[parent_of[k] == p for p in parents] for k in range(C)], float)


def thresholds(lp, y):
    p = np.exp(lp)
    conf = p.max(1)
    pred = p.argmax(1)
    err = pred != y
    order = np.argsort(-conf)
    cum = np.cumsum(err[order]) / np.arange(1, len(y) + 1)
    ok = np.where(cum <= EPS)[0]
    tau_f = conf[order][ok.max()] if len(ok) else 1.01
    rest = conf < tau_f
    pp = p @ pmat
    pconf, ppred = pp.max(1), pp.argmax(1)
    ytrue_p = np.array([parents.index(parent_of[k]) for k in y])
    o2 = np.argsort(-pconf[rest])
    perr = (ppred[rest] != ytrue_p[rest])[o2]
    cum2 = np.cumsum(perr) / np.arange(1, len(perr) + 1) if len(perr) else np.array([])
    ok2 = np.where(cum2 <= EPS)[0]
    tau_p = pconf[rest][o2][ok2.max()] if len(ok2) else 1.01
    return float(tau_f), float(tau_p)


save, rows = {}, []
if mode == 'calib':
    A2 = np.load(RUNS / cfg['a2_run'] / f'{cond["name"]}__{test}__s0.npz', allow_pickle=True)
    assert (A2['cell_ids'] == obs.index.values[te]).all()
    ysrc = yi[src]
    for m in models:
        T = fit_T(oof[m][0], ysrc)
        tf, tp = thresholds(temp_scale(oof[m][0], T), ysrc)
        lp_te = A2[f'logp__{m}'].astype(np.float64)
        save[f'oof__{m}'] = oof[m][0].astype(np.float32)
        rows.append(dict(condition=cond['name'], test_donor=test, model=m, temperature=T, tau_fine=tf, tau_parent=tp,
                         oof_nll_raw=float(-oof[m][0][np.arange(len(ysrc)), ysrc].clip(-50).mean()),
                         oof_nll_cal=float(-temp_scale(oof[m][0], T)[np.arange(len(ysrc)), ysrc].clip(-50).mean())))
    save['src_truth'] = yi[src]
else:
    ysrc = yi[src]
    known_te = yi[te] != LABELS.index(u)
    for m in models:
        T = fit_T(oof[m][0][~pseudo], ysrc[~pseudo]) if (~pseudo).any() else 1.0
        scores = {'maxprob': (np.exp(temp_scale(oof[m][0], T)).max(1), np.exp(temp_scale(outer[m][0], T)).max(1))}
        if outer[m][1] is not None:
            scores['logdens'] = (oof[m][1], outer[m][1])
        for sname, (s_oof, s_te) in scores.items():
            from sklearn.metrics import roc_auc_score
            # Youden J on the source pseudo-unknown task (higher score = more 'known')
            cand = np.quantile(s_oof, np.linspace(0, 1, 201))
            J = [((s_oof[pseudo] < c).mean() - (s_oof[~pseudo] < c).mean()) for c in cand]
            tau = float(cand[int(np.argmax(J))])
            unk = ~known_te
            pred = outer[m][0].argmax(1)
            acc_mask = s_te >= tau
            rows.append(dict(test_donor=test, unknown_label=u, model=m, score=sname, temperature=T, tau=tau,
                             src_pseudo_label=','.join(remaining[g % len(remaining)] for g in range(3)),
                             src_auroc=float(roc_auc_score(pseudo, -s_oof)) if pseudo.any() and (~pseudo).any() else np.nan,
                             n_unknown=int(unk.sum()), n_known=int(known_te.sum()),
                             test_auroc=float(roc_auc_score(unk, -s_te)) if unk.sum() >= 10 else np.nan,
                             unknown_rejected=float((~acc_mask[unk]).mean()) if unk.any() else np.nan,
                             known_rejected=float((~acc_mask[known_te]).mean()),
                             known_acc_accepted=float((pred[known_te & acc_mask] == yi[te][known_te & acc_mask]).mean()) if (known_te & acc_mask).any() else np.nan,
                             unknown_as_class=pd.Series([LABELS[k] for k in pred[unk & acc_mask]]).value_counts().to_dict() if unk.any() else {}))
        save[f'te_logp__{m}'] = outer[m][0].astype(np.float32)
        if outer[m][1] is not None:
            save[f'te_logdens__{m}'] = outer[m][1].astype(np.float32)
    save['te_truth'] = yi[te]
np.savez_compressed(rd / f'{tag}.npz', **save)
write_tsv(pd.DataFrame(rows), rd / f'{tag}.tsv')
write_json(rd / f'{tag}.run.json', dict(run_id=run_id, mode=mode, condition=cond, test=test, unknown=u, groups=groups,
                                        eps=EPS, models=models, commit=git_commit(), resources=t0.finish()))
print('done', tag, t0.finish(), flush=True)
