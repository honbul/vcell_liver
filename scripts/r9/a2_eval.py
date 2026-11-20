"""Score A2 runs (development evidence, outer LOPO on GSE149614).

Per (condition, test donor, seed, model): per-label AUPRC (NA without positives), macro-F1, Brier, NLL, rare->other
confusion. Seeds are technical repeats: averaged within donor; donor is the independent unit (paired differences).
Primary metric by condition kind:
  nat       : donor-macro all-label AUPRC (no natural-rare broad label exists; reported as all-label)
  sparse_*  : target-label AUPRC (artificially sparsified)
  lin_*     : natural-rare multi-donor cluster AUPRC (tier 2, cluster-ID reproduction; fold-wise rare definition)
Usage: a2_eval.py <run_id> [<run_id> ...]  -> reports/r9_rare_state/a2_* tables
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from r9common import *

CONTRASTS = [('A11', 'A10', 'core: adaptive vs fixed allocation under donor-balanced loss'),
             ('A11', 'A01', 'donor-balanced vs class-balanced loss under adaptive allocation'),
             ('A10', 'A00', 'donor-balanced vs class-balanced loss under fixed allocation'),
             ('A01', 'A00', 'adaptive vs fixed under class-balanced loss'),
             ('A11', 'TL_donor', 'donor-supported vs training-loss allocation (same K, donor loss)'),
             ('A11', 'RA_donor', 'donor-supported vs random allocation (same K, donor loss)'),
             ('A11', 'GMM_K', 'adaptive discriminative vs generative GMM (same K)'),
             ('A10', 'P1_donor', 'fixed multi-prototype vs single prototype (donor loss)'),
             ('A11', 'LR_donor', 'A11 vs donor-balanced logistic'),
             ('A10', 'LR_donor', 'A10 vs donor-balanced logistic (selection rule)'),
             ('A11', 'LR_class', 'A11 vs class-balanced logistic'),
             ('A11', 'MLP_donor', 'A11 vs donor-balanced MLP'),
             ('LR_donor', 'LR_class', 'donor- vs class-balanced logistic'),
             ('LR_class', 'LR_cell', 'class-balanced vs unweighted logistic')]
rare_def = pd.read_csv(REPORT / 'rare_definition_by_fold.tsv', sep='\t')


def label_sets(cond, test, labels, present):
    """Which labels form the primary set for this (condition, outer test donor), from training donors only."""
    if cond.get('lineage'):
        r = rare_def[(rare_def.outer_test_donor == test) & (rare_def.tier == 'res3_cluster') & rare_def.label.isin(labels)].set_index('label')
        multi = [l for l in labels if l in r.index and r.loc[l, 'rare_1'] and r.loc[l, 'train_donors_ge10'] >= 3]
        few = [l for l in labels if l in r.index and r.loc[l, 'rare_1'] and r.loc[l, 'train_donors_ge10'] == 2]
        single = [l for l in labels if l not in r.index or r.loc[l, 'train_donors_ge10'] <= 1]
        return dict(primary=multi, rare_few_donor=few, single_donor=single)
    if cond.get('target'):
        return dict(primary=[cond['target']])
    return dict(primary=list(labels))


rows, lab_rows, costs = [], [], []
for run_id in sys.argv[1:]:
    for rj in sorted((RUNS / run_id).glob('*.run.json')):
        meta = json.loads(rj.read_text())
        f = np.load(str(rj).replace('.run.json', '.npz'), allow_pickle=True)
        labels = list(f['labels']) if 'labels' in f else list(BROAD)
        present = set(labels[k] for k in f['train_present']) if 'train_present' in f else set(labels)
        y = f['truth']
        cond, test, seed = meta['condition'], meta['test'], meta['seed']
        sets = label_sets(cond, test, labels, present)
        for m in f['models']:
            lp = f[f'logp__{m}'].astype(np.float64)
            p = np.exp(lp)
            pred = lp.argmax(1)
            per = {}
            for k, l in enumerate(labels):
                pos = y == k
                a = auprc(pos, p[:, k])
                per[l] = a
                lab_rows.append(dict(run_id=run_id, condition=cond['name'], test_donor=test, seed=seed, model=m, label=l,
                                     n_pos=int(pos.sum()), evaluable=bool(pos.sum() >= 10), in_train=l in present,
                                     auprc=a, recall=float((pred[pos] == k).mean()) if pos.any() else np.nan,
                                     false_pos=int(((pred == k) & ~pos).sum()),
                                     to_other=float((pred[pos] != k).mean()) if pos.any() else np.nan))
            ev = [l for l in sets['primary'] if (y == labels.index(l)).sum() >= 10]
            seen = [k for k in range(len(labels)) if (y == k).any() and labels[k] in present]
            onehot = np.eye(len(labels))[y]
            row = dict(run_id=run_id, condition=cond['name'], test_donor=test, seed=seed, model=m,
                       primary_labels=','.join(ev), n_primary=len(ev),
                       primary_auprc=np.mean([per[l] for l in ev]) if ev else np.nan,
                       macro_f1=f1_score(y, pred, labels=seen, average='macro', zero_division=0),
                       brier=float(((p - onehot) ** 2).sum(1).mean()),
                       nll=float(-np.clip(lp[np.arange(len(y)), y], -50, 0).mean()),
                       n_cells=len(y))
            for key in ('rare_few_donor', 'single_donor'):
                if key in sets:
                    e2 = [l for l in sets[key] if (y == labels.index(l)).sum() >= 10]
                    row[f'{key}_labels'] = ','.join(e2)
                    row[f'{key}_auprc'] = np.mean([per[l] for l in e2 if np.isfinite(per[l])]) if e2 else np.nan
            if cond.get('target'):
                k = labels.index(cond['target'])
                pos = y == k
                row['target_recall'] = float((pred[pos] == k).mean()) if pos.any() else np.nan
                row['target_false_pos'] = int(((pred == k) & ~pos).sum())
            rows.append(row)
            costs.append(dict(condition=cond['name'], model=m, cpu_seconds=meta.get('model_cpu_seconds', {}).get(m, np.nan)))
df = pd.DataFrame(rows)
lab = pd.DataFrame(lab_rows)
write_tsv(df, REPORT / 'a2_per_run.tsv')
write_tsv(lab.groupby(['condition', 'model', 'label', 'test_donor']).agg(n_pos=('n_pos', 'first'), evaluable=('evaluable', 'first'),
          in_train=('in_train', 'first'), auprc=('auprc', 'mean'), recall=('recall', 'mean'), false_pos=('false_pos', 'mean'),
          to_other=('to_other', 'mean')).reset_index(), REPORT / 'a2_per_label_donor.tsv')
num = [c for c in df.columns if df[c].dtype != object and c not in ('seed',)]
pdn = df.groupby(['condition', 'model', 'test_donor'])[num].mean().reset_index()   # seeds averaged within donor
write_tsv(pdn, REPORT / 'a2_per_donor.tsv')
cost = pd.DataFrame(costs).groupby(['condition', 'model']).cpu_seconds.median().reset_index()

base = []
for (c, m), g in pdn.groupby(['condition', 'model']):
    ci = donor_paired_ci(g.primary_auprc.values)
    base.append(dict(condition=c, model=m, n_eval_donors=ci['n_donors'], primary_auprc=ci['mean'], ci_lo=ci['boot_lo'], ci_hi=ci['boot_hi'],
                     macro_f1=g.macro_f1.mean(), brier=g.brier.mean(), nll=g.nll.mean(),
                     **{k: g[k].mean() for k in ('rare_few_donor_auprc', 'single_donor_auprc', 'target_recall', 'target_false_pos') if k in g},
                     median_cpu_seconds_per_fit=cost[(cost.condition == c) & (cost.model == m)].cpu_seconds.median()))
base = pd.DataFrame(base)
write_tsv(base, REPORT / 'a2_model_table.tsv')
con = []
for c, g in pdn.groupby('condition'):
    pv = g.pivot(index='test_donor', columns='model', values='primary_auprc')
    pvf = g.pivot(index='test_donor', columns='model', values='macro_f1')
    for a, b, why in CONTRASTS:
        if a in pv and b in pv:
            for metric, P in (('primary_auprc', pv), ('macro_f1', pvf)):
                d = (P[a] - P[b])
                ci = donor_paired_ci(d.values)
                con.append(dict(condition=c, contrast=f'{a}-{b}', meaning=why, metric=metric, **ci,
                                per_donor=';'.join(f'{i}:{v:+.4f}' for i, v in d.items() if np.isfinite(v))))
write_tsv(pd.DataFrame(con), REPORT / 'a2_contrasts.tsv')
print(base.to_string())
print(pd.DataFrame(con)[lambda x: x.metric == 'primary_auprc'][['condition', 'contrast', 'n_donors', 'mean', 'boot_lo', 'boot_hi', 'n_pos', 'n_neg']].to_string())
