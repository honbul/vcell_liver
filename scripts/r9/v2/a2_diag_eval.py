"""R9-E1/E2 evaluation (configs/r9/prototype_diagnostics_v1.json). Reads assets/runs/prototype_diagnostics_v1[_e2*]
and the v1 allocation logs; writes reports/r9_rare_state/prototype_diagnostics_v1/:
  per_run.tsv (all arms x 90 keys), per_donor.tsv (seed mean), capacity_variance.tsv (E1 summaries + diagnostics),
  fallback_audit.tsv (v1 A01/A11 logs + v2 candidate logs), rejection_reasons.tsv, matched_k.tsv (E2 summaries),
  contrasts.tsv (primary + secondary, 95% / simultaneous / 90% CIs), decisions.tsv, replay_vs_v1.tsv, unit_counts.tsv
Usage: a2_diag_eval.py [extra run ids ...]
"""
import json
import sys

import numpy as np
import pandas as pd

from r9common import *

SPEC = json.loads((ROOT / 'configs/r9/prototype_diagnostics_v1.json').read_text())
SC = SPEC['source_contract']
OUT = REPORT / 'prototype_diagnostics_v1'
TH = SPEC['practical_threshold_auprc']
runs = [SPEC['run_id']] + sys.argv[1:]
rows = []
for r in runs:
    for f in sorted((RUNS / r).glob('*__rows.tsv')):
        d = pd.read_csv(f, sep='\t')
        d['run_id'] = r
        rows.append(d)
df = pd.concat(rows, ignore_index=True)
keys = [(c['name'], d, s) for c in SC['conditions'] for d in SC['test_donors'] for s in SC['seeds']]
have = set(map(tuple, df[df.run_id == SPEC['run_id']][['condition', 'test_donor', 'seed']].drop_duplicates().values))
missing = [k for k in keys if k not in have]
write_tsv(df, OUT / 'per_run.tsv')
num = [c for c in df.columns if df[c].dtype != object and c not in ('seed',)]
pdn = df.groupby(['condition', 'arm', 'test_donor'])[num].mean().reset_index()
write_tsv(pdn, OUT / 'per_donor.tsv')


def boot(d, n_boot=10000, seed=0):
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    rng = np.random.default_rng(seed)
    return d, (d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1) if len(d) > 1 else np.array([np.nan]))


def contrast(a, b, family_size):
    """Target-equally-weighted donor vector (targets evaluable for that donor) + per-target rows."""
    out = []
    pv = pdn.pivot_table(index=['test_donor', 'condition'], columns='arm', values='target_auprc')
    if a not in pv or b not in pv:
        return out
    diff = (pv[a] - pv[b]).dropna()
    for scope, dd in [('target_mean', diff.groupby(level='test_donor').mean())] + \
                     [(c, diff.xs(c, level='condition')) for c in sorted(diff.index.get_level_values('condition').unique())]:
        d, bs = boot(dd.values)
        lvl = 1 - 0.05 / family_size
        out.append(dict(contrast=f'{a}-{b}', scope=scope, n_donors=len(d), mean=float(d.mean()),
                        lo95=float(np.quantile(bs, 0.025)), hi95=float(np.quantile(bs, 0.975)),
                        lo_simult=float(np.quantile(bs, (1 - lvl) / 2)), hi_simult=float(np.quantile(bs, 1 - (1 - lvl) / 2)),
                        simult_level=lvl, lo90=float(np.quantile(bs, 0.05)), hi90=float(np.quantile(bs, 0.95)),
                        n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()),
                        per_donor=';'.join(f'{i}:{v:+.4f}' for i, v in dd.items())))
    return out


def category(r):
    if r['lo_simult'] > 0 or r['hi_simult'] < 0:
        return 'established_practical' if abs(r['mean']) >= TH else 'established_below_practical'
    if r['lo90'] >= -TH and r['hi90'] <= TH:
        return 'equivalent_within_margin'
    return 'not_established'


PRIMARY = {'E1': [('a', 'P3_pooledvar', 'P1_fixedvar'), ('b', 'P3_current', 'P3_fixedvar'), ('c', 'P3_fixedvar', 'P3_pooledvar')],
           'E2': [('d', 'supported_stop', 'A11_refit'), ('e', 'matched_kstar', 'A11_refit'), ('f', 'supported_stop', 'matched_kstar')]}
con = []
for fam, lst in PRIMARY.items():
    for lab, a, b in lst:
        for r in contrast(a, b, 3):
            con.append(dict(family=fam, id=lab, role='primary' if r['scope'] == 'target_mean' else 'primary_per_target', **r))
arms_new = ['P1_current', 'P3_current', 'P1_fixedvar', 'P3_fixedvar', 'P3_pooledvar', 'A11_refit', 'supported_stop', 'matched_kstar']
cond_arms = sorted(a for a in df.arm.unique() if a.startswith(('A11_refit_', 'supported_stop_', 'matched_kstar_')))
arms_new += cond_arms
for v in sorted({a.rsplit('_', 1)[1] for a in cond_arms}):
    for lab, a, b in PRIMARY['E2']:
        for r in contrast(f'{a}_{v}', f'{b}_{v}', 3):
            con.append(dict(family=f'E2_conditional_{v}', id=lab, role='conditional' if r['scope'] == 'target_mean' else 'conditional_per_target', **r))
sec = [('P3_fixedvar', 'P1_fixedvar'), ('P3_current', 'P1_current'), ('P1_current', 'P1_fixedvar'), ('matched_kstar', 'P3_current')]
sec += [(x, f'v1_{b}') for x in arms_new for b in ('LR_donor', 'MLP_donor', 'LR_class')]
sec += [('P1_current', 'v1_P1_donor'), ('P3_current', 'v1_A10'), ('A11_refit', 'v1_A11')]
for a, b in sec:
    for r in contrast(a, b, 1):
        con.append(dict(family='secondary', id='', role='secondary', **r))
C = pd.DataFrame(con)
C['category'] = [category(r) if r['role'] not in ('secondary',) else '' for _, r in C.iterrows()]
write_tsv(C, OUT / 'contrasts.tsv')

# decisions (literal application of the committed rules)
P = C[C.role == 'primary'].set_index('id')
dec = []
for i in 'abcdef':
    if i in P.index:
        r = P.loc[i]
        dec.append(dict(id=i, contrast=r.contrast, mean=r['mean'], simult_ci=f'[{r.lo_simult:+.4f}, {r.hi_simult:+.4f}]',
                        ci95=f'[{r.lo95:+.4f}, {r.hi95:+.4f}]', ci90=f'[{r.lo90:+.4f}, {r.hi90:+.4f}]', category=r.category))
cond_b = 'b' in P.index and P.loc['b'].hi_simult < 0 and P.loc['b']['mean'] <= -TH
cond_c = 'c' in P.index and P.loc['c'].hi_simult < 0 and P.loc['c']['mean'] <= -TH
dec.append(dict(id='E2_var_fixedvar_rerun', contrast='condition: (b) simult CI upper<0 and mean<=-0.01', category='MET -> run' if cond_b else 'not met -> not run'))
dec.append(dict(id='E2_var_pooledvar_rerun', contrast='condition: (c) simult CI upper<0 and mean<=-0.01', category='MET -> run' if cond_c else 'not met -> not run'))
write_tsv(pd.DataFrame(dec), OUT / 'decisions.tsv')

# E1 capacity/variance and E2 matched-K summaries (arm level, seed-mean donor then target-mean)
diag_cols = ['target_auprc', 'target_recall', 'target_false_pos', 'nll', 'n_comp', 'floor_frac', 'mean_abs_dlogvar', 'max_abs_dlogvar',
             'ess_min', 'ess_median', 'max_donor_share', 'train_ce_weighted', 'test_ce_classbal', 'ce_gap', 'target_n_comp',
             'target_ess_min', 'target_max_donor_share', 'target_floor_frac', 'target_mean_abs_dlogvar', 'cpu_seconds']
summ = []
for (c, a), g in pdn.groupby(['condition', 'arm']):
    summ.append(dict(condition=c, arm=a, n_donors_eval=int(g.target_auprc.notna().sum()),
                     **{k: float(g[k].mean()) for k in diag_cols if k in g}))
S = pd.DataFrame(summ)
E1 = ['P1_current', 'P3_current', 'P1_fixedvar', 'P3_fixedvar', 'P3_pooledvar', 'v1_P1_donor', 'v1_A10', 'v1_LR_class', 'v1_LR_donor', 'v1_MLP_donor']
write_tsv(S[S.arm.isin(E1)], OUT / 'capacity_variance.tsv')
write_tsv(S[~S.arm.isin(E1[:5])], OUT / 'matched_k.tsv')

# fallback audit: v1 logs (A01 == A11 allocation) and v2 logs
fa = []
for c in SC['conditions']:
    for d in SC['test_donors']:
        for s in SC['seeds']:
            tag = f'{c["name"]}__{d}__s{s}'
            meta = json.loads((RUNS / 'a2_main_v1' / f'{tag}.run.json').read_text())
            labels = meta['labels']
            log = json.loads((RUNS / 'a2_main_v1' / f'{tag}__alloclog_A01.json').read_text())
            fb = [r for r in log if r.get('fallback')]
            first = fb[0]['step'] if fb else None
            row = dict(condition=c['name'], test_donor=d, seed=s, source='v1_alloclog_A01', n_steps=len(log),
                       n_fallback=len(fb), first_fallback_step=first,
                       kstar=(len(labels) + first - 1) if first else len(labels) + len(log),
                       fallback_classes=','.join(labels[r['k']] for r in fb),
                       target_splits_supported=sum(1 for r in log if not r.get('fallback') and labels[r['k']] == c['target']),
                       target_splits_fallback=sum(1 for r in fb if labels[r['k']] == c['target']))
            p2 = RUNS / SPEC['run_id'] / f'{tag}.run.json'
            if p2.exists():
                info = json.loads(p2.read_text())['info']
                row.update(v2_kstar=info.get('kstar'), v2_first_fallback_step=info.get('first_fallback_step'),
                           v2_n_fallback=info.get('n_fallback_steps'), v2_target_kstar=info.get('target_kstar'),
                           v2_matches_v1=bool(info.get('first_fallback_step') == first and info.get('n_fallback_steps') == len(fb)))
            fa.append(row)
FA = pd.DataFrame(fa)
write_tsv(FA, OUT / 'fallback_audit.tsv')
rj = []
for f in sorted((RUNS / SPEC['run_id']).glob('*__candlog.json')):
    tag = f.name.replace('__candlog.json', '')
    cname, d, s = tag.split('__')
    L = json.loads(f.read_text())
    first = next((r['step'] for r in L['steps'] if r['fallback']), None)
    for r in L['candidates']:
        rj.append(dict(condition=cname, test_donor=d, seed=int(s[1:]), step=r['step'], at_first_fallback=r['step'] == first,
                       before_first_fallback=(first is None or r['step'] < first), k=r['k'], reason=r['rejection_reason']))
RJ = pd.DataFrame(rj)
if len(RJ):
    RJ['class_label'] = RJ.k.map(dict(enumerate(BROAD)))
    agg = RJ.groupby(['condition', 'at_first_fallback', 'class_label', 'reason']).size().rename('n').reset_index()
    write_tsv(agg, OUT / 'rejection_reasons.tsv')
rep = df[df.v1_ref.notna()] if 'v1_ref' in df else pd.DataFrame()
if len(rep):
    write_tsv(rep.groupby(['arm', 'v1_ref']).agg(n=('seed', 'size'), min_argmax_agreement=('v1_argmax_agreement', 'min'),
                                                 mean_argmax_agreement=('v1_argmax_agreement', 'mean'),
                                                 max_abs_dprob=('v1_max_abs_dprob', 'max')).reset_index(), OUT / 'replay_vs_v1.tsv')
write_tsv(pd.DataFrame([dict(expected_keys=len(keys), present_keys=len(have), missing=';'.join('__'.join(map(str, k)) for k in missing),
                             rows=len(df), arms=df.arm.nunique())]), OUT / 'unit_counts.tsv')
print(C[C.role == 'primary'][['family', 'id', 'contrast', 'n_donors', 'mean', 'lo_simult', 'hi_simult', 'lo95', 'hi95', 'category']].to_string())
print(pd.DataFrame(dec).to_string())
