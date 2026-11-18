"""Score A3: calibration / three-level output / risk-coverage (calib run) and pseudo-unknown abstention (unknown run).
Usage: a3_eval.py <calib_run_id> <unknown_run_id> <a2_run_id>
Outputs: risk_coverage.tsv, a3_calibration_per_donor.tsv, a3_unknown_summary.tsv, a3_unknown_per_donor.tsv, a3_contrasts.tsv
"""
import json
import sys

import numpy as np
import pandas as pd

from r9common import *

calib_run, unk_run, a2_run = sys.argv[1:4]
COVS = [1.0, 0.95, 0.9, 0.8, 0.7]
rare_def = pd.read_csv(REPORT / 'rare_definition_by_fold.tsv', sep='\t')


def temp(lp, T):
    x = lp / T
    return x - np.logaddexp.reduce(x, axis=1, keepdims=True)


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def primary_labels(cond, test, labels):
    if cond.get('lineage'):
        r = rare_def[(rare_def.outer_test_donor == test) & (rare_def.tier == 'res3_cluster') & rare_def.label.isin(labels)].set_index('label')
        return [l for l in labels if l in r.index and r.loc[l, 'rare_1'] and r.loc[l, 'train_donors_ge10'] >= 3]
    if cond.get('target'):
        return [cond['target']]
    return []


rows, rc = [], []
for tsv in sorted((RUNS / calib_run).glob('*.tsv')):
    meta = json.loads(open(str(tsv).replace('.tsv', '.run.json')).read())
    cond, test = meta['condition'], meta['test']
    par = pd.read_csv(tsv, sep='\t')
    A2 = np.load(RUNS / a2_run / f'{cond["name"]}__{test}__s0.npz', allow_pickle=True)
    labels = list(A2['labels'])
    y = A2['truth']
    parent_of = [PARENT.get(l, cond.get('lineage', 'lineage')) for l in labels]
    prim = [labels.index(l) for l in primary_labels(cond, test, labels)]
    for _, r in par.iterrows():
        m = r.model
        lp_raw = A2[f'logp__{m}'].astype(np.float64)
        lp = temp(lp_raw, r.temperature)
        p = np.exp(lp)
        pr = np.exp(lp_raw)
        conf, pred = p.max(1), p.argmax(1)
        ok = pred == y
        fine = conf >= r.tau_fine
        pnames = sorted(set(parent_of))
        pp = np.stack([p[:, [k for k in range(len(labels)) if parent_of[k] == q]].sum(1) for q in pnames], 1)
        par_lvl = (~fine) & (pp.max(1) >= r.tau_parent)
        par_ok = np.array([pnames[j] for j in pp.argmax(1)]) == np.array([parent_of[k] for k in y])
        abst = ~fine & ~par_lvl
        onehot = np.eye(len(labels))[y]
        order = np.argsort(-conf)
        risk_curve = np.cumsum(~ok[order]) / np.arange(1, len(y) + 1)
        row = dict(condition=cond['name'], test_donor=test, model=m, n=len(y), temperature=r.temperature,
                   ece_raw=ece(pr, y), ece_cal=ece(p, y),
                   nll_raw=float(-np.clip(lp_raw[np.arange(len(y)), y], -50, 0).mean()),
                   nll_cal=float(-np.clip(lp[np.arange(len(y)), y], -50, 0).mean()),
                   brier_cal=float(((p - onehot) ** 2).sum(1).mean()),
                   aurc=float(risk_curve.mean()), tau_fine=r.tau_fine, tau_parent=r.tau_parent,
                   fine_coverage=fine.mean(), fine_error=float((~ok[fine]).mean()) if fine.any() else np.nan,
                   parent_level_frac=par_lvl.mean(), parent_error=float((~par_ok[par_lvl]).mean()) if par_lvl.any() else np.nan,
                   abstain_frac=abst.mean(),
                   output_error=float(((fine & ~ok) | (par_lvl & ~par_ok)).sum() / max((fine | par_lvl).sum(), 1)))
        if prim:
            pos = np.isin(y, prim)
            row['rare_fine_recall'] = float((fine & ok & pos).sum() / max(pos.sum(), 1))
            row['rare_false_pos_fine'] = int((fine & np.isin(pred, prim) & ~ok).sum())
        rows.append(row)
        for cov in COVS:
            n = max(1, int(round(cov * len(y))))
            acc = np.zeros(len(y), bool)
            acc[order[:n]] = True
            d = dict(condition=cond['name'], test_donor=test, model=m, coverage=cov,
                     selective_error=float((~ok[acc]).mean()))
            if prim:
                pos = np.isin(y, prim)
                d['rare_recall'] = float((acc & ok & pos).sum() / max(pos.sum(), 1))
                d['rare_false_pos'] = int((acc & np.isin(pred, prim) & ~ok).sum())
            rc.append(d)
cal = pd.DataFrame(rows)
write_tsv(cal, REPORT / 'a3_calibration_per_donor.tsv')
rcd = pd.DataFrame(rc)
out = []
for (c, m, cov), g in rcd.groupby(['condition', 'model', 'coverage']):
    ci = donor_paired_ci(g.selective_error.values)
    out.append(dict(condition=c, model=m, kind='fixed_coverage', coverage=cov, n_donors=ci['n_donors'],
                    selective_error=ci['mean'], ci_lo=ci['boot_lo'], ci_hi=ci['boot_hi'],
                    rare_recall=g.rare_recall.mean() if 'rare_recall' in g else np.nan,
                    rare_false_pos=g.rare_false_pos.mean() if 'rare_false_pos' in g else np.nan))
for (c, m), g in cal.groupby(['condition', 'model']):
    ci = donor_paired_ci(g.output_error.values)
    out.append(dict(condition=c, model=m, kind='source_calibrated_three_level', coverage=(g.fine_coverage + g.parent_level_frac).mean(),
                    n_donors=ci['n_donors'], selective_error=ci['mean'], ci_lo=ci['boot_lo'], ci_hi=ci['boot_hi'],
                    fine_coverage=g.fine_coverage.mean(), fine_error=g.fine_error.mean(), parent_level_frac=g.parent_level_frac.mean(),
                    parent_error=g.parent_error.mean(), abstain_frac=g.abstain_frac.mean(), aurc=g.aurc.mean(),
                    ece_raw=g.ece_raw.mean(), ece_cal=g.ece_cal.mean(), nll_raw=g.nll_raw.mean(), nll_cal=g.nll_cal.mean(),
                    rare_recall=g.rare_fine_recall.mean() if 'rare_fine_recall' in g else np.nan,
                    rare_false_pos=g.rare_false_pos_fine.mean() if 'rare_false_pos_fine' in g else np.nan))
write_tsv(pd.DataFrame(out), REPORT / 'risk_coverage.tsv')

con = []
for c, g in cal.groupby('condition'):
    for metric in ('aurc', 'ece_cal', 'output_error', 'abstain_frac'):
        pv = g.pivot(index='test_donor', columns='model', values=metric)
        for a, b in [('A11', 'A10'), ('A10', 'A00'), ('A11', 'LR_donor'), ('A10', 'LR_donor')]:
            if a in pv and b in pv:
                ci = donor_paired_ci((pv[a] - pv[b]).values)
                con.append(dict(part='calibration', condition=c, contrast=f'{a}-{b}', metric=metric, **ci))

if unk_run != '-':
    uk = pd.concat([pd.read_csv(p, sep='\t') for p in sorted((RUNS / unk_run).glob('*.tsv'))], ignore_index=True)
    write_tsv(uk, REPORT / 'a3_unknown_per_donor.tsv')
    s = []
    for (u, m, sc), g in uk.groupby(['unknown_label', 'model', 'score']):
        ci = donor_paired_ci(g.test_auroc.values)
        s.append(dict(unknown_label=u, model=m, score=sc, n_donors=ci['n_donors'], test_auroc=ci['mean'], ci_lo=ci['boot_lo'],
                      ci_hi=ci['boot_hi'], src_auroc=g.src_auroc.mean(), unknown_rejected=g.unknown_rejected.mean(),
                      known_rejected=g.known_rejected.mean(), known_acc_accepted=g.known_acc_accepted.mean()))
    for (m, sc), g in uk.groupby(['model', 'score']):
        pdn = g.groupby('test_donor')[['test_auroc', 'unknown_rejected', 'known_rejected']].mean()
        ci = donor_paired_ci(pdn.test_auroc.values)
        s.append(dict(unknown_label='ALL(mean over labels)', model=m, score=sc, n_donors=ci['n_donors'], test_auroc=ci['mean'],
                      ci_lo=ci['boot_lo'], ci_hi=ci['boot_hi'], unknown_rejected=pdn.unknown_rejected.mean(),
                      known_rejected=pdn.known_rejected.mean()))
    write_tsv(pd.DataFrame(s), REPORT / 'a3_unknown_summary.tsv')
    for sc in ('maxprob', 'logdens'):
        g = uk[uk.score == sc].groupby(['test_donor', 'model'])[['test_auroc', 'unknown_rejected', 'known_rejected']].mean().reset_index()
        for metric in ('test_auroc', 'unknown_rejected', 'known_rejected'):
            pv = g.pivot(index='test_donor', columns='model', values=metric)
            for a, b in [('A11', 'A10'), ('A10', 'A00'), ('A11', 'LR_donor'), ('A10', 'LR_donor'), ('A10', 'P1_donor'), ('GMM_K', 'A10')]:
                if a in pv and b in pv:
                    ci = donor_paired_ci((pv[a] - pv[b]).values)
                    con.append(dict(part=f'unknown_{sc}', condition='nat', contrast=f'{a}-{b}', metric=metric, **ci))
write_tsv(pd.DataFrame(con), REPORT / 'a3_contrasts.tsv')
print(pd.DataFrame(out)[lambda x: x.kind != 'fixed_coverage'].to_string())
print(pd.DataFrame(con).to_string())
