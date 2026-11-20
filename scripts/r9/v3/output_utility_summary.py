"""P0-R9 v3 step 2: regenerate every summary number from the step-1 raw count tables only (plan 7.2 item 4).

Reads reports/r9_output_utility_v3/output_counts_per_donor_seed*.tsv (+ stored v1/v2 summary tables for the errata).
Writes coverage_per_donor.tsv, coverage_error_summary.tsv, no_output_summary.tsv, decision_mapping.tsv,
errata_output_error_old_vs_new.tsv, errata_paired_output_error.tsv, checks_step2.tsv.
"""
import json
import sys

import numpy as np
import pandas as pd

from r9common import ROOT, REPORT, donor_paired_ci, write_tsv, write_json, git_commit, Timer

TIMER = Timer()
SPEC = json.loads((ROOT / 'configs/r9/output_utility_v3.json').read_text())
OUT = ROOT / 'reports/r9_output_utility_v3'
V2REP = REPORT / 'a4_external_norm_v2'
df = pd.concat([pd.read_csv(OUT / 'output_counts_per_donor_seed.tsv', sep='\t'),
                pd.read_csv(OUT / 'output_counts_per_donor_seed_original_replay.tsv', sep='\t')], ignore_index=True)
G3 = ['version', 'condition', 'model']
COUNTS = ['n_known', 'n_fine', 'n_parent', 'n_abstain', 'n_output', 'n_wrong_fine', 'n_wrong_parent', 'n_wrong_output',
          'n_malignant', 'n_malignant_fine', 'n_malignant_parent', 'n_malignant_abstain',
          'n_unclassified', 'n_unclassified_fine', 'n_unclassified_parent', 'n_unclassified_abstain']
COV = ['fine_coverage', 'parent_fraction', 'abstain_fraction', 'output_coverage']
ERR = ['output_error', 'fine_error', 'parent_error']
chk = []

# recompute fractions from counts (do not trust the stored fraction columns)
re = df.copy()
nk = re.n_known.where(re.n_known > 0)
for c, num in (('fine_coverage', 'n_fine'), ('parent_fraction', 'n_parent'), ('abstain_fraction', 'n_abstain'), ('output_coverage', 'n_output')):
    re[c] = re[num] / nk
re['output_error'] = re.n_wrong_output / re.n_output.where(re.n_output > 0)
re['fine_error'] = re.n_wrong_fine / re.n_fine.where(re.n_fine > 0)
re['parent_error'] = re.n_wrong_parent / re.n_parent.where(re.n_parent > 0)
dmax = max(float(np.nanmax(np.abs(re[c] - df[c]))) for c in COV + ERR)
chk.append(dict(check='S1', description='fractions/errors recomputed from counts == step-1 columns (tsv %.6g)', passed=dmax <= 1e-5, detail=f'max|diff|={dmax:.3g}'))
nan_ok = all((re[c].isna() == df[c].isna()).all() for c in COV + ERR)
chk.append(dict(check='S2', description='NA pattern recomputed == step-1', passed=nan_ok, detail=''))

# ---- per donor (seed mean), all 25 donors always listed ----
rows = []
for (v, c, m, d), g in re.groupby(G3 + ['ext_donor']):
    ho = g[g.n_output > 0]
    r = dict(version=v, condition=c, model=m, ext_donor=d, n_known=int(g.n_known.iloc[0]), n_seeds=len(g),
             n_seeds_with_output=len(ho), n_seeds_no_output=int((g.output_status == 'no_output').sum()),
             **{k: g[k].mean() for k in COV},
             output_error=ho.output_error.mean() if len(ho) else np.nan,
             fine_error=g.fine_error.mean(), parent_error=g.parent_error.mean(),
             n_output_seed_mean=g.n_output.mean(), n_wrong_output_seed_mean=g.n_wrong_output.mean())
    for lab in ('malignant', 'unclassified'):
        n = g[f'n_{lab}']
        r[f'n_{lab}'] = int(n.iloc[0])
        for lev in ('fine', 'parent', 'abstain'):
            r[f'{lab}_{lev}_fraction'] = (g[f'n_{lab}_{lev}'] / n.where(n > 0)).mean()
    rows.append(r)
PD = pd.DataFrame(rows)
assert (PD.groupby(G3).size() == 25).all()
assert (re.groupby(G3 + ['ext_donor']).n_known.nunique() == 1).all()
write_tsv(PD, OUT / 'coverage_per_donor.tsv')

# ---- summary per version x condition x model ----
summ = []
for (v, c, m), g in PD.groupby(G3):
    r = dict(version=v, condition=c, model=m, n_donors_total=len(g), n_donors_with_known=int((g.n_known > 0).sum()))
    gk = g[g.n_known > 0]
    for k in COV:
        ci = donor_paired_ci(gk[k].values)
        r[k] = ci['mean']
        r[f'{k}_lo'] = ci['boot_lo']
        r[f'{k}_hi'] = ci['boot_hi']
    r['output_coverage_donor_min'] = gk.output_coverage.min()
    r['output_coverage_donor_median'] = gk.output_coverage.median()
    r['output_coverage_donor_max'] = gk.output_coverage.max()
    raw = re[(re.version == v) & (re.condition == c) & (re.model == m)]
    for k in ERR:
        ci = donor_paired_ci(g[k].values)
        r[k] = ci['mean']
        r[f'{k}_lo'] = ci['boot_lo']
        r[f'{k}_hi'] = ci['boot_hi']
        r[f'{k}_n_valid_donors'] = ci['n_donors']
    r['output_error_n_valid_donor_seed_rows'] = int((raw.n_output > 0).sum())
    r['n_donor_seed_rows'] = len(raw)
    r['n_no_output_rows'] = int((raw.output_status == 'no_output').sum())
    r['n_no_known_rows'] = int((raw.output_status == 'no_known_cells').sum())
    S = raw[COUNTS].sum()
    r.update({f'sum_{k}': int(S[k]) for k in COUNTS})
    r['pooled_output_error_aux'] = S.n_wrong_output / S.n_output if S.n_output > 0 else np.nan
    r['pooled_fine_error_aux'] = S.n_wrong_fine / S.n_fine if S.n_fine > 0 else np.nan
    r['pooled_output_coverage_aux'] = S.n_output / S.n_known if S.n_known > 0 else np.nan
    for lab in ('malignant', 'unclassified'):
        for lev in ('fine', 'parent', 'abstain'):
            r[f'pooled_{lab}_{lev}_fraction'] = S[f'n_{lab}_{lev}'] / S[f'n_{lab}']
    r['temperature_by_seed'] = ';'.join(f'{t:.4g}' for t in raw.groupby('seed').temperature.first())
    r['tau_fine_by_seed'] = ';'.join(f'{t:.4g}' for t in raw.groupby('seed').tau_fine.first())
    r['tau_parent_by_seed'] = ';'.join(f'{t:.4g}' for t in raw.groupby('seed').tau_parent.first())
    summ.append(r)
SM = pd.DataFrame(summ)
SM['_v'] = SM.version.map({'corrected': 0, 'replay': 1, 'original': 2})
SM['_m'] = SM.model.map({m: i for i, m in enumerate(SPEC['models'])})
SM['_c'] = SM.condition.map({c: i for i, c in enumerate(SPEC['conditions'])})
SM = SM.sort_values(['_v', '_c', '_m']).drop(columns=['_v', '_m', '_c'])
write_tsv(SM, OUT / 'coverage_error_summary.tsv')
fsum = (SM.fine_coverage + SM.parent_fraction + SM.abstain_fraction)
chk.append(dict(check='S3', description='donor-macro fine+parent+abstain == 1 per summary row', passed=bool((np.abs(fsum - 1) <= 1e-9).all()),
                detail=f'max|sum-1|={float(np.abs(fsum - 1).max()):.3g}'))
chk.append(dict(check='S4', description='sum_n_known == 3 seeds x 27212 per summary row', passed=bool((SM.sum_n_known == 3 * 27212).all()), detail=''))

# ---- no-output summary ----
no = []
for (v, c, m), g in re.groupby(G3):
    nod = g[g.output_status == 'no_output']
    per = nod.groupby('ext_donor').size()
    no.append(dict(version=v, condition=c, model=m, n_donor_seed_rows=len(g), n_has_output=int((g.output_status == 'has_output').sum()),
                   n_no_output=len(nod), n_no_known_cells=int((g.output_status == 'no_known_cells').sum()),
                   n_known_cells_in_no_output_rows=int(nod.n_known.sum()),
                   donors_no_output_any_seed=';'.join(f'{d.replace("GSE151530_", "")}(s{",".join(str(s) for s in sorted(nod[nod.ext_donor == d].seed))})' for d in per.index),
                   n_donors_no_output_all_seeds=int((per == g.seed.nunique()).sum()),
                   n_rows_fine_zero=int((g.n_fine == 0).sum()), n_rows_parent_zero=int((g.n_parent == 0).sum())))
NO = pd.DataFrame(no)
write_tsv(NO, OUT / 'no_output_summary.tsv')

# ---- errata: stored donor-macro output_error (zeros for no-output rows) vs NA-aware ----
old2 = pd.read_csv(V2REP / 'a4_model_table_all_versions.tsv', sep='\t')
old1 = pd.read_csv(REPORT / 'a4_model_table.tsv', sep='\t').assign(version='original')
er = SM[G3 + ['output_error', 'output_error_n_valid_donors', 'fine_error', 'n_no_output_rows']].merge(
    old2[G3 + ['output_error', 'fine_error']].rename(columns={'output_error': 'old_v2_output_error', 'fine_error': 'old_v2_fine_error'}), on=G3, how='left').merge(
    old1[G3 + ['output_error']].rename(columns={'output_error': 'old_v1_output_error'}), on=G3, how='left')
er['diff_new_minus_old_v2'] = er.output_error - er.old_v2_output_error
er['fine_error_diff_new_minus_old_v2'] = er.fine_error - er.old_v2_fine_error
write_tsv(er, OUT / 'errata_output_error_old_vs_new.tsv')
chk.append(dict(check='S5', description='errata rows matched to stored model tables (84 v2; 28 v1 original)',
                passed=bool(er.old_v2_output_error.notna().sum() == 84 and er.old_v1_output_error.notna().sum() == 28), detail=''))

# ---- errata: v2 paired impact output_error (Corrected-Replay etc.) with NA-aware donor values ----
pi = pd.read_csv(V2REP / 'paired_impact.tsv', sep='\t')
pi = pi[pi.metric == 'output_error']
pe = []
for (c, m), g in PD.groupby(['condition', 'model']):
    pv = g.pivot(index='ext_donor', columns='version', values='output_error')
    for a_, b_ in (('corrected', 'replay'), ('replay', 'original'), ('corrected', 'original')):
        d = (pv[a_] - pv[b_]).values
        ci = donor_paired_ci(d)
        o = pi[(pi.condition == c) & (pi.model == m) & (pi.contrast == f'{a_}-{b_}')].iloc[0]
        pe.append(dict(condition=c, model=m, contrast=f'{a_}-{b_}', new_n_donors=ci['n_donors'], new_mean=ci['mean'], new_lo=ci['boot_lo'],
                       new_hi=ci['boot_hi'], old_n_donors=int(o.n_donors), old_mean=o['mean'], old_lo=o.boot_lo, old_hi=o.boot_hi))
PE = pd.DataFrame(pe)
PE['changed'] = (PE.new_n_donors != PE.old_n_donors) | (np.abs(PE.new_mean - PE.old_mean) > 1e-6)
write_tsv(PE, OUT / 'errata_paired_output_error.tsv')

# ---- plan 5.4 decision mapping (frozen rules), LR_donor corrected ----
dm = []
for _, r in SM[(SM.version == 'corrected') & (SM.model == 'LR_donor')].iterrows():
    A = r.output_coverage < 0.50
    B = r.output_error_lo > 0.02
    Cc = (r.output_coverage >= 0.50) and (r.output_error_hi <= 0.02)
    rows_ = [x for x, f in (('row1_few_outputs', A), ('row2_high_conditional_error', B), ('row3_some_condition_ok', Cc)) if f]
    if not B and not Cc:
        rows_.append('error_vs_0.02_undetermined' if np.isfinite(r.output_error_lo) else 'error_undefined')
    dm.append(dict(condition=r.condition, model='LR_donor', version='corrected', output_coverage=r.output_coverage, output_coverage_lo=r.output_coverage_lo,
                   output_coverage_hi=r.output_coverage_hi, fine_coverage=r.fine_coverage, abstain_fraction=r.abstain_fraction,
                   output_error=r.output_error, output_error_lo=r.output_error_lo, output_error_hi=r.output_error_hi,
                   n_valid_donors=r.output_error_n_valid_donors, n_valid_donor_seed_rows=r.output_error_n_valid_donor_seed_rows,
                   n_no_output_rows=r.n_no_output_rows, pooled_output_error_aux=r.pooled_output_error_aux,
                   rule_A=bool(A), rule_B=bool(B), rule_C=bool(Cc), plan_5_4_rows='+'.join(rows_)))
DM = pd.DataFrame(dm)
write_tsv(DM, OUT / 'decision_mapping.tsv')
K = pd.DataFrame(chk)
write_tsv(K, OUT / 'checks_step2.tsv')
print(K.to_string())
pd.set_option('display.width', 250)
print(DM.to_string())
print(SM[SM.version == 'corrected'][G3 + ['fine_coverage', 'parent_fraction', 'abstain_fraction', 'output_coverage', 'output_error', 'output_error_lo',
                                          'output_error_hi', 'output_error_n_valid_donors', 'n_no_output_rows', 'pooled_output_error_aux']].to_string())
print(NO[NO.n_no_output > 0].to_string())
print(PE[PE.changed].to_string())
write_json(OUT / 'step2_provenance.json', dict(commit=git_commit(), spec='configs/r9/output_utility_v3.json', **TIMER.finish()))
if not K.passed.all():
    sys.exit('[v3r9] a step-2 check failed')
