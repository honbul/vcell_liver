"""Summarise A1 N x D stress into reports/r9_rare_state/n_by_d_stress.tsv (+ effects table and figure 1).
Rotations are technical repeats inside a test donor: averaged within donor first; the donor is the independent unit.
Evaluable donor-label: >= 10 positive cells of the target in the test donor (pre-specified)."""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from r9common import *

run_id = sys.argv[1]
rd = RUNS / run_id
df = pd.concat([pd.read_csv(p, sep='\t') for p in sorted(rd.glob('*__HCC*.tsv'))], ignore_index=True)
df['evaluable'] = df.n_test_pos >= 10
metrics = ['auprc', 'recall', 'precision', 'false_pos', 'brier_target', 'macro_f1_all']
conf_cols = [c for c in df.columns if c.startswith('to_')]
per_donor = df.groupby(['target', 'head', 'N', 'D', 'test_donor', 'evaluable', 'n_test_pos'])[metrics + conf_cols].mean().reset_index()
per_donor.loc[per_donor.N == 0, 'design'] = 'full_target_pool'
per_donor['design'] = per_donor['design'].fillna('grid')
write_tsv(per_donor, REPORT / 'n_by_d_per_donor.tsv')

rows = []
for (tg, hd, N, D), g in per_donor[per_donor.evaluable].groupby(['target', 'head', 'N', 'D']):
    ci = donor_paired_ci(g.auprc.values)
    comp = df[(df.target == tg) & (df['head'] == hd) & (df.N == N) & (df.D == D)]
    rows.append(dict(target=tg, head=hd, N=N if N else 'all', D=D if D else 'all', label_condition='artificially_sparsified' if N else 'natural_target_pool',
                     n_eval_donors=ci['n_donors'], eval_donors=','.join(g.test_donor), donor_macro_auprc=ci['mean'],
                     auprc_ci_lo=ci['boot_lo'], auprc_ci_hi=ci['boot_hi'], recall=g.recall.mean(),
                     precision=g.precision.mean(), false_pos_per_donor=g.false_pos.mean(), brier=g.brier_target.mean(),
                     macro_f1_all=g.macro_f1_all.mean(),
                     **{f'rare_to_{c[3:]}': g[c].mean() for c in conf_cols if g[c].notna().any()},
                     rotations=int(comp.rotation.nunique()), train_target_cells=int(comp.n_train_target.median()),
                     example_donor_sets=';'.join(comp.donors.astype(str).unique()[:5])))
write_tsv(pd.DataFrame(rows), REPORT / 'n_by_d_stress.tsv')

# Pre-specified effects: D effect (D 1->4 at fixed N) vs N effect (N 32->128 at fixed D), paired by donor.
eff = []
pv = per_donor[per_donor.evaluable & (per_donor.N > 0)].pivot_table(index=['target', 'head', 'test_donor'], columns=['N', 'D'], values='auprc')
for (tg, hd), g in pv.groupby(level=[0, 1]):
    dD = np.nanmean(np.stack([g[(N, 4)] - g[(N, 1)] for N in (32, 64, 128)]), 0)
    dN = np.nanmean(np.stack([g[(128, D)] - g[(32, D)] for D in (1, 2, 4)]), 0)
    for name, v in [('D_effect_1to4_avgN', dD), ('N_effect_32to128_avgD', dN), ('D_minus_N_effect', dD - dN),
                    ('D_effect_at_N32', (g[(32, 4)] - g[(32, 1)]).values), ('D_effect_at_N128', (g[(128, 4)] - g[(128, 1)]).values),
                    ('N_effect_at_D1', (g[(128, 1)] - g[(32, 1)]).values), ('N_effect_at_D4', (g[(128, 4)] - g[(32, 4)]).values)]:
        ci = donor_paired_ci(v)
        eff.append(dict(target=tg, head=hd, contrast=name, **ci,
                        per_donor=';'.join(f'{d}:{x:.3f}' for d, x in zip(g.index.get_level_values(2), v))))
write_tsv(pd.DataFrame(eff), REPORT / 'n_by_d_effects.tsv')

fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
for i, hd in enumerate(sorted(per_donor['head'].unique())):
    for j, tg in enumerate(sorted(per_donor.target.unique())):
        ax = axes[i, j]
        sub = pd.DataFrame(rows)
        sub = sub[(sub.target == tg) & (sub['head'] == hd)]
        for N, col in zip([32, 64, 128], ['#9ecae1', '#4292c6', '#08519c']):
            s = sub[sub.N == N].sort_values('D')
            ax.errorbar(s.D.astype(int), s.donor_macro_auprc, yerr=[s.donor_macro_auprc - s.auprc_ci_lo, s.auprc_ci_hi - s.donor_macro_auprc],
                        marker='o', color=col, label=f'N={N}', capsize=3)
        full = sub[sub.N == 'all']
        if len(full):
            ax.axhline(full.donor_macro_auprc.iloc[0], ls='--', color='grey', label='all source target cells')
        ax.set_xticks([1, 2, 4])
        ax.set_title(f'{tg} | {hd}', fontsize=10)
        ax.set_xlabel('supporting source donors D')
        if j == 0:
            ax.set_ylabel('donor-macro AUPRC (new donor)')
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
fig.suptitle('Fig1. Same rare-cell count N, varying supporting donors D (artificially sparsified, GSE149614 LOPO; dev evidence)', fontsize=10)
fig.tight_layout()
fig.savefig(REPORT / 'figures/fig1_n_by_d.png', dpi=130)
print(pd.DataFrame(eff)[['target', 'head', 'contrast', 'n_donors', 'mean', 'boot_lo', 'boot_hi']].to_string())
