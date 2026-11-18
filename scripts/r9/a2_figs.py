"""prototype_support.tsv (allocation evidence) + figures 2 and 3 from A2 outputs.
Usage: a2_figs.py <a2_run_id>   (run after a2_eval.py)"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from r9common import *

run_id = sys.argv[1]
P = pd.concat([pd.read_csv(p, sep='\t') for p in sorted((RUNS / run_id).glob('*__prototypes.tsv')) if p.stat().st_size > 1],
              ignore_index=True)
P['single_donor_proto'] = P.support <= 1
P['dominated_proto'] = P.max_donor_share >= 0.8
per_fit = P.groupby(['condition', 'test_donor', 'seed', 'allocation', 'class_label']).agg(
    n_proto=('proto', 'count'), mean_support=('support', 'mean'), n_single_donor=('single_donor_proto', 'sum'),
    n_dominated=('dominated_proto', 'sum'), mean_max_share=('max_donor_share', 'mean')).reset_index()
summ = per_fit.groupby(['condition', 'allocation', 'class_label']).agg(
    mean_n_proto=('n_proto', 'mean'), min_n_proto=('n_proto', 'min'), max_n_proto=('n_proto', 'max'),
    mean_support=('mean_support', 'mean'), mean_single_donor_protos=('n_single_donor', 'mean'),
    mean_dominated_protos=('n_dominated', 'mean'), mean_max_donor_share=('mean_max_share', 'mean'),
    n_fits=('n_proto', 'size')).reset_index()
tot = P.groupby(['condition', 'allocation']).agg(frac_single_donor=('single_donor_proto', 'mean'),
                                                 frac_dominated=('dominated_proto', 'mean')).reset_index()
tot['class_label'] = 'ALL'
summ = pd.concat([summ, tot], ignore_index=True)
# allocation logs: how often the donor-supported rule hit the fixed fallback
fb = []
for lj in sorted((RUNS / run_id).glob('*__alloclog_A01.json')):
    L = json.loads(lj.read_text())
    cond = lj.name.split('__')[0]
    fb.append(dict(condition=cond, steps=len(L), fallback_steps=sum(bool(x.get('fallback')) for x in L)))
fb = pd.DataFrame(fb).groupby('condition').sum().reset_index()
fb['allocation'] = 'donor_supported'
fb['class_label'] = 'ALL(fallback steps / greedy steps)'
summ = pd.concat([summ, fb.rename(columns={'steps': 'greedy_steps'})], ignore_index=True)
write_tsv(summ, REPORT / 'prototype_support.tsv')

# Fig 2: same-total-prototype contributions (paired differences vs A10 per condition)
con = pd.read_csv(REPORT / 'a2_contrasts.tsv', sep='\t')
con = con[con.metric == 'primary_auprc']
keep = ['A11-A10', 'A11-A01', 'A10-A00', 'A01-A00', 'A11-TL_donor', 'A11-RA_donor', 'A11-GMM_K', 'A10-P1_donor', 'A11-LR_donor']
conds = [c for c in ['nat', 'sparse_B_N64_D4', 'sparse_Endothelial_N64_D4', 'sparse_Fibroblast_N64_D4', 'lin_Myeloid', 'lin_B'] if c in set(con.condition)]
fig, axes = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 4.8), sharey=True)
for ax, c in zip(np.atleast_1d(axes), conds):
    s = con[con.condition == c].set_index('contrast').reindex(keep)
    yv = np.arange(len(keep))
    ax.errorbar(s['mean'], yv, xerr=[s['mean'] - s.boot_lo, s.boot_hi - s['mean']], fmt='o', capsize=2,
                color=['#d62728' if k == 'A11-A10' else '#1f77b4' for k in keep][0])
    ax.scatter(s['mean'], yv, c=['#d62728' if k == 'A11-A10' else '#1f77b4' for k in keep], zorder=3)
    ax.axvline(0, color='grey', lw=0.8)
    ax.set_yticks(yv)
    ax.set_yticklabels(keep, fontsize=8)
    ax.set_title(c, fontsize=8)
    ax.set_xlabel('paired diff, primary AUPRC', fontsize=8)
fig.suptitle('Fig2. Same total prototypes: loss x allocation contributions (donor-paired mean, 95% donor bootstrap CI; GSE149614 dev)', fontsize=9)
fig.tight_layout()
fig.savefig(FIG / 'fig2_2x2_contrasts.png', dpi=130)

# Fig 3: natural-rare (tier 2) per-label per-donor AUPRC + donor support of donor-supported prototypes
lab = pd.read_csv(REPORT / 'a2_per_label_donor.tsv', sep='\t')
rare = pd.read_csv(REPORT / 'rare_definition_by_fold.tsv', sep='\t')
rare = rare[(rare.tier == 'res3_cluster') & rare.rare_1]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sub = lab[lab.condition.str.startswith('lin_') & lab.evaluable & lab.model.isin(['A10', 'A11', 'LR_donor', 'TL_donor'])]
sub = sub.merge(rare[['outer_test_donor', 'label', 'train_donors_ge10']], left_on=['test_donor', 'label'], right_on=['outer_test_donor', 'label'])
sub['bucket'] = np.where(sub.train_donors_ge10 >= 3, 'multi-donor rare', np.where(sub.train_donors_ge10 == 2, 'two-donor rare', 'single-donor'))
labs = sorted(sub.label.unique())
for j, m in enumerate(['LR_donor', 'A10', 'A11', 'TL_donor']):
    s = sub[sub.model == m]
    xs = np.array([labs.index(l) for l in s.label]) + (j - 1.5) * 0.18
    axes[0].scatter(xs, s.auprc, s=14, label=m)
axes[0].set_xticks(range(len(labs)))
axes[0].set_xticklabels([f'{l}\n({sub[sub.label == l].bucket.iloc[0]})' for l in labs], fontsize=7, rotation=45)
axes[0].set_ylabel('AUPRC in held-out donor (seed mean)')
axes[0].set_title('Tier-2 natural-rare clusters: each dot = one evaluable test donor', fontsize=9)
axes[0].legend(fontsize=7)
ds = P[(P.allocation == 'donor_supported') & P.condition.str.startswith('lin_')]
h = ds.groupby(['class_label']).agg(protos=('proto', 'count'), single=('single_donor_proto', 'mean'), support=('support', 'mean'))
fx = P[(P.allocation == 'fixed') & P.condition.str.startswith('lin_')].groupby('class_label').agg(protos=('proto', 'count'),
                                                                                              single=('single_donor_proto', 'mean'))
nfit = ds.groupby(['condition', 'test_donor', 'seed']).ngroups
idx = h.index.union(fx.index)
x = np.arange(len(idx))
axes[1].bar(x - 0.2, h.reindex(idx).protos / max(nfit, 1), 0.4, label='donor-supported: protos/fit')
axes[1].bar(x + 0.2, fx.reindex(idx).protos / max(nfit, 1), 0.4, label='fixed: protos/fit')
for i, l in enumerate(idx):
    v = h.reindex(idx).single.iloc[i]
    if np.isfinite(v):
        axes[1].text(i - 0.2, h.reindex(idx).protos.iloc[i] / max(nfit, 1) + 0.05, f'{v:.0%}', fontsize=6, ha='center', rotation=90)
axes[1].set_xticks(x)
axes[1].set_xticklabels(idx, rotation=90, fontsize=7)
axes[1].set_title('Prototypes per cluster (tier 2); text = share supported by <=1 donor (donor-supported)', fontsize=8)
axes[1].legend(fontsize=7)
fig.suptitle('Fig3. Natural-rare cluster errors and donor support of prototypes (cluster-ID reproduction; exploratory)', fontsize=9)
fig.tight_layout()
fig.savefig(FIG / 'fig3_natural_rare_support.png', dpi=130)
print(summ[summ.class_label.str.startswith('ALL')].to_string())
