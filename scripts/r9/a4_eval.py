"""Score A4 (GSE151530 HCC, frozen version(s)). Known-label population = cells whose author Type maps to a GSE149614
broad label (T/NK, B, Myeloid, Endothelial, Fibroblast). Malignant/unclassified cells are not scored as known truth;
their prediction / abstention distribution is reported descriptively. Hepatocyte AUPRC is undefined (no positives).
Calibration (temperature, tau_fine, tau_parent) comes only from the GSE149614 source OOF predictions saved by A4.
Usage: a4_eval.py <a4_run_id> [<a4_ext_run_id> ...]
"""
import json
import sys

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from r9common import *

EPS = 0.02
MAPPED = ['B', 'Endothelial', 'Fibroblast', 'Myeloid', 'T/NK']
CONTRASTS = [('A11', 'A10'), ('A11', 'A01'), ('A10', 'A00'), ('A01', 'A00'), ('A11', 'LR_donor'), ('A10', 'LR_donor'),
             ('A11', 'GMM_K'), ('A10', 'P1_donor'), ('LR_donor', 'LR_class')]


def temp(lp, T):
    x = lp / T
    return x - np.logaddexp.reduce(x, axis=1, keepdims=True)


def fit_T(lp, y):
    grid = np.exp(np.linspace(np.log(0.25), np.log(8), 61))
    nll = [-temp(lp, T)[np.arange(len(y)), y].clip(-50).mean() for T in grid]
    return float(grid[int(np.argmin(nll))])


parent_of = [PARENT[l] for l in BROAD]
pnames = sorted(set(parent_of))
pmat = np.array([[parent_of[k] == p for p in pnames] for k in range(len(BROAD))], float)


def thresholds(lp, y):
    p = np.exp(lp)
    conf, pred = p.max(1), p.argmax(1)
    order = np.argsort(-conf)
    cum = np.cumsum((pred != y)[order]) / np.arange(1, len(y) + 1)
    ok = np.where(cum <= EPS)[0]
    tau_f = conf[order][ok.max()] if len(ok) else 1.01
    rest = conf < tau_f
    pp = p @ pmat
    yp = np.array([pnames.index(parent_of[k]) for k in y])
    o2 = np.argsort(-pp.max(1)[rest])
    perr = (pp.argmax(1)[rest] != yp[rest])[o2]
    cum2 = np.cumsum(perr) / np.arange(1, len(perr) + 1) if len(perr) else np.array([])
    ok2 = np.where(cum2 <= EPS)[0]
    tau_p = pp.max(1)[rest][o2][ok2.max()] if len(ok2) else 1.01
    return float(tau_f), float(tau_p)


ext = ad.read_h5ad(CACHE / 'gse151530_hcc_counts.h5ad', backed='r').obs
rows, unm, rcs = [], [], []
for run_id in sys.argv[1:]:
    for rj in sorted((RUNS / run_id).glob('*.run.json')):
        meta = json.loads(rj.read_text())
        f = np.load(str(rj).replace('.run.json', '.npz'), allow_pickle=True)
        assert (f['ext_cell_ids'] == ext.index.values).all()
        cond, seed = meta['condition'], meta['seed']
        mapped = ext.mapped.values
        known = np.isin(mapped, MAPPED)
        y = np.array([BROAD.index(m) if m in BROAD else -1 for m in mapped])
        for m in meta['models']:
            lp_raw = f[f'ext_logp__{m}'].astype(np.float64)
            if f'oof_logp__{m}' in f:
                T = fit_T(f[f'oof_logp__{m}'].astype(np.float64), f['src_truth'])
                tf, tp = thresholds(temp(f[f'oof_logp__{m}'].astype(np.float64), T), f['src_truth'])
            else:
                T, tf, tp = 1.0, np.nan, np.nan
            lp = temp(lp_raw, T)
            p = np.exp(lp)
            pred, conf = p.argmax(1), p.max(1)
            fine = conf >= tf
            pp = p @ pmat
            par = ~fine & (pp.max(1) >= tp)
            for d in sorted(ext.patient.unique()):
                dm = (ext.patient.values == d)
                k = dm & known
                yy, pk = y[k], p[k]
                row = dict(condition=cond['name'], seed=seed, model=m, ext_donor=d, n_known=int(k.sum()),
                           n_unmapped=int((dm & ~known).sum()), temperature=T, tau_fine=tf, tau_parent=tp)
                labs = [l for l in MAPPED if (yy == BROAD.index(l)).sum() >= 10]
                for l in MAPPED:
                    pos = yy == BROAD.index(l)
                    row[f'auprc_{l}'] = auprc(pos, pk[:, BROAD.index(l)]) if pos.sum() >= 10 else np.nan
                    row[f'npos_{l}'] = int(pos.sum())
                row['allmapped_auprc'] = np.nanmean([row[f'auprc_{l}'] for l in labs]) if labs else np.nan
                row['n_eval_labels'] = len(labs)
                present = [BROAD.index(l) for l in MAPPED if (yy == BROAD.index(l)).any()]
                row['macro_f1_mapped'] = f1_score(yy, pred[k], labels=present, average='macro', zero_division=0) if len(yy) else np.nan
                row['pred_hepatocyte_frac_known'] = float((pred[k] == BROAD.index('Hepatocyte')).mean()) if len(yy) else np.nan
                if cond.get('target'):
                    row['target_auprc'] = row[f'auprc_{cond["target"]}']
                row['primary'] = row['target_auprc'] if cond.get('target') else row['allmapped_auprc']
                ok = pred[k] == yy
                okp = np.array([pnames[j] for j in pp[k].argmax(1)]) == np.array([parent_of[c] for c in yy])
                fk, pk_ = fine[k], par[k]
                row.update(fine_coverage=fk.mean() if len(yy) else np.nan, fine_error=float((~ok[fk]).mean()) if fk.any() else np.nan,
                           parent_level_frac=pk_.mean() if len(yy) else np.nan, abstain_frac=float((~fk & ~pk_).mean()) if len(yy) else np.nan,
                           output_error=float(((fk & ~ok) | (pk_ & ~okp)).sum() / max((fk | pk_).sum(), 1)),
                           aurc=float((np.cumsum(~ok[np.argsort(-conf[k])]) / np.arange(1, len(yy) + 1)).mean()) if len(yy) else np.nan)
                rows.append(row)
            for t in ('UNMAPPED_malignant', 'UNMAPPED_unclassified'):
                mm = mapped == t
                d = dict(condition=cond['name'], seed=seed, model=m, author_type=t, n=int(mm.sum()),
                         fine_frac=float(fine[mm].mean()), parent_frac=float(par[mm].mean()), abstain_frac=float((~fine[mm] & ~par[mm]).mean()))
                for c, l in enumerate(BROAD):
                    d[f'pred_{l}'] = float((pred[mm] == c).mean())
                unm.append(d)
df = pd.DataFrame(rows)
write_tsv(df, REPORT / 'a4_per_donor_seed.tsv')
num = [c for c in df.columns if df[c].dtype != object and c != 'seed']
pdn = df.groupby(['condition', 'model', 'ext_donor'])[num].mean().reset_index()
write_tsv(pdn, REPORT / 'a4_per_donor.tsv')
write_tsv(pd.DataFrame(unm).groupby(['condition', 'model', 'author_type']).mean(numeric_only=True).reset_index().drop(columns='seed'),
          REPORT / 'a4_unmapped_cells.tsv')
tab = []
for (c, m), g in pdn.groupby(['condition', 'model']):
    ci = donor_paired_ci(g.primary.values)
    tab.append(dict(condition=c, model=m, n_eval_donors=ci['n_donors'], primary=ci['mean'], ci_lo=ci['boot_lo'], ci_hi=ci['boot_hi'],
                    macro_f1_mapped=g.macro_f1_mapped.mean(), pred_hepatocyte_frac_known=g.pred_hepatocyte_frac_known.mean(),
                    fine_coverage=g.fine_coverage.mean(), fine_error=g.fine_error.mean(), parent_level_frac=g.parent_level_frac.mean(),
                    abstain_frac=g.abstain_frac.mean(), output_error=g.output_error.mean(), aurc=g.aurc.mean(),
                    **{f'auprc_{l}': g[f'auprc_{l}'].mean() for l in MAPPED}))
tab = pd.DataFrame(tab)
write_tsv(tab, REPORT / 'a4_model_table.tsv')
con = []
for c, g in pdn.groupby('condition'):
    for metric in ('primary', 'macro_f1_mapped', 'aurc', 'output_error'):
        pv = g.pivot(index='ext_donor', columns='model', values=metric)
        for a, b in CONTRASTS:
            if a in pv and b in pv:
                d = pv[a] - pv[b]
                ci = donor_paired_ci(d.values)
                con.append(dict(condition=c, contrast=f'{a}-{b}', metric=metric, **ci,
                                per_donor=';'.join(f'{i.replace("GSE151530_", "")}:{v:+.4f}' for i, v in d.items() if np.isfinite(v))))
write_tsv(pd.DataFrame(con), REPORT / 'a4_contrasts.tsv')
# prior shift
dev = pd.read_csv(CACHE / 'gse149614_obs.tsv.gz', sep='\t').broad.value_counts(normalize=True)
extp = pd.Series(ext.mapped.values).value_counts(normalize=True)
write_tsv(pd.DataFrame(dict(label=BROAD + ['UNMAPPED_malignant', 'UNMAPPED_unclassified'],
                            dev_prevalence=[dev.get(l, 0) for l in BROAD] + [0, 0],
                            ext_prevalence=[extp.get(l, 0) for l in BROAD] + [extp.get('UNMAPPED_malignant', 0), extp.get('UNMAPPED_unclassified', 0)])),
          REPORT / 'a4_prior_shift.tsv')
# Fig 4: external per-donor primary metric by model, and three-level output composition
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
conds = sorted(tab.condition.unique())
ms = [m for m in ['LR_donor', 'P1_donor', 'GMM_K', 'A00', 'A10', 'A01', 'A11', 'scpoli', 'protocloud'] if m in set(tab.model)]
for i, c in enumerate(conds):
    sub = tab[tab.condition == c].set_index('model').reindex(ms)
    axes[0].errorbar(np.arange(len(ms)) + i * 0.12, sub.primary, yerr=[sub.primary - sub.ci_lo, sub.ci_hi - sub.primary], fmt='o', capsize=2, label=c)
axes[0].set_xticks(np.arange(len(ms)))
axes[0].set_xticklabels(ms, rotation=45)
axes[0].set_ylabel('external donor-macro AUPRC (primary)')
axes[0].legend(fontsize=7)
sub = tab[tab.condition == 'nat'].set_index('model').reindex(ms).dropna(subset=['fine_coverage'])
axes[1].bar(sub.index, sub.fine_coverage, label='fine label')
axes[1].bar(sub.index, sub.parent_level_frac, bottom=sub.fine_coverage, label='parent only')
axes[1].bar(sub.index, sub.abstain_frac, bottom=sub.fine_coverage + sub.parent_level_frac, label='abstain')
for j, (mm, r) in enumerate(sub.iterrows()):
    axes[1].text(j, 1.01, f'err {r.output_error:.3f}', ha='center', fontsize=7)
axes[1].set_ylim(0, 1.1)
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_title('nat: source-calibrated three-level output on GSE151530 known cells', fontsize=9)
axes[1].legend(fontsize=7)
fig.suptitle('Fig4. GSE151530 HCC (25 donors) one-time evaluation of frozen R9 versions', fontsize=10)
fig.tight_layout()
fig.savefig(FIG / 'fig4_external.png', dpi=130)
print(tab.to_string())
print(pd.DataFrame(con)[lambda x: x.metric == 'primary'][['condition', 'contrast', 'n_donors', 'mean', 'boot_lo', 'boot_hi', 'n_pos', 'n_neg']].to_string())
