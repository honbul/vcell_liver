"""R9-C1 step 3: score Original / Replay / Corrected on GSE151530 with the v1 A4 scoring rules (copied verbatim from
a4_eval.py) and write the v2 tables. Calibration per the committed policy (configs/r9/a4_external_norm_v2.json).
Original = stored v1 predictions + v1 calibration (re-scored here; must reproduce reports/r9_rare_state/a4_per_donor_seed.tsv).
Outputs: reports/r9_rare_state/a4_external_norm_v2/*.tsv (never the v1 file names).
"""
import json

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from r9common import *

SPEC = json.loads((ROOT / 'configs/r9/a4_external_norm_v2.json').read_text())
REP = REPORT / 'a4_external_norm_v2'
EPS = 0.02
MAPPED = ['B', 'Endothelial', 'Fibroblast', 'Myeloid', 'T/NK']
VERSIONS = ['original', 'replay', 'corrected']
COMPARISONS = [('corrected', 'replay'), ('replay', 'original'), ('corrected', 'original')]
METRICS = ['primary', 'allmapped_auprc', 'target_auprc', 'macro_f1_mapped', 'pred_hepatocyte_frac_known', 'fine_coverage',
           'fine_error', 'parent_level_frac', 'abstain_frac', 'output_error', 'aurc'] + [f'auprc_{l}' for l in MAPPED]


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


def calib(oof_lp, src_truth):
    T = fit_T(oof_lp.astype(np.float64), src_truth)
    tf, tp = thresholds(temp(oof_lp.astype(np.float64), T), src_truth)
    return T, tf, tp


ext = ad.read_h5ad(CACHE / 'gse151530_hcc_counts.h5ad', backed='r').obs
mapped = ext.mapped.values
known = np.isin(mapped, MAPPED)
y = np.array([BROAD.index(m) if m in BROAD else -1 for m in mapped])


def score(version, cond, seed, m, lp_raw, T, tf, tp, calib_source):
    lp = temp(lp_raw.astype(np.float64), T)
    p = np.exp(lp)
    pred, conf = p.argmax(1), p.max(1)
    fine = conf >= tf
    pp = p @ pmat
    par = ~fine & (pp.max(1) >= tp)
    rows, unm = [], []
    for d in sorted(ext.patient.unique()):
        dm = (ext.patient.values == d)
        k = dm & known
        yy, pk = y[k], p[k]
        row = dict(version=version, condition=cond['name'], seed=seed, model=m, ext_donor=d, n_known=int(k.sum()),
                   n_unmapped=int((dm & ~known).sum()), temperature=T, tau_fine=tf, tau_parent=tp, calibration_source=calib_source)
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
        row['target_auprc'] = row[f'auprc_{cond["target"]}'] if cond.get('target') else np.nan
        row['primary'] = row['target_auprc'] if cond.get('target') else row['allmapped_auprc']
        row['primary_missing_reason'] = ('' if np.isfinite(row['primary']) else
                                         (f'<10 {cond["target"]} positives' if cond.get('target') else 'no class with >=10 positives'))
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
        dd = dict(version=version, condition=cond['name'], seed=seed, model=m, author_type=t, n=int(mm.sum()),
                  fine_frac=float(fine[mm].mean()), parent_frac=float(par[mm].mean()), abstain_frac=float((~fine[mm] & ~par[mm]).mean()))
        for c, l in enumerate(BROAD):
            dd[f'pred_{l}'] = float((pred[mm] == c).mean())
        unm.append(dd)
    level = np.where(fine, 0, np.where(par, 1, 2))
    return rows, unm, dict(pred=pred, conf=conf, level=level, maxp_raw=np.exp(lp_raw.astype(np.float64)).max(1))


v1dir = RUNS / 'a4_external_v1'
v2dir = RUNS / SPEC['run_id']
oofdir = RUNS / (SPEC['run_id'] + '_oof')
rows, unm, prob, calrows, missing = [], [], [], [], []
LEV = ['fine', 'parent', 'abstain']
for cond in SPEC['conditions']:
    for seed in SPEC['seeds']:
        tag = f'{cond["name"]}__s{seed}'
        f1 = np.load(v1dir / f'{tag}.npz', allow_pickle=True)
        assert (f1['ext_cell_ids'] == ext.index.values).all()
        p2 = v2dir / f'{tag}.npz'
        if not p2.exists():
            missing.append(dict(condition=cond['name'], seed=seed, reason='v2 run part missing'))
            continue
        f2 = np.load(p2, allow_pickle=True)
        m2 = json.loads((v2dir / f'{tag}.run.json').read_text())
        assert (f2['ext_cell_ids'] == ext.index.values).all() and np.array_equal(f2['src_truth'], f1['src_truth'])
        reuse = m2['path'] == 'compatible' and m2['task_T2_pass']
        if not reuse:
            po = oofdir / f'{tag}.npz'
            if not po.exists():
                missing.append(dict(condition=cond['name'], seed=seed, reason='calibration policy requires v2 OOF; missing'))
                continue
            fo = np.load(po, allow_pickle=True)
        for m in SPEC['models']:
            T1_, tf1, tp1 = calib(f1[f'oof_logp__{m}'], f1['src_truth'])
            if reuse:
                T2_, tf2, tp2, src2 = T1_, tf1, tp1, 'v1_oof'
            else:
                T2_, tf2, tp2 = calib(fo[f'oof_logp__{m}'], f2['src_truth'])
                src2 = 'v2_oof'
            calrows.append(dict(condition=cond['name'], seed=seed, model=m, v1_T=T1_, v1_tau_fine=tf1, v1_tau_parent=tp1,
                                v2_T=T2_, v2_tau_fine=tf2, v2_tau_parent=tp2, v2_calibration_source=src2,
                                identical=bool((T1_, tf1, tp1) == (T2_, tf2, tp2))))
            outs = {}
            for ver, lp_raw, (T, tf, tp, cs) in (('original', f1[f'ext_logp__{m}'], (T1_, tf1, tp1, 'v1_oof')),
                                                 ('replay', f2[f'ext_logp_replay__{m}'], (T2_, tf2, tp2, src2)),
                                                 ('corrected', f2[f'ext_logp_corrected__{m}'], (T2_, tf2, tp2, src2))):
                r, u, o = score(ver, cond, seed, m, lp_raw, T, tf, tp, cs)
                rows += r
                unm += u
                outs[ver] = o
            R_, Cc = outs['replay'], outs['corrected']
            acc_both = (R_['level'] == 0) & (Cc['level'] == 0)
            pr = dict(condition=cond['name'], seed=seed, model=m, calibration_source=src2,
                      mean_maxprob_raw_replay=float(R_['maxp_raw'].mean()), mean_maxprob_raw_corrected=float(Cc['maxp_raw'].mean()),
                      mean_maxprob_cal_replay=float(R_['conf'].mean()), mean_maxprob_cal_corrected=float(Cc['conf'].mean()),
                      label_change_rate_all=float((R_['pred'] != Cc['pred']).mean()),
                      label_change_rate_known=float((R_['pred'][known] != Cc['pred'][known]).mean()),
                      label_change_rate_common_fine_known=float((R_['pred'][known & acc_both] != Cc['pred'][known & acc_both]).mean()) if (known & acc_both).any() else np.nan,
                      replay_vs_original_label_change_known=float((R_['pred'][known] != outs['original']['pred'][known]).mean()))
            for a_, la in enumerate(LEV):
                for b_, lb in enumerate(LEV):
                    pr[f'known_{la}_to_{lb}'] = int(((R_['level'] == a_) & (Cc['level'] == b_) & known).sum())
            prob.append(pr)
df = pd.DataFrame(rows)
for ver in VERSIONS:
    write_tsv(df[df.version == ver].drop(columns='version'), REP / f'a4_per_donor_seed_{ver}.tsv')
num = [c for c in df.columns if df[c].dtype != object and c != 'seed']
pdn = df.groupby(['version', 'condition', 'model', 'ext_donor'])[num].mean().reset_index()
write_tsv(pdn, REP / 'a4_per_donor_all_versions.tsv')
U = pd.DataFrame(unm)
write_tsv(U, REP / 'a4_unmapped_cells_per_seed.tsv')
write_tsv(U.groupby(['version', 'condition', 'model', 'author_type']).mean(numeric_only=True).reset_index().drop(columns='seed'),
          REP / 'a4_unmapped_cells.tsv')
write_tsv(pd.DataFrame(prob), REP / 'probability_and_abstention_change.tsv')
write_tsv(pd.DataFrame(calrows), REP / 'calibration_provenance.tsv')
tab = []
for (v, c, m), g in pdn.groupby(['version', 'condition', 'model']):
    ci = donor_paired_ci(g.primary.values)
    tab.append(dict(version=v, condition=c, model=m, n_eval_donors=ci['n_donors'], primary=ci['mean'], ci_lo=ci['boot_lo'], ci_hi=ci['boot_hi'],
                    **{k: g[k].mean() for k in METRICS if k != 'primary'}))
write_tsv(pd.DataFrame(tab), REP / 'a4_model_table_all_versions.tsv')


def ci_level(d, level, n_boot=10000, seed=0):
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    if len(d) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


imp = []
for (c, m), g in pdn.groupby(['condition', 'model']):
    for metric in METRICS:
        pv = g.pivot(index='ext_donor', columns='version', values=metric)
        for a_, b_ in COMPARISONS:
            d = (pv[a_] - pv[b_])
            ci = donor_paired_ci(d.values)
            lo9875, hi9875 = ci_level(d.values, 1 - 0.05 / 4)
            imp.append(dict(condition=c, model=m, metric=metric, contrast=f'{a_}-{b_}',
                            primary_family=bool(m == SPEC['selected_version'] and metric == 'primary' and (a_, b_) == ('corrected', 'replay')),
                            **ci, simult_lo_98p75=lo9875, simult_hi_98p75=hi9875,
                            abs_mean_ge_practical=bool(abs(ci['mean']) >= SPEC['contrasts']['practical_threshold_auprc']) if np.isfinite(ci['mean']) else False,
                            per_donor=';'.join(f'{i.replace("GSE151530_", "")}:{v:+.4f}' for i, v in d.items() if np.isfinite(v))))
write_tsv(pd.DataFrame(imp), REP / 'paired_impact.tsv')
# unit counts
cnt = []
for ver in VERSIONS:
    s = df[df.version == ver]
    cnt.append(dict(version=ver, rows=len(s), dup_keys=int(s.duplicated(['condition', 'seed', 'model', 'ext_donor']).sum()),
                    valid_primary=int(np.isfinite(s.primary).sum()),
                    seed_mean_donor_rows=int((pdn.version == ver).sum()), model_rows=int(sum(1 for t in tab if t['version'] == ver)),
                    unmapped_seed_rows=int((U.version == ver).sum()), known_cells=int(known.sum()),
                    malignant=int((mapped == 'UNMAPPED_malignant').sum()), unclassified=int((mapped == 'UNMAPPED_unclassified').sum()),
                    **{f'valid_primary_{c}_per_head': int(np.isfinite(s[(s.condition == c)].primary).sum() / s[s.condition == c].model.nunique())
                       for c in sorted(s.condition.unique())}))
write_tsv(pd.DataFrame(cnt), REP / 'evaluation_unit_counts.tsv')
write_tsv(pd.DataFrame(missing, columns=['condition', 'seed', 'reason']), REP / 'missing_keys.tsv')
# Original re-score must reproduce v1
v1t = pd.read_csv(REPORT / 'a4_per_donor_seed.tsv', sep='\t')
o = df[df.version == 'original'].merge(v1t, on=['condition', 'seed', 'model', 'ext_donor'], suffixes=('', '_v1'))
chk = dict(n_merged=len(o), max_abs_primary_diff=float(np.nanmax(np.abs(o.primary - o.primary_v1))),
           nan_pattern_equal=bool((o.primary.isna() == o.primary_v1.isna()).all()),
           max_abs_coverage_diff=float(np.nanmax(np.abs(o.fine_coverage - o.fine_coverage_v1))))
write_json(REP / 'original_rescore_check.json', chk)
print(pd.DataFrame(cnt).to_string())
print(chk)
P = pd.DataFrame(imp)
print(P[(P.metric == 'primary')][['condition', 'model', 'contrast', 'mean', 'boot_lo', 'boot_hi', 'n_pos', 'n_neg']].to_string())
