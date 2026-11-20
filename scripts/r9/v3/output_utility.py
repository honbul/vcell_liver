"""P0-R9 v3 step 1: re-aggregate STORED GSE151530 predictions into per (version, condition, seed, model, donor) output counts.

No training, no refit. Calibration (temperature + 2% source-OOF thresholds) is recomputed exactly as in
scripts/r9/v2/c1_a4_eval.py (functions copied verbatim) and checked against calibration_provenance.tsv.
AUPRC fields are recomputed with the v2 code only to verify they are unchanged.
Spec: configs/r9/output_utility_v3.json. Outputs: reports/r9_output_utility_v3/raw/*.tsv and check tables.
"""
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

from r9common import ROOT, RUNS, CACHE, REPORT, BROAD, PARENT, auprc, write_tsv, write_json, git_commit, Timer

TIMER = Timer()
SPEC = json.loads((ROOT / 'configs/r9/output_utility_v3.json').read_text())
V2SPEC = json.loads((ROOT / 'configs/r9/a4_external_norm_v2.json').read_text())
OUT = ROOT / 'reports/r9_output_utility_v3'
V2REP = REPORT / 'a4_external_norm_v2'
EPS = 0.02
MAPPED = ['B', 'Endothelial', 'Fibroblast', 'Myeloid', 'T/NK']
VERSIONS = ['original', 'replay', 'corrected']
COND = {c['name']: c for c in V2SPEC['conditions']}
assert [c for c in SPEC['conditions']] == list(COND)
assert SPEC['models'] == V2SPEC['models'] and SPEC['seeds'] == V2SPEC['seeds']


# ---- verbatim from scripts/r9/v2/c1_a4_eval.py ----
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
# ---- end verbatim ----


ext = ad.read_h5ad(CACHE / 'gse151530_hcc_counts.h5ad', backed='r').obs
mapped = ext.mapped.values
known = np.isin(mapped, MAPPED)
y = np.array([BROAD.index(m) if m in BROAD else -1 for m in mapped])
ypar = np.array([pnames.index(parent_of[c]) if c >= 0 else -1 for c in y])
patient = ext.patient.values
DONORS = sorted(pd.unique(patient))


def count_rows(version, cname, seed, m, lp_raw, T, tf, tp, cs):
    cond = COND[cname]
    lp = temp(lp_raw.astype(np.float64), T)
    p = np.exp(lp)
    pred, conf = p.argmax(1), p.max(1)
    fine = conf >= tf
    pp = p @ pmat
    par = ~fine & (pp.max(1) >= tp)
    absn = ~fine & ~par
    wrong_f = fine & known & (pred != y)
    wrong_p = par & known & (pp.argmax(1) != ypar)
    rows, cls = [], []
    for d in DONORS:
        dm = patient == d
        k = dm & known
        nk = int(k.sum())
        r = dict(version=version, condition=cname, seed=seed, model=m, ext_donor=d,
                 temperature=T, tau_fine=tf, tau_parent=tp, calibration_source=cs, n_known=nk,
                 n_fine=int((fine & k).sum()), n_parent=int((par & k).sum()), n_abstain=int((absn & k).sum()),
                 n_wrong_fine=int((wrong_f & dm).sum()), n_wrong_parent=int((wrong_p & dm).sum()))
        r['n_output'] = r['n_fine'] + r['n_parent']
        r['n_wrong_output'] = r['n_wrong_fine'] + r['n_wrong_parent']
        if nk > 0:
            r.update(fine_coverage=r['n_fine'] / nk, parent_fraction=r['n_parent'] / nk, abstain_fraction=r['n_abstain'] / nk,
                     output_coverage=r['n_output'] / nk)
        else:
            r.update(fine_coverage=np.nan, parent_fraction=np.nan, abstain_fraction=np.nan, output_coverage=np.nan)
        r['output_error'] = r['n_wrong_output'] / r['n_output'] if r['n_output'] > 0 else np.nan
        r['fine_error'] = r['n_wrong_fine'] / r['n_fine'] if r['n_fine'] > 0 else np.nan
        r['parent_error'] = r['n_wrong_parent'] / r['n_parent'] if r['n_parent'] > 0 else np.nan
        r['output_status'] = 'no_known_cells' if nk == 0 else ('no_output' if r['n_output'] == 0 else 'has_output')
        for t, lab in (('UNMAPPED_malignant', 'malignant'), ('UNMAPPED_unclassified', 'unclassified')):
            mm = dm & (mapped == t)
            r[f'n_{lab}'] = int(mm.sum())
            r[f'n_{lab}_fine'] = int((mm & fine).sum())
            r[f'n_{lab}_parent'] = int((mm & par).sum())
            r[f'n_{lab}_abstain'] = int((mm & absn).sum())
        # AUPRC recomputed with the v2 code (unchanged-check only)
        yy, pk = y[k], p[k]
        labs = [l for l in MAPPED if (yy == BROAD.index(l)).sum() >= 10]
        for l in MAPPED:
            pos = yy == BROAD.index(l)
            r[f'auprc_{l}'] = auprc(pos, pk[:, BROAD.index(l)]) if pos.sum() >= 10 else np.nan
        r['allmapped_auprc'] = np.nanmean([r[f'auprc_{l}'] for l in labs]) if labs else np.nan
        r['target_auprc'] = r[f'auprc_{cond["target"]}'] if cond.get('target') else np.nan
        r['primary'] = r['target_auprc'] if cond.get('target') else r['allmapped_auprc']
        rows.append(r)
        for c, l in enumerate(BROAD):
            if l not in MAPPED:
                continue
            kc = k & (y == c)
            cls.append(dict(version=version, condition=cname, seed=seed, model=m, ext_donor=d, true_class=l, n=int(kc.sum()),
                            n_fine=int((kc & fine).sum()), n_parent=int((kc & par).sum()), n_abstain=int((kc & absn).sum()),
                            n_wrong_fine=int((kc & wrong_f).sum()), n_wrong_parent=int((kc & wrong_p).sum())))
    return rows, cls


v1dir, v2dir, oofdir = RUNS / 'a4_external_v1', RUNS / V2SPEC['run_id'], RUNS / (V2SPEC['run_id'] + '_oof')
rows, cls, cal, inputs = [], [], [], []
for cname in SPEC['conditions']:
    for seed in SPEC['seeds']:
        tag = f'{cname}__s{seed}'
        f1 = np.load(v1dir / f'{tag}.npz', allow_pickle=True)
        f2 = np.load(v2dir / f'{tag}.npz', allow_pickle=True)
        assert (f1['ext_cell_ids'] == ext.index.values).all() and (f2['ext_cell_ids'] == ext.index.values).all()
        assert np.array_equal(f2['src_truth'], f1['src_truth'])
        m2 = json.loads((v2dir / f'{tag}.run.json').read_text())
        reuse = m2['path'] == 'compatible' and m2['task_T2_pass']
        fo = None if reuse else np.load(oofdir / f'{tag}.npz', allow_pickle=True)
        inputs.append(dict(condition=cname, seed=seed, v1_npz=str(v1dir / f'{tag}.npz'), v2_npz=str(v2dir / f'{tag}.npz'),
                           v2_calibration='v1_oof' if reuse else 'v2_oof', v2_oof_npz='' if reuse else str(oofdir / f'{tag}.npz')))
        for m in SPEC['models']:
            T1_, tf1, tp1 = calib(f1[f'oof_logp__{m}'], f1['src_truth'])
            if reuse:
                T2_, tf2, tp2, src2 = T1_, tf1, tp1, 'v1_oof'
            else:
                T2_, tf2, tp2 = calib(fo[f'oof_logp__{m}'], f2['src_truth'])
                src2 = 'v2_oof'
            cal.append(dict(condition=cname, seed=seed, model=m, v1_T=T1_, v1_tau_fine=tf1, v1_tau_parent=tp1,
                            v2_T=T2_, v2_tau_fine=tf2, v2_tau_parent=tp2, v2_calibration_source=src2))
            for ver, lp_raw, (T, tf, tp, cs) in (('original', f1[f'ext_logp__{m}'], (T1_, tf1, tp1, 'v1_oof')),
                                                 ('replay', f2[f'ext_logp_replay__{m}'], (T2_, tf2, tp2, src2)),
                                                 ('corrected', f2[f'ext_logp_corrected__{m}'], (T2_, tf2, tp2, src2))):
                r, c_ = count_rows(ver, cname, seed, m, lp_raw, T, tf, tp, cs)
                rows += r
                cls += c_
        print(f'[v3r9] {tag} done', flush=True)
        if os.environ.get('V3R9_PILOT') == '1':  # resource pilot: one task, nothing written to reports
            write_json(ROOT / 'assets/pilot_step1.json', dict(task=tag, rows=len(rows), **TIMER.finish()))
            sys.exit(0)

df = pd.DataFrame(rows)
C = pd.DataFrame(cls)
RAW = OUT / 'raw'
AUP = [f'auprc_{l}' for l in MAPPED] + ['allmapped_auprc', 'target_auprc', 'primary']
count_cols = [c for c in df.columns if c not in AUP]
# plan 5.5 file name holds the PRIMARY version (corrected); original/replay kept in a sibling file
write_tsv(df[df.version == 'corrected'][count_cols], OUT / 'output_counts_per_donor_seed.tsv')
write_tsv(df[df.version != 'corrected'][count_cols], OUT / 'output_counts_per_donor_seed_original_replay.tsv')
write_tsv(C.groupby(['version', 'condition', 'model', 'true_class'])[['n', 'n_fine', 'n_parent', 'n_abstain', 'n_wrong_fine', 'n_wrong_parent']]
          .sum().reset_index(), OUT / 'per_class_output_pooled.tsv')
write_tsv(pd.DataFrame(inputs), OUT / 'input_manifest.tsv')

# ---------------- checks ----------------
chk = []


def add(cid, desc, ok, detail):
    chk.append(dict(check=cid, description=desc, passed=bool(ok), detail=detail))


hk = df.n_known > 0
add('C1a', 'n_fine+n_parent+n_abstain == n_known (all rows)', (df.n_fine + df.n_parent + df.n_abstain == df.n_known).all(), f'rows={len(df)}')
fs = (df.fine_coverage + df.parent_fraction + df.abstain_fraction)[hk]
add('C1b', 'fine_coverage+parent_fraction+abstain_fraction == 1 on n_known>0 rows', (np.abs(fs - 1) <= 1e-12).all(),
    f'rows_with_known={int(hk.sum())}; max|sum-1|={float(np.abs(fs - 1).max()):.3g}')
add('C2a', 'n_output == n_fine+n_parent; n_wrong_output == n_wrong_fine+n_wrong_parent',
    ((df.n_output == df.n_fine + df.n_parent) & (df.n_wrong_output == df.n_wrong_fine + df.n_wrong_parent)).all(), '')
add('C2b', 'n_wrong_x <= n_x', ((df.n_wrong_fine <= df.n_fine) & (df.n_wrong_parent <= df.n_parent)).all(), '')
tot = df.groupby(['version', 'condition', 'seed', 'model'])[['n_known', 'n_malignant', 'n_unclassified']].sum()
add('C3a', 'per (version,condition,seed,model): sum n_known=27212, malignant=15723, unclassified=7088, total=50023',
    ((tot.n_known == 27212) & (tot.n_malignant == 15723) & (tot.n_unclassified == 7088)).all() and len(ext) == 50023,
    f'groups={len(tot)}; n_known range {tot.n_known.min()}-{tot.n_known.max()}')
per_ver = df.groupby('version').size().to_dict()
dups = int(df.duplicated(['version', 'condition', 'seed', 'model', 'ext_donor']).sum())
add('C3b', '2100 rows per version, 0 duplicate keys', all(v == 2100 for v in per_ver.values()) and len(per_ver) == 3 and dups == 0,
    f'{per_ver}; dup={dups}')
add('C4', 'NA iff zero denominator (output_error/fine_error/parent_error/coverages)',
    (df.output_error.isna() == (df.n_output == 0)).all() and (df.fine_error.isna() == (df.n_fine == 0)).all()
    and (df.parent_error.isna() == (df.n_parent == 0)).all() and (df.fine_coverage.isna() == (df.n_known == 0)).all()
    and (df.output_coverage.isna() == (df.n_known == 0)).all(), f'no_output rows={int((df.output_status == "no_output").sum())}; '
    f'no_known rows={int((df.output_status == "no_known_cells").sum())}')

# C5 / C6 vs stored v2 per-donor-seed tables
key = ['condition', 'seed', 'model', 'ext_donor']
aup_rows, frac_rows, bug = [], [], []
for ver in VERSIONS:
    st = pd.read_csv(V2REP / f'a4_per_donor_seed_{ver}.tsv', sep='\t')
    mg = df[df.version == ver].merge(st, on=key, suffixes=('', '_stored'), validate='one_to_one')
    assert len(mg) == 2100
    for c in AUP:
        a, b = mg[c].astype(float), mg[f'{c}_stored'].astype(float)
        both = a.notna() & b.notna()
        aup_rows.append(dict(version=ver, field=c, n_rows=len(mg), n_finite=int(both.sum()), na_pattern_equal=bool((a.isna() == b.isna()).all()),
                             max_abs_diff=float(np.abs(a - b)[both].max()) if both.any() else np.nan,
                             max_rel_diff=float((np.abs(a - b) / np.abs(b).clip(1e-12))[both].max()) if both.any() else np.nan))
    for new, old in (('fine_coverage', 'fine_coverage'), ('parent_fraction', 'parent_level_frac'), ('abstain_fraction', 'abstain_frac'),
                     ('fine_error', 'fine_error')):
        a, b = mg[new].astype(float), mg[f'{old}_stored' if old == new else old].astype(float)
        both = a.notna() & b.notna()
        frac_rows.append(dict(version=ver, new_field=new, stored_field=old, na_pattern_equal=bool((a.isna() == b.isna()).all()),
                              max_abs_diff=float(np.abs(a - b)[both].max()) if both.any() else np.nan))
    a, b = mg.output_error.astype(float), mg.output_error_stored.astype(float)
    ho = mg.n_output > 0
    frac_rows.append(dict(version=ver, new_field='output_error[has_output]', stored_field='output_error', na_pattern_equal=bool(b[ho].notna().all()),
                          max_abs_diff=float(np.abs(a - b)[ho].max())))
    no = mg[~ho]
    bug.append(dict(version=ver, table=f'reports/r9_rare_state/a4_external_norm_v2/a4_per_donor_seed_{ver}.tsv', n_rows=len(mg),
                    n_no_output_rows=len(no), n_no_output_rows_stored_as_0=int((no.output_error_stored == 0).sum()),
                    n_no_output_rows_stored_na=int(no.output_error_stored.isna().sum()),
                    n_no_fine_rows_fine_error_stored_na=int(mg[mg.n_fine == 0].fine_error_stored.isna().sum()), n_no_fine_rows=int((mg.n_fine == 0).sum())))
# v1 original table (a4_eval.py) as well
v1t = pd.read_csv(REPORT / 'a4_per_donor_seed.tsv', sep='\t')
mg = df[df.version == 'original'].merge(v1t, on=key, suffixes=('', '_v1'), validate='one_to_one')
no = mg[mg.n_output == 0]
bug.append(dict(version='original', table='reports/r9_rare_state/a4_per_donor_seed.tsv (v1)', n_rows=len(mg), n_no_output_rows=len(no),
                n_no_output_rows_stored_as_0=int((no.output_error_v1 == 0).sum()), n_no_output_rows_stored_na=int(no.output_error_v1.isna().sum()),
                n_no_fine_rows_fine_error_stored_na=int(mg[mg.n_fine == 0].fine_error_v1.isna().sum()), n_no_fine_rows=int((mg.n_fine == 0).sum())))
aup_rows.append(dict(version='original_vs_v1_table', field='primary', n_rows=len(mg), n_finite=int((mg.primary.notna() & mg.primary_v1.notna()).sum()),
                     na_pattern_equal=bool((mg.primary.isna() == mg.primary_v1.isna()).all()),
                     max_abs_diff=float(np.nanmax(np.abs(mg.primary - mg.primary_v1))), max_rel_diff=np.nan))
A = pd.DataFrame(aup_rows)
F = pd.DataFrame(frac_rows)
write_tsv(A, OUT / 'auprc_unchanged_check.tsv', float_format='%.3g')
write_tsv(F, OUT / 'stored_fraction_check.tsv', float_format='%.3g')
write_tsv(pd.DataFrame(bug), OUT / 'stored_output_error_no_output_rows.tsv')
add('C5a', 'recomputed AUPRC fields == stored v2 per-donor-seed tables (NA pattern equal, max|diff|<=1e-5)',
    A.na_pattern_equal.all() and (A.max_abs_diff.fillna(0) <= 1e-5).all(), f'max|diff|={A.max_abs_diff.max():.3g}')
# model-level primary vs stored model table
mt = pd.read_csv(V2REP / 'a4_model_table_all_versions.tsv', sep='\t')
pm = df.groupby(['version', 'condition', 'model', 'ext_donor']).primary.mean().groupby(['version', 'condition', 'model']).mean().reset_index()
pm = pm.merge(mt[['version', 'condition', 'model', 'primary']], on=['version', 'condition', 'model'], suffixes=('', '_stored'))
add('C5b', 'model-level donor-macro primary == a4_model_table_all_versions.tsv', len(pm) == 84 and (np.abs(pm.primary - pm.primary_stored) <= 1e-5).all(),
    f'rows={len(pm)}; max|diff|={float(np.abs(pm.primary - pm.primary_stored).max()):.3g}')
add('C6', 'fine/parent/abstain fractions & fine_error & has-output output_error == stored v2 tables',
    F.na_pattern_equal.all() and (F.max_abs_diff.fillna(0) <= 1e-5).all(), f'max|diff|={F.max_abs_diff.max():.3g}')
cp = pd.read_csv(V2REP / 'calibration_provenance.tsv', sep='\t')
cm = pd.DataFrame(cal).merge(cp, on=['condition', 'seed', 'model'], suffixes=('', '_stored'))
cc = ['v1_T', 'v1_tau_fine', 'v1_tau_parent', 'v2_T', 'v2_tau_fine', 'v2_tau_parent']
cdiff = max(float(np.abs(cm[c] - cm[f'{c}_stored']).max()) for c in cc)
add('C7', 'recomputed T / tau == calibration_provenance.tsv', len(cm) == 84 and cdiff <= 1e-5 and (cm.v2_calibration_source == cm.v2_calibration_source_stored).all(),
    f'rows={len(cm)}; max|diff|={cdiff:.3g}')
write_tsv(pd.DataFrame(cal), OUT / 'calibration_recomputed.tsv')
K = pd.DataFrame(chk)
write_tsv(K, OUT / 'checks_step1.tsv')
print(K.to_string())
write_json(OUT / 'step1_provenance.json', dict(commit=git_commit(), spec='configs/r9/output_utility_v3.json', rows=len(df), **TIMER.finish()))
if not K.passed.all():
    sys.exit('[v3r9] a check failed; see checks_step1.tsv')
