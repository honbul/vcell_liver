"""R9-C2 (v2 correction): seed-0 matched contrasts between same-representation heads (A11, LR_donor, MLP_donor) and the
external full-pipeline models (scPoli, ProtoCloud) of A2.

v1 `a2_external_model_contrasts.tsv` subtracted the external seed-0 score from the 3-seed donor mean of our heads.
`r9tasks.select` uses the seed to choose sparse source donors and target cells, so the 3-seed mean and the external
seed-0 run were trained on different samples. This script pairs both sides on seed 0 (same condition, held-out donor,
source design and training cells), keeps the v1 numbers alongside, and verifies the pairing keys.

Inputs (read-only): reports/r9_rare_state/a2_per_run.tsv (v1 scored rows), assets/runs/{a2_main_v1,a2_ext_v1_*}
(symlinks to the v1 run directories). Outputs:
  reports/r9_rare_state/a2_external_model_contrasts_matched_seed0.tsv
  reports/r9_rare_state/a2_external_model_contrasts_matched_seed0_keys.tsv
"""
import hashlib
import json

import numpy as np
import pandas as pd

from r9common import *
from r9tasks import load_obs, select

OURS = ['A11', 'LR_donor', 'MLP_donor']
EXT = {'scpoli': 'a2_ext_v1_scpoli', 'protocloud': 'a2_ext_v1_protocloud'}
CONDS = {'nat': {'name': 'nat'},
         'sparse_B_N64_D4': {'name': 'sparse_B_N64_D4', 'target': 'B', 'N': 64, 'D': 4},
         'sparse_Endothelial_N64_D4': {'name': 'sparse_Endothelial_N64_D4', 'target': 'Endothelial', 'N': 64, 'D': 4},
         'sparse_Fibroblast_N64_D4': {'name': 'sparse_Fibroblast_N64_D4', 'target': 'Fibroblast', 'N': 64, 'D': 4}}
DONORS = [f'HCC{i:02d}' for i in range(1, 11)]


def h(arr):
    return hashlib.sha256('\n'.join(map(str, arr)).encode()).hexdigest()[:16]


pr = pd.read_csv(REPORT / 'a2_per_run.tsv', sep='\t')
rows = []
for cn in CONDS:
    for em, erun in EXT.items():
        ext = pr[(pr.run_id == erun) & (pr.condition == cn) & (pr.model == em) & (pr.seed == 0)].set_index('test_donor').primary_auprc
        if ext.empty:
            continue
        for m in OURS:
            sub = pr[(pr.run_id == 'a2_main_v1') & (pr.condition == cn) & (pr.model == m)]
            s0 = sub[sub.seed == 0].set_index('test_donor').primary_auprc
            m3 = sub.groupby('test_donor').primary_auprc.mean()
            assert len(s0) == 10 and len(ext) == 10 and sub.seed.nunique() == 3
            d_new = (s0 - ext).reindex(DONORS)
            d_old = (m3 - ext).reindex(DONORS)
            ci_new, ci_old = donor_paired_ci(d_new.values), donor_paired_ci(d_old.values)
            lo, hi = ci_new['boot_lo'], ci_new['boot_hi']
            if lo > 0:
                wording = f'{m} higher (CI excludes 0)'
            elif hi < 0:
                wording = f'{m} lower (CI excludes 0)'
            else:
                wording = 'difference not established (CI contains 0)'
            o_lo, o_hi = ci_old['boot_lo'], ci_old['boot_hi']
            old_dir = 'higher' if o_lo > 0 else ('lower' if o_hi < 0 else 'not established')
            new_dir = 'higher' if lo > 0 else ('lower' if hi < 0 else 'not established')
            rows.append(dict(condition=cn, contrast=f'{m}-{em}', ours=m, external=em, pairing='seed0_matched',
                             n_donors=ci_new['n_donors'], ours_s0=s0.mean(), ext_s0=ext.mean(), mean=ci_new['mean'],
                             boot_lo=lo, boot_hi=hi, t_lo=ci_new['t_lo'], t_hi=ci_new['t_hi'], n_pos=ci_new['n_pos'],
                             n_neg=ci_new['n_neg'], wording=wording,
                             v1_unmatched_mean=ci_old['mean'], v1_unmatched_boot_lo=o_lo, v1_unmatched_boot_hi=o_hi,
                             v1_direction=old_dir, matched_direction=new_dir, direction_changed=old_dir != new_dir,
                             per_donor=';'.join(f'{i}:{v:+.4f}' for i, v in d_new.items() if np.isfinite(v))))
out = pd.DataFrame(rows)
write_tsv(out, REPORT / 'a2_external_model_contrasts_matched_seed0.tsv')

# pairing keys: the seed-0 training/test sample of a2_main_v1 vs each external run
obs = load_obs()
keys = []
for cn, cond in CONDS.items():
    for test in DONORS:
        LABELS, yi, src, te, design = select(obs, cond, test, 0)
        exp_src, exp_te = h(obs.index.values[src]), h(obs.index.values[te])
        mm = json.loads((RUNS / 'a2_main_v1' / f'{cn}__{test}__s0.run.json').read_text())
        mf = np.load(RUNS / 'a2_main_v1' / f'{cn}__{test}__s0.npz', allow_pickle=True)
        for em, erun in EXT.items():
            p = RUNS / erun / f'{cn}__{test}__s0.run.json'
            if not p.exists():
                continue
            em_ = json.loads(p.read_text())
            ef = np.load(str(p).replace('.run.json', '.npz'), allow_pickle=True)
            keys.append(dict(condition=cn, test_donor=test, external=em, seed=0,
                             expected_train_ids_sha16=exp_src, expected_test_ids_sha16=exp_te,
                             main_n_train=mm['n_train'], ext_n_train=em_['n_train'], expected_n_train=len(src),
                             main_chosen=','.join(mm['design'].get('chosen', [])), ext_chosen=','.join(em_['design'].get('chosen', [])),
                             main_test_ids_sha16=h(mf['cell_ids']), ext_test_ids_sha16=h(ef['cell_ids']),
                             main_truth_equal_ext=bool(np.array_equal(mf['truth'], ef['truth'])),
                             main_seed=mm['seed'], ext_seed=em_['seed'], ext_gene_set=em_.get('gene_set', ''),
                             match=bool(mm['n_train'] == em_['n_train'] == len(src) and mm['design'] == em_['design']
                                        and h(mf['cell_ids']) == h(ef['cell_ids']) == exp_te
                                        and np.array_equal(mf['truth'], ef['truth']) and mm['seed'] == em_['seed'] == 0)))
kd = pd.DataFrame(keys)
write_tsv(kd, REPORT / 'a2_external_model_contrasts_matched_seed0_keys.tsv')
print(out[['condition', 'contrast', 'n_donors', 'mean', 'boot_lo', 'boot_hi', 'v1_unmatched_mean', 'v1_unmatched_boot_lo',
           'v1_unmatched_boot_hi', 'wording', 'direction_changed']].to_string())
print('key rows', len(kd), 'all match', bool(kd.match.all()))
