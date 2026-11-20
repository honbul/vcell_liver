"""R9 v2 run ledger: one row per run part (resources from run JSONs) + prediction/asset sha256 index.
Outputs: reports/r9_normalization_prototype_v2/run_ledger_v2.tsv, resource_summary_v2.tsv,
manifests/r9_rare_state/prediction_index_a4_external_norm_v2.tsv, manifests/r9_rare_state/a4_external_norm_v2_manifest.json,
manifests/r9_rare_state/prototype_diagnostics_v1_index.tsv
"""
import json
import shutil

import numpy as np
import pandas as pd

from r9common import *

OUT = ROOT / 'reports/r9_normalization_prototype_v2'
RUN_IDS = ['a4_external_norm_v2', 'a4_external_norm_v2_oof', 'prototype_diagnostics_v1', 'prototype_diagnostics_v1_e2fixedvar',
           'prototype_diagnostics_v1_e2pooledvar']
rows, idx = [], {}
for r in RUN_IDS:
    d = RUNS / r
    if not d.exists():
        continue
    idx[r] = []
    for rj in sorted(d.glob('*.run.json')):
        m = json.loads(rj.read_text())
        res = m.get('resources', {})
        rows.append(dict(run_id=r, part=rj.name.replace('.run.json', ''), commit=m.get('commit'), host=res.get('host'),
                         slurm_job=res.get('slurm_job'), wall_seconds=res.get('wall_seconds'), cpu_seconds=res.get('cpu_seconds'),
                         peak_rss_gib=res.get('peak_rss_gib')))
    for f in sorted(d.glob('*.npz')):
        idx[r].append(dict(run_id=r, file=f.name, bytes=f.stat().st_size, sha256=sha(f)))
L = pd.DataFrame(rows)
write_tsv(L, OUT / 'run_ledger_v2.tsv')
summ = L.groupby('run_id').agg(parts=('part', 'size'), cpu_hours=('cpu_seconds', lambda x: x.sum() / 3600),
                               wall_hours_sum=('wall_seconds', lambda x: x.sum() / 3600), max_rss_gib=('peak_rss_gib', 'max')).reset_index()
tr = json.loads((CACHE / 'r9_repr_external_norm_v2' / 'bundle_meta.json').read_text())
summ = pd.concat([summ, pd.DataFrame([dict(run_id='c1_transform', parts=1, cpu_hours=tr['resources']['cpu_seconds'] / 3600,
                                           wall_hours_sum=tr['resources']['wall_seconds'] / 3600, max_rss_gib=tr['resources']['peak_rss_gib'])])])
summ['note'] = ''
write_tsv(summ, OUT / 'resource_summary_v2.tsv')
man = ROOT / 'manifests/r9_rare_state'
if 'a4_external_norm_v2' in idx:
    write_tsv(pd.DataFrame(idx['a4_external_norm_v2'] + idx.get('a4_external_norm_v2_oof', [])), man / 'prediction_index_a4_external_norm_v2.tsv')
    rep = pd.concat([pd.read_csv(f, sep='\t') for f in sorted((RUNS / 'a4_external_norm_v2').glob('*__replay_check.tsv'))])
    write_json(man / 'a4_external_norm_v2_manifest.json', dict(
        spec='configs/r9/a4_external_norm_v2.json', transform_bundle=tr, compat=json.loads((CACHE / 'r9_repr_external_norm_v2' / 'compat.json').read_text()),
        replay_T2_pass_rows=int(rep.T2_pass.sum()), replay_rows=len(rep),
        tasks_failing_T2=sorted(set(f'{a}__s{b}' for a, b in rep[~rep.T2_pass][['condition', 'seed']].values)),
        calibration_provenance='reports/r9_rare_state/a4_external_norm_v2/calibration_provenance.tsv'))
pi = [x for k in idx if k.startswith('prototype_diagnostics') for x in idx[k]]
if pi:
    write_tsv(pd.DataFrame(pi), man / 'prototype_diagnostics_v1_index.tsv')
print(summ.to_string())
