"""Compile reports/r9_rare_state/run_ledger.tsv from every assets/runs/<run_id>/*.run.json (+ repr meta) and disk use."""
import json
import subprocess

import pandas as pd

from r9common import *

rows = []
for rj in sorted(RUNS.glob('*/*.run.json')):
    d = json.loads(rj.read_text())
    r = d.get('resources', {})
    cond = d.get('condition', {})
    rows.append(dict(run_id=rj.parent.name, part=rj.name.replace('.run.json', ''),
                     kind='smoke (not reported)' if rj.parent.name.startswith('smoke') else 'result',
                     condition=cond.get('name') if isinstance(cond, dict) else cond, test=d.get('test'),
                     seed=d.get('seed'), unknown=d.get('unknown'), method=d.get('method'),
                     repr=d.get('repr'), commit=d.get('commit'), host=r.get('host'), slurm_job=r.get('slurm_job'),
                     wall_seconds=r.get('wall_seconds'), cpu_seconds=r.get('cpu_seconds'), peak_rss_gib=r.get('peak_rss_gib'),
                     gpu=d.get('gpu'), peak_cuda_gib=d.get('peak_cuda_gib'),
                     data_version='GSE149614 processed sha256 b234f5ca; GSE151530 raw (a0_data_hashes.json)'))
for mj in sorted((CACHE / 'repr').glob('*__meta.json')):
    d = json.loads(mj.read_text())
    r = d['resources']
    rows.append(dict(run_id='repr_v1', part=d['fold'], kind='result', commit=d.get('commit'), host=r.get('host'),
                     slurm_job=r.get('slurm_job'), wall_seconds=r.get('wall_seconds'), cpu_seconds=r.get('cpu_seconds'),
                     peak_rss_gib=r.get('peak_rss_gib')))
df = pd.DataFrame(rows)
write_tsv(df, REPORT / 'run_ledger.tsv')
s = df.groupby(['run_id', 'kind']).agg(parts=('part', 'count'), wall_h=('wall_seconds', lambda x: x.sum() / 3600),
                                        cpu_h=('cpu_seconds', lambda x: x.sum() / 3600)).reset_index()
gpu = df[df.gpu.notna()] if 'gpu' in df else df.iloc[:0]
du = subprocess.check_output(['du', '-sh', str(ASSETS)], text=True).split()[0]
dus = subprocess.check_output(['du', '-sh'] + [str(p) for p in sorted(ASSETS.iterdir())], text=True)
write_tsv(s, REPORT / 'run_ledger_summary.tsv')
(REPORT / 'resource_summary.txt').write_text(
    s.to_string() + f"\n\nGPU parts: {len(gpu)}, GPU wall hours (fit + overhead): {gpu.wall_seconds.sum() / 3600:.2f}\n"
    f"Total CPU hours (process CPU time, all parts): {df.cpu_seconds.sum() / 3600:.1f}\n"
    f"Total task wall hours: {df.wall_seconds.sum() / 3600:.1f}\nassets disk: {du}\n{dus}")
print((REPORT / 'resource_summary.txt').read_text())
