"""Summarize smoke evidence of every registered F1 ablation option (run on G1 after the smoke job).

    .venvs/scgpt/bin/python src/feedback/f1_ablation_smoke_summary.py
Writes reports/feedback/F1/ablation/smoke_evidence.tsv and smoke_evidence.json.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'experiments/feedback/f1_ablation_runs.json'
OUT = ROOT / 'reports/feedback/F1/ablation'
PRIOR_INIT = {'M0': 'prior_assets/runs/C0/C0-Norman-v1-scgpt-M0-D0-s26-r0',
              'M2': 'prior_assets/runs/M/M-Norman-v1-scgpt-M2-D0-s26-r0',
              'M3': 'prior_assets/runs/M/M-Norman-v1-scgpt-M3-D0-s26-r0'}


def main():
    reg = json.loads(REGISTRY.read_text())
    rows, full = [], {}
    for spec in reg['runs']:
        run = ROOT / 'assets/runs/A-smoke' / spec['run_id']
        ev = json.loads((run / 'smoke_evidence.json').read_text())
        res = json.loads((run / 'result.json').read_text())
        init = json.loads((run / 'model_init.json').read_text())
        prior = json.loads((ROOT / PRIOR_INIT[spec['mode']] / 'model_init.json').read_text())
        shapes = sorted({f"{d[0]}x{d[1]}|{u[0]}x{u[1]}|s{scale:g}"
                         for d, u, scale in ev['adapter_shapes'].values()})
        sched = ev['schedule'][0]
        fb = ev['first_batch']
        rows.append(dict(
            run_id=spec['run_id'], experiment=spec['experiment'],
            n_trainable=ev['n_trainable'], prior_mode_n_trainable=prior['n_trainable'],
            model_init_identical_to_prior_mode=init == prior,
            adapter_mounts=len(ev['adapter_mounts']),
            ffn_mounts=sum('.linear' in m for m in ev['adapter_mounts']),
            adapter_shapes=';'.join(shapes), adapter_up_norm_min=ev['adapter_up_norm_min'],
            trainable_blocks=','.join(map(str, ev['trainable_blocks'])),
            frozen_checked=ev.get('frozen_backbone_tensors_checked'),
            frozen_changed=len(ev.get('frozen_backbone_tensors_changed', [])),
            lr_epoch1=ev['lr_per_epoch'][0], lr_epoch2=ev['lr_per_epoch'][1],
            updates_per_epoch=','.join(map(str, ev['updates_run'])),
            control_input=('pool_mean' if fb['control_equals_pool_mean'] and fb['control_rows_identical']
                           else 'paired_cell' if fb['control_equals_paired_cells'] else 'other'),
            sched_examples=sched['examples'], sched_updates=sched['updates'],
            sched_distinct=sched['distinct_cells'], sched_max_repeat=sched['max_cell_repeats'],
            sched_exposure=f"{sched['intervention_exposure_min']}-{sched['intervention_exposure_max']}",
            smoke_val_delta_rmse_epochs=','.join(f"{h['val_delta_rmse']:.4f}" for h in res['history']),
            replay_ok=res['status'] == 'smoke_only', gpu=res['gpu'],
            seconds=round(res['total_seconds_this_invocation'], 1)))
        full[spec['run_id']] = dict(evidence=ev, history=res['history'])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'smoke_evidence.tsv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    (OUT / 'smoke_evidence.json').write_text(json.dumps(full, indent=1) + '\n')
    for r in rows:
        print(json.dumps(r))


if __name__ == '__main__':
    main()
