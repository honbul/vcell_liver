"""Aggregate F1 ablation runs (A6-A9) into a results table and per-epoch curves.

Run on G1 from the worktree root (needs assets/ and prior_assets/; no torch import):
    .venvs/scgpt/bin/python src/feedback/f1_ablation_aggregate.py
Reads only result.json / history.json / model_init.json / settings.json and the Slurm
submission journal. Unfinished runs are listed with status 'running' or 'pending'.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'experiments/feedback/f1_ablation_runs.json'
JOURNAL = ROOT / 'logs/f1_ablation_submissions.tsv'
OUT = ROOT / 'reports/feedback/F1/ablation'

COLUMNS = ['run_id', 'experiment', 'change', 'mode', 'lr', 'adapt', 'data_mode', 'status',
           'trainable_params', 'best_epoch', 'epochs_run', 'val_delta_rmse',
           'reference_run', 'reference_val_delta_rmse', 'delta_vs_reference',
           'secondary_reference_run', 'secondary_reference_val_delta_rmse',
           'delta_vs_secondary_reference', 'val_delta_pearson', 'val_top20_recall',
           'val_direction_top20', 'updates', 'examples', 'gpu', 'train_val_minutes',
           'slurm_job_id']
CURVE_COLUMNS = ['run_id', 'kind', 'epoch', 'val_delta_rmse', 'train_loss', 'lr', 'updates',
                 'examples', 'distinct_examples', 'max_cell_repeats', 'intervention_min',
                 'intervention_max', 'val_delta_rmse_mean_control_input', 'seconds']


def read(path):
    return json.loads(Path(path).read_text())


def journal():
    rows = {}
    if JOURNAL.exists():
        for line in JOURNAL.read_text().splitlines():
            parts = line.split('\t')
            if len(parts) >= 2:
                rows[parts[0]] = parts[1]
    return rows


def prior_row(rid, ref):
    run = ROOT / ref['path']
    res = read(run / 'result.json')
    return res, dict(
        run_id=rid, experiment='prior', change=ref['label'], mode=res['mode'], lr=res['lr'],
        adapt='{}', data_mode=res['data_mode'], status='prior_reference',
        trainable_params=res['model_init']['n_trainable'], best_epoch=res['best_epoch'],
        epochs_run=len(res['history']), val_delta_rmse=res['validation']['delta_rmse'],
        val_delta_pearson=res['validation']['delta_pearson'],
        val_top20_recall=res['validation']['absolute_delta_top20_recall'],
        val_direction_top20=res['validation']['direction_top20'], updates=res['updates'],
        examples=res['examples'], gpu=res['gpu'],
        train_val_minutes=res['epoch_seconds'] / 60,
        slurm_job_id=read(run / 'settings.json').get('slurm_job_id'))


def aggregate(out):
    reg = read(REGISTRY)
    jobs = journal()
    refs, rows, curves = {}, [], []
    for rid, ref in reg['prior_references'].items():
        res, row = prior_row(rid, ref)
        refs[rid] = row['val_delta_rmse']
        rows.append(row)
        curves.extend(dict(run_id=rid, kind='prior', **{k: h.get(k) for k in CURVE_COLUMNS[2:]})
                      for h in res['history'])
    for spec in reg['runs']:
        rid = spec['run_id']
        run = ROOT / 'assets/runs/A' / rid
        row = dict(run_id=rid, experiment=spec['experiment'], change=spec['change'],
                   mode=spec['mode'], lr=spec['lr'], adapt=json.dumps(spec['adapt'], sort_keys=True),
                   data_mode=spec['data_mode'], reference_run=spec['reference'],
                   reference_val_delta_rmse=refs[spec['reference']],
                   secondary_reference_run=spec['secondary_reference'],
                   secondary_reference_val_delta_rmse=refs.get(spec['secondary_reference']),
                   slurm_job_id=jobs.get(rid))
        history = []
        if (run / 'result.json').exists():
            res = read(run / 'result.json')
            history = res['history']
            row.update(status=res['status'], trainable_params=res['model_init']['n_trainable'],
                       best_epoch=res['best_epoch'], val_delta_rmse=res['validation']['delta_rmse'],
                       val_delta_pearson=res['validation']['delta_pearson'],
                       val_top20_recall=res['validation']['absolute_delta_top20_recall'],
                       val_direction_top20=res['validation']['direction_top20'],
                       updates=res['updates'], examples=res['examples'], gpu=res['gpu'],
                       train_val_minutes=res['epoch_seconds'] / 60,
                       slurm_job_id=res.get('slurm_job_id') or row['slurm_job_id'])
        elif (run / 'history.json').exists():
            history = read(run / 'history.json')
            best = min(history, key=lambda h: h['val_delta_rmse'])
            row.update(status=f'running (epoch {len(history)})', best_epoch=best['epoch'],
                       val_delta_rmse=best['val_delta_rmse'],
                       updates=sum(h['updates'] for h in history),
                       examples=sum(h['examples'] for h in history),
                       train_val_minutes=sum(h['seconds'] for h in history) / 60)
        else:
            row.update(status='pending' if row['slurm_job_id'] else 'not_submitted')
        if (run / 'model_init.json').exists():
            row['trainable_params'] = read(run / 'model_init.json')['n_trainable']
        row['epochs_run'] = len(history) or None
        if row.get('val_delta_rmse') is not None:
            row['delta_vs_reference'] = row['val_delta_rmse'] - row['reference_val_delta_rmse']
            if row['secondary_reference_val_delta_rmse'] is not None:
                row['delta_vs_secondary_reference'] = (
                    row['val_delta_rmse'] - row['secondary_reference_val_delta_rmse'])
        rows.append(row)
        curves.extend(dict(run_id=rid, kind='ablation', **{k: h.get(k) for k in CURVE_COLUMNS[2:]})
                      for h in history)
    out.mkdir(parents=True, exist_ok=True)

    def fmt(v):
        if v is None:
            return ''
        if isinstance(v, float):
            return f'{v:.6g}'
        return str(v)

    for name, cols, data in (('ablation_results.tsv', COLUMNS, rows),
                             ('ablation_curves.tsv', CURVE_COLUMNS, curves)):
        with (out / name).open('w', newline='') as f:
            w = csv.writer(f, delimiter='\t', lineterminator='\n')
            w.writerow(cols)
            for r in data:
                w.writerow([fmt(r.get(c)) for c in cols])
    plot(rows, curves, reg, out)
    return rows


def plot(rows, curves, reg, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    groups = {}
    for spec in reg['runs']:
        groups.setdefault(spec['experiment'], set()).update(
            {spec['run_id'], spec['reference']} | ({spec['secondary_reference']} - {None}))
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), sharey=True)
    for ax, (exp, ids) in zip(axes, sorted(groups.items())):
        for rid in sorted(ids):
            pts = [(c['epoch'], c['val_delta_rmse']) for c in curves if c['run_id'] == rid]
            if not pts:
                continue
            prior = not rid.startswith('A-')
            label = rid.replace('-Norman-v1-scgpt', '').replace('-s26-r0', '')
            ax.plot(*zip(*pts), ls='--' if prior else '-', marker='' if prior else 'o', ms=3,
                    label=label + (' (prior)' if prior else ''))
        ax.set_title(exp)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('validation macro delta RMSE')
    fig.tight_layout()
    fig.savefig(out / 'ablation_val_curves.png', dpi=130)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    for r in aggregate(a.out):
        print(r['run_id'], r.get('status'), r.get('val_delta_rmse'), r.get('delta_vs_reference'))


if __name__ == '__main__':
    main()
