"""Export compact, evidence-bound C0, M, D and F tables for the research report."""
import argparse
import csv
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from artifact_io import digest
from common_selection import PROTOCOL, immutable, read


def save_table(path, fields, rows):
    with TemporaryDirectory(prefix='table-',dir=path.parent) as tmp:
        staged = Path(tmp)/path.name
        with staged.open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        if path.exists():
            if digest(path) != digest(staged):
                raise FileExistsError(f'Existing research table differs: {path}')
        else:
            staged.replace(path)


def report(work, out):
    evidence = work/'common_evidence.json'
    verified = read(evidence)
    if verified['status'] != 'common_experiments_executed':
        raise ValueError('Common experimental evidence is incomplete')
    for name,sha in verified['files'].items():
        if digest(work/name) != sha:
            raise ValueError(f'Common report input changed: {name}')
    selection = read(work/'model_selection.json')
    plan = read(work/'confirmation_plan.json')
    frozen = read(work/'frozen_test.json')
    main = read(work/'confirmation_summary.json')
    dsha = verified['dataset_sha256']
    if any(x['dataset_sha256'] != dsha for x in (selection,plan,frozen,main)):
        raise ValueError('Common reports used different datasets')
    initial_split = read(PROTOCOL)['initial_split']
    main_rows = []
    effects_rows = []
    arm_summaries = [('',main,frozen)]
    for split in verified['reserve_splits']:
        path = work/f'reserve-s{split}'
        summary = read(path/'confirmation_summary.json')
        lock = read(path/'frozen_test.json')
        if summary['freeze_sha256'] != digest(path/'frozen_test.json'):
            raise ValueError('Reserve report does not match its freeze')
        if summary['primary_arm'] != main['primary_arm']:
            raise ValueError('Reserve report changed the frozen primary candidate')
        arm_summaries.append((split,summary,lock))
    for split,summary,lock in arm_summaries:
        for arm,rows in summary['arm_metrics'].items():
            if arm=='GEARS':
                continue
            training = [r for r in lock['rows'] if r['arm']==arm]
            rmse = [r['metrics']['delta_rmse'] for r in rows]
            selected = {r['job']['mode'] for r in training}
            sampling = {r['job']['data_mode'] for r in training}
            if len(selected)!=1 or len(sampling)!=1:
                raise ValueError('Mixed model or data setting within one arm')
            main_rows.append(dict(
                split_seed=split or initial_split,dataset_sha256=summary['dataset_sha256'],arm=arm,
                model_mode=selected.pop(),data_mode=sampling.pop(),n_learning_seeds=len(rows),
                n_test_interventions=len(summary['conditions']),
                test_delta_rmse_mean=mean(rmse),test_delta_rmse_min=min(rmse),
                test_delta_rmse_max=max(rmse),
                train_epoch_seconds_mean=mean(r['epoch_seconds'] for r in training),
                inference_seconds_mean=mean(r['inference_seconds'] for r in rows)))
        for name,effect in summary['effects'].items():
            lo,hi=effect['condition_bootstrap_95']
            effects_rows.append(dict(
                split_seed=split or initial_split, contrast=name, primary=name==summary['primary_arm'],
                mean_gain=effect['mean_gain'], seed_gains=','.join(map(str,effect['per_seed_gain'])),
                condition_ci95_low=lo,condition_ci95_high=hi))
    selected_runs = {r['job']['run_id'] for r in frozen['rows']}
    test_main_by_run = {}
    for r in frozen['rows']:
        source = next(t for t in main['arm_metrics'][r['arm']] if t['seed']==r['job']['seed'])
        test_main_by_run[r['job']['run_id']] = source['metrics']['delta_rmse']
    tunes = []
    for row in [*selection['rows'],*plan['data_screen_rows']]:
        job = row['job']
        tunes.append(dict(
            phase=job['phase'],run_id=job['run_id'],model=job['model'],mode=job['mode'],
            data_mode=job['data_mode'],learning_rate=job['lr'],n_trainable=row['trainable'],
            validation_delta_rmse=row['val_delta_rmse'],
            fixed_for_data_screen=job['run_id']==selection['fixed_model']['job']['run_id'],
            carried_to_frozen_test=job['run_id'] in selected_runs,
            test_delta_rmse_same_checkpoint=test_main_by_run.get(job['run_id']),
            gpu=row['gpu'],epoch_seconds=row['epoch_seconds'],
            result_sha256=row['result_sha256'],checkpoint_sha256=row['checkpoint_sha256']))
    baseline = main['c0_benchmark']
    baselines = []
    eligible = baseline['additive_subset_delta_rmse']
    common_names = {'scGPT':'D0M0','GEARS':'GEARS'}
    for name,m in baseline['complete_coverage'].items():
        baselines.append(dict(method=name,n_test_interventions=len(main['conditions']),
                              additive_subset_delta_rmse=eligible.get(common_names.get(name,name)),
                              additive_subset_n=baseline['additive_n'],**m))
    baselines.append(dict(method='additive',n_test_interventions=baseline['additive_n'],
                          additive_subset_delta_rmse=eligible.get('additive'),
                          additive_subset_n=baseline['additive_n'],
                          **(baseline['additive_metrics'] or {})))
    out.mkdir(parents=True,exist_ok=True)
    files = {}
    for file,rows,fields in (
            ('tuning_results.tsv',tunes,list(tunes[0])),
            ('common_benchmark.tsv',baselines,list(baselines[0])),
            ('factorial_arms.tsv',main_rows,list(main_rows[0])),
            ('factorial_effects.tsv',effects_rows,list(effects_rows[0]))):
        p = out/file
        save_table(p,fields,rows)
        files[file] = digest(p)
    immutable(out/'table_receipt.json',dict(evidence_sha256=digest(evidence),files=files))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report(a.workspace,a.output)


if __name__=='__main__':
    main()
