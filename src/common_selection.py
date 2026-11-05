"""Select from prespecified validation runs and freeze confirmation without test RNA."""
import argparse
import json
import math
from pathlib import Path

from artifact_io import digest, write

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'protocols/common/selection-v1.json'
NEURAL = ROOT/'protocols/common/neural-v1.json'


def read(path):
    return json.loads(path.read_text())


def immutable(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + '.lock')
    lock.mkdir()
    try:
        if path.exists():
            if read(path) == item:
                return
            raise FileExistsError(f'Frozen output differs: {path}')
        write(path, item)
    finally:
        lock.rmdir()


def spec(phase, model, mode, data_mode, split, seed, lr):
    return dict(phase=phase, run_id=f'{phase}-Norman-v1-{model}-{mode}-{data_mode}-s{split}-r{seed}',
                model=model, mode=mode, data_mode=data_mode, seed=seed, lr=lr)


def declared_screen(contract):
    split, seed = contract['initial_split'], contract['initial_seed']
    m = contract['model_screen']
    jobs = [spec('C0','scgpt','M0','D0',split,seed,m['lr']),
            spec('C0','gears','M0','D0',split,seed,0.001)]
    if [j['run_id'] for j in jobs] != [m['scgpt_baseline'],m['gears_baseline']]:
        raise ValueError('Inconsistent baseline run IDs')
    return jobs + [spec('M','scgpt',mode,'D0',split,seed,m['lr']) for mode in m['modes']]


def completed(runs, job, dataset_sha):
    path = runs/job['phase']/job['run_id']
    result_file = path/'result.json'
    r = read(result_file)
    settings = read(path/'settings.json')['contract']
    if r['status'] != 'validation_complete' or r['dataset_sha256'] != dataset_sha:
        raise ValueError(f'Unfinished or wrong data: {path}')
    if any(r[k] != v for k, v in job.items()):
        raise ValueError(f'Run does not match registered trial: {path}')
    if settings['protocol_sha256'] != digest(NEURAL) or settings['dataset_sha256'] != dataset_sha:
        raise ValueError(f'Unexpected training protocol: {path}')
    for field in ('model','mode','data_mode','seed','lr'):
        if settings[field] != job[field]:
            raise ValueError(f'Training settings mismatch: {path}')
    if any(digest(ROOT/'src'/n) != sha for n,sha in settings['adapter_sha256'].items()):
        raise ValueError(f'Active code differs from registered run code: {path}')
    validation = read(path/'best_validation.json')
    val = r['validation']['delta_rmse']
    if validation['role'] != 'val' or validation['n_available'] != validation['n_scored']:
        raise ValueError(f'Validation coverage mismatch: {path}')
    if not math.isfinite(val) or validation['metrics'] != r['validation']:
        raise ValueError(f'Validation report mismatch: {path}')
    if digest(path/'best_model.pt') != r['best_model_sha256']:
        raise ValueError(f'Checkpoint hash mismatch: {path}')
    return dict(job=job, val_delta_rmse=val, result_sha256=digest(result_file),
                checkpoint_sha256=r['best_model_sha256'],
                trainable=r['model_init']['n_trainable'], gpu=r['gpu'],
                epoch_seconds=r['epoch_seconds'])


def rank(row):
    return (row['val_delta_rmse'], row['job']['mode'], row['job']['data_mode'], row['job']['lr'])


def select_model(contract, runs, dataset_sha):
    jobs = declared_screen(contract)
    rows = [completed(runs, j, dataset_sha) for j in jobs]
    eligible = [r for r in rows if r['job']['model'] == 'scgpt' and r['job']['mode'] != 'M-random']
    fixed = min(eligible, key=rank)
    alternative = min((r for r in eligible if r['job']['mode'] != 'M0'), key=rank)
    d_jobs = [spec('D','scgpt',fixed['job']['mode'],mode,contract['initial_split'],
                   contract['initial_seed'],fixed['job']['lr'])
              for mode in contract['data_screen']['modes']]
    return dict(rows=rows, fixed_model=fixed, model_alternative=alternative, jobs=d_jobs,
                protocol_sha256=digest(PROTOCOL), dataset_sha256=dataset_sha)


def choose_confirmation(contract, model_pick, runs, dataset_sha):
    if model_pick != select_model(contract, runs, dataset_sha):
        raise ValueError('Model selection is no longer reproducible from the declared validation trials')
    rows = [completed(runs, j, dataset_sha) for j in model_pick['jobs']]
    base, best_m, fixed = model_pick['rows'][0], model_pick['model_alternative'], model_pick['fixed_model']
    best_d = min(rows,key=rank)
    m_gain = base['val_delta_rmse'] - best_m['val_delta_rmse']
    d_gain = fixed['val_delta_rmse'] - best_d['val_delta_rmse']
    arms = {'D0M0': ('D0', 'M0', base['job']['lr'])}
    model_only = ('D0', best_m['job']['mode'], best_m['job']['lr'])
    data_only = (best_d['job']['data_mode'], 'M0', base['job']['lr'])
    if m_gain > 0 and d_gain > 0:
        arms.update({'D0M*': model_only, 'D*M0': data_only,
                     'D*M*': (data_only[0], model_only[1], model_only[2])})
        primary = 'D*M*'
    elif m_gain > 0 or (d_gain <= 0 and m_gain >= d_gain):
        arms['D0M*'] = model_only
        primary = 'D0M*'
    else:
        arms['D*M0'] = data_only
        primary = 'D*M0'
    previous = {tuple(r['job'][k] for k in ('data_mode','mode','lr','seed')):r
                for r in [*model_pick['rows'], *rows] if r['job']['model'] == 'scgpt'}
    split = contract['initial_split']
    repeats = []
    for arm, (d,m,lr) in arms.items():
        for seed in contract['confirmation']['seeds']:
            job = spec('F','scgpt',m,d,split,seed,lr)
            old = previous.get((d,m,lr,seed))
            repeats.append(dict(arm=arm, job=old['job'] if old else job,
                                reused_validation=old is not None))
    return dict(
        dataset_sha256=dataset_sha, protocol_sha256=digest(PROTOCOL),
        selection_role='val', primary_metric='delta_rmse', primary_arm=primary,
        arms=arms, repeats=repeats, data_screen_rows=rows,
        validation_gain_model=m_gain, validation_gain_data=d_gain,
        data_screen_fixed_model=fixed['job'], model_screen_rows=model_pick['rows'])


def lock_test(plan, runs, baseline):
    if plan['protocol_sha256'] != digest(PROTOCOL) or plan['selection_role'] != 'val':
        raise ValueError('Confirmation plan was not derived from the declared validation protocol')
    simple = read(baseline)
    if (simple['role'] != 'val' or simple['dataset_sha256'] != plan['dataset_sha256']
            or simple['run_code_sha256'] != digest(ROOT/'src/common_baselines.py')):
        raise ValueError('Simple baselines were not fit on the frozen validation split')
    gears = next(r['job'] for r in plan['model_screen_rows'] if r['job']['model']=='gears')
    return dict(
        stage='frozen_test', plan_sha256=None,
        simple_baselines={'validation_sha256':digest(baseline),'selected_alpha':simple['selected_alpha']},
        benchmark_rows=[dict(arm='GEARS', **completed(runs,gears,plan['dataset_sha256']))],
        protocol_sha256=digest(PROTOCOL), dataset_sha256=plan['dataset_sha256'],
        primary_arm=plan['primary_arm'], primary_metric=plan['primary_metric'],
        rows=[dict(arm=r['arm'], **completed(runs,r['job'],plan['dataset_sha256']))
              for r in plan['repeats']])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['declare', 'model', 'confirm', 'freeze'])
    p.add_argument('--dataset', type=Path)
    p.add_argument('--runs', type=Path, default=ROOT/'assets/runs')
    p.add_argument('--prior', type=Path)
    p.add_argument('--model-selection', type=Path)
    p.add_argument('--simple-baselines', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    contract = read(PROTOCOL)
    if a.stage == 'declare':
        item = dict(jobs=declared_screen(contract), protocol_sha256=digest(PROTOCOL))
    elif a.stage == 'freeze':
        plan = read(a.prior)
        expected = choose_confirmation(contract, read(a.model_selection), a.runs,
                                       digest(a.dataset/'receipt.json'))
        expected['model_selection_sha256'] = digest(a.model_selection)
        if plan != json.loads(json.dumps(expected)):
            raise ValueError('Confirmation plan was not produced by the declared validation selection')
        item = lock_test(plan, a.runs, a.simple_baselines)
        item['plan_sha256'] = digest(a.prior)
    else:
        dataset_sha = digest(a.dataset/'receipt.json')
        if read(a.dataset/'receipt.json')['split_seed'] != contract['initial_split']:
            raise ValueError('Screening requires the initial common split')
        if a.stage == 'model':
            item = select_model(contract, a.runs, dataset_sha)
        else:
            item = choose_confirmation(contract, read(a.prior), a.runs, dataset_sha)
            item['model_selection_sha256'] = digest(a.prior)
    immutable(a.output, item)


if __name__ == '__main__':
    main()
