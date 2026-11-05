"""Collect finished paper-track runs and compare with the pinned author references."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

from artifact_io import digest, write
from paper_audit import audit, resume_logs, submissions

ROOT = Path(__file__).resolve().parents[1]
R1_SEEDS = range(1, 6)


def load_json(path):
    return json.loads(path.read_text())


def tsv(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def check_result(path, checks):
    item = load_json(path)
    if item['status'] != 'modified_reproduction_executed':
        raise ValueError(f'Unexpected paper result status: {path}')
    for name, expected_key in checks.items():
        if digest(path.parent/name) != item[expected_key]:
            raise ValueError(f'Paper result artifact hash mismatch: {name}')
    return item


def collect(runs, output, r2_id='R2-manuscript-public-v1', r1_stem='R1-paper-public-v1',
            submission_log=None, strict=False, table_output=None, resume_ledger=None):
    submitted = submissions(submission_log) if submission_log else None
    if strict and not submitted:
        raise ValueError('Strict paper status needs the project submission ledger')
    if submitted and resume_ledger is None:
        resume_ledger = submission_log.parent/'paper_resume_ledger.json'
    ref = tsv(ROOT/'manifests/reference/gears_fig2f_author.tsv')
    author = {(r['method'], int(r['split_seed']), r['metric']):float(r['value']) for r in ref}
    sub = defaultdict(list)
    evidence = []
    complete = []
    for seed in R1_SEEDS:
        path = runs/'R1'/f'{r1_stem}-s{seed}'/'result.json'
        if not path.exists():
            continue
        r = check_result(path, {'split.json':'split_sha256',
                                'prediction.npz':'prediction_sha256','model/model.pt':'model_sha256'})
        if r['split_seed'] != seed:
            raise ValueError(f'Split seed mismatch: {path}')
        table = path.with_name('fig2f_conditions.tsv')
        split = load_json(path.with_name('split.json'))
        if split['n_source'] != 91205 or split['n_genes'] != 5045:
            raise ValueError('Unexpected public Norman dataset shape')
        item = dict(seed=seed, result_sha256=digest(path), model_sha256=r['model_sha256'],
                    prediction_sha256=r['prediction_sha256'],
                    condition_metrics_sha256=digest(table), split_sha256=r['split_sha256'],
                    cells=split['n_cells'], n_train_batches=r['n_train_batches'])
        if submitted:
            job = submitted['R1-seed1' if seed==1 else 'R1-seeds2-5']
            logs = submission_log.parent
            restarts = resume_logs(resume_ledger,logs,path.parent.name,'R1',
                                   digest(path.with_name('settings.json')),seed)
            item['execution'] = audit(logs/f'gears_fig2f_{job}_{seed}.log',
                                      'R1_FINISHED',20,restarts)
        evidence.append(item)
        raw = defaultdict(list)
        groups = tsv(table)
        listed = [row['condition'] for row in groups]
        if set(listed) != set(split['conditions']['test']) or len(listed) != len(set(listed)):
            raise ValueError(f'Test subgroup rows disagree with the held-out condition manifest: {table}')
        for row in groups:
            raw[row['group']].append(row)
        for group in ('combo_seen0','combo_seen1','combo_seen2'):
            cells = raw[group]
            n_seen = int(group[-1])
            if not cells:
                raise ValueError(f'Missing group {group}: {path}')
            sub[group].append(dict(
                seed=seed, n_conditions=len(cells),
                group_disagreements=sum(int(x['n_train_singles']) != n_seen for x in cells),
                gears=mean(float(x['gears']) for x in cells),
                no_change=mean(float(x['no_perturb']) for x in cells)))
        complete.append(seed)
    comparison = []
    for n_seen in (2, 1, 0):
        group = f'combo_seen{n_seen}'
        key = f'test_{group}_mse_top20_de_non_dropout'
        pub_mean = mean(author['GEARS',s,key] for s in R1_SEEDS)
        pub_ctrl = mean(author['No perturb',s,key] for s in R1_SEEDS)
        result = dict(panel='GEARS Fig. 2f', unseen_targets=2-n_seen,
                      author_mse=pub_mean, author_no_change_mse=pub_ctrl, author_ratio=pub_mean/pub_ctrl,
                      run_seeds=complete, per_seed=sub[group])
        if complete:
            local_mean = mean(x['gears'] for x in sub[group])
            local_ctrl = mean(x['no_change'] for x in sub[group])
            result.update(run_mse=local_mean, run_no_change_mse=local_ctrl,
                          run_ratio=local_mean/local_ctrl, run_mse_over_author_no_change=local_mean/pub_ctrl,
                          run_ratio_minus_author=local_mean/local_ctrl - pub_mean/pub_ctrl)
        comparison.append(result)
    scgpt = load_json(ROOT/'manifests/reference/scgpt_fig3a_author.json')['metrics']
    sc_path = runs/'R2'/r2_id/'result.json'
    sc_rerun = None
    if sc_path.exists():
        sc_rerun = check_result(sc_path, {'split.json':'split_sha256','best_model.pt':'model_sha256',
                                        'prediction.npz':'prediction_sha256',
                                        'manuscript_runtime.py':'runtime_sha256'})
    sc_evidence = None
    if sc_rerun:
        split = load_json(sc_path.with_name('split.json'))
        if split['n_source'] != 91205 or split['n_genes'] != 5045 or not 1 <= sc_rerun['epochs'] <= 15:
            raise ValueError('Unexpected scGPT reproduction dataset or epoch count')
        loaded = load_json(sc_path.with_name('checkpoint_load.json'))
        if not loaded['tensor_values_verified'] or loaded['vocab_matched'] != 4547:
            raise ValueError('Checkpoint or vocabulary verification failed')
        sc_evidence = dict(result_sha256=digest(sc_path),split_sha256=sc_rerun['split_sha256'],
                           model_sha256=sc_rerun['model_sha256'],
                           prediction_sha256=sc_rerun['prediction_sha256'], cells=split['n_cells'],
                           checkpoint_load_sha256=digest(sc_path.with_name('checkpoint_load.json')))
        if submitted:
            job = submitted['R2']
            logs = submission_log.parent
            restarts = resume_logs(resume_ledger,logs,sc_path.parent.name,'R2',
                                   digest(sc_path.with_name('settings.json')))
            sc_evidence['execution'] = audit(logs/f'scgpt_fig3a_{job}.log',
                                             'R2_FINISHED',sc_rerun['epochs'],restarts)
    sc_table = []
    for key, approximate in [('pearson_delta',0.58), ('pearson_de_delta',0.77)]:
        row = dict(metric=key, author_log=scgpt[key], approx_publisher_figure=approximate)
        if sc_rerun:
            value = sc_rerun['metrics'][key]
            row.update(run=value, run_minus_author_log=value-scgpt[key])
        sc_table.append(row)
    result = {
        'R1': {'status': 'modified_reproduction_executed' if len(complete)==5 else 'incomplete',
               'evidence': evidence, 'comparisons': comparison},
        'R2': {'status': 'modified_reproduction_executed' if sc_rerun else 'incomplete',
               'evidence': sc_evidence,
               'comparisons': sc_table, 'rerun_metrics': sc_rerun}
    }
    if strict and (len(complete)!=5 or sc_rerun is None):
        raise ValueError('Strict handoff needs five GEARS runs and the complete scGPT replay')
    for track in result.values():
        track['comparison_class'] = 'public_asset_modified_reproduction'
        track['exact_equivalence'] = 'unverified_original_private_assets'
    if table_output is not None:
        fields = ['track','metric','group','author_value','publisher_approximate',
                  'local_value','difference','n_local_repetitions','status']
        with table_output.open('w') as f:
            table = csv.DictWriter(f,fieldnames=fields,delimiter='\t')
            table.writeheader()
            for row in comparison:
                table.writerow(dict(track='R1',metric='normalized_mse_top20_de_non_dropout',
                                    group=f"{row['unseen_targets']} unseen targets",
                                    author_value=row['author_ratio'],local_value=row.get('run_ratio'),
                                    difference=row.get('run_ratio_minus_author'),n_local_repetitions=len(complete),
                                    status=result['R1']['status']))
            for row in sc_table:
                table.writerow(dict(track='R2',metric=row['metric'],group='Norman',
                                    author_value=row['author_log'],publisher_approximate=row['approx_publisher_figure'],
                                    local_value=row.get('run'),difference=row.get('run_minus_author_log'),
                                    n_local_repetitions=int(sc_rerun is not None),status=result['R2']['status']))
    write(output,result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=ROOT/'assets/runs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--r2-id', default='R2-manuscript-public-v1')
    p.add_argument('--r1-stem', default='R1-paper-public-v1')
    p.add_argument('--submission-log',type=Path)
    p.add_argument('--strict',action='store_true')
    p.add_argument('--table',type=Path)
    p.add_argument('--resume-ledger',type=Path)
    a = p.parse_args()
    collect(a.runs,a.output,a.r2_id,a.r1_stem,a.submission_log,a.strict,a.table,a.resume_ledger)


if __name__ == '__main__':
    main()
