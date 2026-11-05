"""Collect finished paper-track runs and compare with the pinned author references."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

from artifact_io import digest, write

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


def collect(runs, output, r2_id='R2-manuscript-public-v1', r1_stem='R1-paper-public-v1'):
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
        evidence.append({'seed': seed, 'result_sha256': digest(path),
                         'condition_metrics_sha256': digest(table), 'split_sha256':r['split_sha256']})
        raw = defaultdict(list)
        for row in tsv(table):
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
               'evidence': {'result_sha256': digest(sc_path), 'split_sha256': sc_rerun['split_sha256']}
                           if sc_rerun else None,
               'comparisons': sc_table, 'rerun_metrics': sc_rerun}
    }
    write(output, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=ROOT/'assets/runs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--r2-id', default='R2-manuscript-public-v1')
    p.add_argument('--r1-stem', default='R1-paper-public-v1')
    a = p.parse_args()
    collect(a.runs, a.output, a.r2_id, a.r1_stem)


if __name__ == '__main__':
    main()
