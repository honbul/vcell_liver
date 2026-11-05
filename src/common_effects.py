"""Summarize prespecified paired test contrasts without choosing a new candidate."""
import argparse
from pathlib import Path

import numpy as np

from artifact_io import digest
from common_selection import PROTOCOL, immutable, read


def effect(differences, bootstrap, rng):
    # Rows are training seeds and columns are interventions. Keep the same
    # intervention index across all arms and seeds and average seeds first.
    condition_mean = differences.mean(axis=0)
    sample = rng.integers(len(condition_mean), size=(bootstrap['draws'], len(condition_mean)))
    draws = condition_mean[sample].mean(axis=1)
    by_seed = differences.mean(axis=1)
    return dict(mean_gain=float(condition_mean.mean()), per_seed_gain=by_seed.tolist(),
                seed_range=[float(by_seed.min()),float(by_seed.max())],
                condition_gains=condition_mean.tolist(),
                condition_bootstrap_95=[float(x) for x in np.quantile(draws,[.025,.975])])


def arm_matrix(arms, arm, seeds, conditions):
    return np.asarray([[arms[arm][seed][c]['delta_rmse'] for c in conditions] for seed in seeds])


def summarize(freeze_path, scores, simple_path=None):
    freeze = read(freeze_path)
    if freeze['stage'] != 'frozen_test' or freeze['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('Expected declared frozen test bundle')
    freeze_sha = digest(freeze_path)
    arm_metrics, arms = {}, {}
    conditions = None
    for row in [*freeze['rows'], *freeze.get('benchmark_rows',[])]:
        arm, job = row['arm'], row['job']
        item = read(scores/f"{job['run_id']}.json")
        if (item['freeze_sha256'] != freeze_sha or item['dataset_sha256'] != freeze['dataset_sha256']
                or item['run_id'] != job['run_id'] or item['arm'] != arm or item['seed'] != job['seed']
                or item['checkpoint_sha256'] != row['checkpoint_sha256']):
            raise ValueError('Test prediction did not use the frozen run')
        if digest(scores/f"{job['run_id']}.predictions.npz") != item['prediction_sha256']:
            raise ValueError('Prediction artifact differs from the scored expression')
        scored = item['metrics']
        per = {c['condition']:c for c in scored['per_condition']}
        if scored['role'] != 'test' or len(per) != scored['n_available']:
            raise ValueError('Missing frozen test predictions')
        if conditions is None:
            conditions = sorted(per)
        if sorted(per) != conditions:
            raise ValueError('Test condition sets differ')
        target = arms.setdefault(arm,{})
        if job['seed'] in target:
            raise ValueError('Duplicate seed in arm')
        target[job['seed']] = per
        arm_metrics.setdefault(arm,[]).append(dict(
            seed=job['seed'], metrics=scored['metrics'], score_sha256=digest(scores/f"{job['run_id']}.json"),
            prediction_sha256=item['prediction_sha256'],
            inference_seconds=item.get('seconds'), inference_gpu=item.get('gpu'),
            inference_peak_gib=item.get('peak_allocated_gib')))
    base = 'D0M0'
    primary = freeze['primary_arm']
    cfg = read(PROTOCOL)['confirmation']
    bootstrap = cfg['bootstrap']
    expected = [cfg['reserve_training_seed']] if freeze.get('reserve') else cfg['seeds']
    seeds = sorted(arms[base])
    if seeds != expected:
        raise ValueError('Unexpected confirmatory learning seeds')
    orig = arm_matrix(arms, base, seeds, conditions)
    matrices, results = {}, {}
    for arm, items in arms.items():
        if arm == 'GEARS':
            continue
        if sorted(items) != seeds:
            raise ValueError('Test arms must share the training seed set')
        matrices[arm] = arm_matrix(arms,arm,seeds,conditions)
        if arm != base:
            results[arm] = effect(orig - matrices[arm],bootstrap,np.random.default_rng(bootstrap['seed']))
    if {'D*M0','D0M*','D*M*'} <= matrices.keys():
        results['interaction'] = effect(
            matrices['D*M0'] + matrices['D0M*'] - orig - matrices['D*M*'],
            bootstrap,np.random.default_rng(bootstrap['seed']))
    benchmark = None
    if simple_path is not None:
        simple = read(simple_path)
        if (simple['role'] != 'test' or simple['freeze_sha256'] != freeze_sha or
            simple['dataset_sha256'] != freeze['dataset_sha256'] or
            simple['selected_alpha'] != freeze['simple_baselines']['selected_alpha']):
            raise ValueError('Simple baseline report does not match frozen test')
        full = {}
        for name in ('no_change','train_mean','linear_go_ridge'):
            method = simple['methods'][name]
            if sorted(r['condition'] for r in method['per_condition']) != conditions:
                raise ValueError(f'Incomplete shared test coverage for {name}')
            full[name] = method['metrics']
        additive = simple['methods']['additive']
        subset = sorted(r['condition'] for r in additive['per_condition']) if additive else []
        comparable = simple['delta_rmse_on_additive_subset']
        if subset:
            for arm in (base,'GEARS'):
                comparable[arm] = float(np.mean([arms[arm][0][c]['delta_rmse'] for c in subset]))
        full['GEARS'] = arm_metrics['GEARS'][0]['metrics']
        full['scGPT'] = next(r['metrics'] for r in arm_metrics[base] if r['seed']==0)
        benchmark = dict(complete_coverage=full, additive_n=len(subset),
                         additive_subset_delta_rmse=comparable,
                         simple_baselines_test_sha256=digest(simple_path))
    if not freeze.get('reserve') and benchmark is None:
        raise ValueError('Primary split requires the full common benchmark')
    main = results[primary]
    promising = (not freeze.get('reserve') and all(x > 0 for x in main['per_seed_gain'])
                 and main['condition_bootstrap_95'][0] > 0)
    return dict(
        freeze_sha256=freeze_sha, dataset_sha256=freeze['dataset_sha256'], primary_arm=primary,
        conditions=conditions, arm_metrics=arm_metrics, seed_order=seeds, effects=results,
        c0_benchmark=benchmark, followup_reserve_splits=cfg['reserve_splits'] if promising else [],
        limitations=[
            'Learning seeds are not independent biological samples.',
            'Intervals are descriptive and conditional on observed interventions, averaged over training seeds.',
            'Shared target genes and control cells can induce dependence; no cell bootstrap or p values.',
            'The only prespecified primary test is the frozen candidate versus D0M0; factorial contrasts are descriptive.'
        ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--simple-baselines', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    immutable(a.output,summarize(a.freeze,a.scores,a.simple_baselines))


if __name__ == '__main__':
    main()
