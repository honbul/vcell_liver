"""Fit train-only baselines and select ridge alpha on validation."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from artifact_io import digest, write
from common_data import ROOT, PUBLIC, PROTOCOL
from common_eval import evaluate
from common_prediction_io import export
from common_selection import PROTOCOL as SELECTION, read


def train_responses(dataset):
    control = sparse.load_npz(dataset / 'controls_train.npz')
    treated = sparse.load_npz(dataset / 'treated_train.npz')
    with np.load(dataset / 'train_index.npz', allow_pickle=False) as ix:
        condition, pair = ix['condition'], ix['fixed_control_index']
    names = sorted(set(condition))
    responses = []
    for name in names:
        rows = np.flatnonzero(condition == name)
        ct = np.asarray(control[pair[rows]].mean(axis=0, dtype=np.float64)).ravel()
        tx = np.asarray(treated[rows].mean(axis=0, dtype=np.float64)).ravel()
        responses.append(tx - ct)
    return names, np.stack(responses)


def prior_vectors(go, genes, interventions):
    index = {g: i for i, g in enumerate(genes)}
    source, target, weight = [], [], []
    with go.open() as f:
        for row in csv.DictReader(f):
            a, b = index.get(row['source']), index.get(row['target'])
            if a is not None and b is not None:
                source.append(a)
                target.append(b)
                weight.append(float(row['importance']))
    graph = sparse.csr_matrix((weight, (source, target)), shape=(len(genes), len(genes)))
    norm = np.sqrt(np.asarray(graph.power(2).sum(axis=1)).ravel())
    norm[norm == 0] = 1
    graph = sparse.diags(1/norm) @ graph
    features = sparse.hstack([sparse.eye(len(genes)), graph], format='csr')
    vec = []
    for targets in interventions:
        vec.append(sparse.csr_matrix(features[[index[g] for g in targets]].mean(axis=0)))
    return sparse.vstack(vec, format='csr')


def score(dataset, go, role='val', fixed_alpha=None):
    started = time.monotonic()
    contract = json.loads(PROTOCOL.read_text())
    manifest = json.loads((dataset / 'receipt.json').read_text())
    if manifest['contract_sha256'] != digest(PROTOCOL):
        raise ValueError('Dataset does not match the current protocol')
    inputs = ('genes.json','train_conditions.json',f'{role}_conditions.json','controls_train.npz',
              'treated_train.npz','train_index.npz','eval_controls.npy',f'outcomes_{role}.npz')
    for name in inputs:
        if digest(dataset/name) != manifest['files'][name]:
            raise ValueError(f'Dataset file hash mismatch: {name}')
    registry = {r['file']: r['sha256'] for r in json.loads(
        (ROOT / 'manifests/public_receipt.json').read_text())}
    if digest(go) != registry['norman/go.csv']:
        raise ValueError('Unexpected GO prior')
    genes = json.loads((dataset / 'genes.json').read_text())['output']
    train_meta = json.loads((dataset / 'train_conditions.json').read_text())
    targets = read(dataset/f'{role}_conditions.json')
    names, y = train_responses(dataset)
    assert set(names) == set(train_meta) and not set(names) & set(targets)
    reference = np.load(dataset / 'eval_controls.npy', allow_pickle=False).mean(axis=0, dtype=np.float64)
    mean = y.mean(axis=0)
    predictions = {'no_change':{c:reference for c in targets},
                   'train_mean':{c:reference + mean for c in targets}}
    results = {method:evaluate(dataset,role,pred) for method,pred in predictions.items()}
    singles = {train_meta[c]['targets'][0]: row for c, row in zip(names, y)
               if len(train_meta[c]['targets']) == 1}
    additive = {c: reference + sum(singles[g] for g in info['targets'])
                for c, info in targets.items()
                if len(info['targets']) == 2 and all(g in singles for g in info['targets'])}
    results['additive'] = evaluate(dataset, role, additive, allow_partial=True) if additive else None
    predictions['additive'] = additive
    train_features = prior_vectors(go, genes, [train_meta[c]['targets'] for c in names])
    val_features = prior_vectors(go, genes, [d['targets'] for d in targets.values()])
    gram = (train_features @ train_features.T).toarray()
    cross = (val_features @ train_features.T).toarray()
    alpha_rows = []
    best = None
    if (role == 'test') != (fixed_alpha is not None):
        raise ValueError('Test scoring requires one previously selected alpha')
    if fixed_alpha is not None and fixed_alpha not in contract['linear']['alphas']:
        raise ValueError('Frozen alpha is outside the declared grid')
    for alpha in contract['linear']['alphas'] if fixed_alpha is None else [fixed_alpha]:
        weights = np.linalg.solve(gram + alpha * np.eye(len(names)), y - mean)
        pred = dict(zip(targets, reference + mean + cross @ weights))
        scored = evaluate(dataset, role, pred)
        score = scored['metrics']['delta_rmse']
        alpha_rows.append(dict(alpha=alpha, delta_rmse=score))
        if best is None or (score, alpha) < best[:2]:
            best = (score, alpha, scored, pred)
    results['linear_go_ridge'] = best[2]
    predictions['linear_go_ridge'] = best[3]
    eligible = set(additive)
    shared = {}
    for method, data in results.items():
        if data is not None and eligible:
            shared[method] = float(np.mean([r['delta_rmse'] for r in data['per_condition']
                                            if r['condition'] in eligible]))
    result = dict(
        role=role, contract_sha256=digest(PROTOCOL), dataset_sha256=digest(dataset/'receipt.json'),
        go_sha256=digest(go), run_code_sha256=digest(Path(__file__)), alpha_candidates=alpha_rows,
        selected_alpha=best[1], total_seconds=time.monotonic()-started,
        additive_common_subset_n=len(eligible), delta_rmse_on_additive_subset=shared, methods=results)
    return result,genes,predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--paper-status', type=Path)
    p.add_argument('--freeze', type=Path)
    a = p.parse_args()
    role, alpha, freeze_hash = 'val', None, None
    if a.freeze is not None:
        plan = read(a.freeze)
        if (plan['stage'] != 'frozen_test' or plan['protocol_sha256'] != digest(SELECTION)
                or plan['dataset_sha256'] != digest(a.dataset/'receipt.json')):
            raise ValueError('Invalid test freeze')
        role = 'test'
        alpha = plan['simple_baselines']['selected_alpha']
        freeze_hash = digest(a.freeze)
    else:
        from common_submit import paper_status
        if a.paper_status is None:
            raise ValueError('Record both paper track statuses before fitting')
        paper_status(a.paper_status)
    # Exclusive reservation; an existing result is never overwritten.
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock = a.output.with_suffix('.lock')
    lock.mkdir()
    try:
        if a.output.exists():
            raise FileExistsError(a.output)
        result,genes,predictions = score(a.dataset,PUBLIC/'norman/go.csv',role,alpha)
        pred_path = a.output.with_suffix('.predictions.npz')
        prediction_hash = export(pred_path,genes,predictions)
        write(a.output,dict(**result,freeze_sha256=freeze_hash,prediction_sha256=prediction_hash))
    finally:
        lock.rmdir()


if __name__ == '__main__':
    main()
