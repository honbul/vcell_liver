"""Fit common validation-only baselines, using only train response and a public GO prior."""
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


def validation(dataset, go):
    started = time.monotonic()
    contract = json.loads(PROTOCOL.read_text())
    manifest = json.loads((dataset / 'receipt.json').read_text())
    if manifest['contract_sha256'] != digest(PROTOCOL):
        raise ValueError('Dataset does not match the current protocol')
    registry = {r['file']: r['sha256'] for r in json.loads(
        (ROOT / 'manifests/public_receipt.json').read_text())}
    if digest(go) != registry['norman/go.csv']:
        raise ValueError('Unexpected GO prior')
    genes = json.loads((dataset / 'genes.json').read_text())['output']
    train_meta = json.loads((dataset / 'train_conditions.json').read_text())
    val = json.loads((dataset / 'val_conditions.json').read_text())
    names, y = train_responses(dataset)
    assert set(names) == set(train_meta) and not set(names) & set(val)
    reference = np.load(dataset / 'eval_controls.npy', allow_pickle=False).mean(axis=0, dtype=np.float64)
    mean = y.mean(axis=0)
    results = {
        'no_change': evaluate(dataset, 'val', {c: reference for c in val}),
        'train_mean': evaluate(dataset, 'val', {c: reference + mean for c in val})
    }
    singles = {train_meta[c]['targets'][0]: row for c, row in zip(names, y)
               if len(train_meta[c]['targets']) == 1}
    additive = {c: reference + sum(singles[g] for g in info['targets'])
                for c, info in val.items()
                if len(info['targets']) == 2 and all(g in singles for g in info['targets'])}
    results['additive'] = evaluate(dataset, 'val', additive, allow_partial=True) if additive else None
    train_features = prior_vectors(go, genes, [train_meta[c]['targets'] for c in names])
    val_features = prior_vectors(go, genes, [d['targets'] for d in val.values()])
    gram = (train_features @ train_features.T).toarray()
    cross = (val_features @ train_features.T).toarray()
    alpha_rows = []
    best = None
    for alpha in contract['linear']['alphas']:
        weights = np.linalg.solve(gram + alpha * np.eye(len(names)), y - mean)
        pred = dict(zip(val, reference + mean + cross @ weights))
        scored = evaluate(dataset, 'val', pred)
        score = scored['metrics']['delta_rmse']
        alpha_rows.append(dict(alpha=alpha, delta_rmse=score))
        if best is None or (score, alpha) < best[:2]:
            best = (score, alpha, scored)
    results['linear_go_ridge'] = best[2]
    eligible = set(additive)
    shared = {}
    for method, data in results.items():
        if data is not None and eligible:
            shared[method] = float(np.mean([r['delta_rmse'] for r in data['per_condition']
                                            if r['condition'] in eligible]))
    return dict(
        role='val', contract_sha256=digest(PROTOCOL), dataset_sha256=digest(dataset/'receipt.json'),
        go_sha256=digest(go), run_code_sha256=digest(Path(__file__)), alpha_candidates=alpha_rows,
        selected_alpha=best[1], total_seconds=time.monotonic()-started,
        additive_common_subset_n=len(eligible), delta_rmse_on_additive_subset=shared, methods=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    # Exclusive reservation; an existing result is never overwritten.
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock = a.output.with_suffix('.lock')
    lock.mkdir()
    try:
        if a.output.exists():
            raise FileExistsError(a.output)
        write(a.output, validation(a.dataset, PUBLIC/'norman/go.csv'))
    finally:
        lock.rmdir()


if __name__ == '__main__':
    main()
