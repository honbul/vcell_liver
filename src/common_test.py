"""Score one validation-selected checkpoint on the frozen common test split."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

import numpy as np
import torch

from artifact_io import digest
from common_eval import evaluate
from common_prediction_io import export
from common_runtime import expression_model, predict_conditions
from common_selection import NEURAL, PROTOCOL, completed, immutable, read


def inputs(dataset, frozen):
    receipt = read(dataset/'receipt.json')
    if digest(dataset/'receipt.json') != frozen['dataset_sha256']:
        raise ValueError('Test data differs from the frozen dataset')
    for file in ('genes.json','test_conditions.json','eval_controls.npy','outcomes_test.npz'):
        if digest(dataset/file) != receipt['files'][file]:
            raise ValueError(f'Test data hash mismatch: {file}')
    genes = read(dataset/'genes.json')['output']
    return genes, read(dataset/'test_conditions.json'), np.load(dataset/'eval_controls.npy', allow_pickle=False)


def run(freeze_path, dataset, runs, run_id, output):
    if output.exists():
        raise FileExistsError(output)
    frozen = read(freeze_path)
    if frozen['stage'] != 'frozen_test' or frozen['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('Missing frozen common evaluation plan')
    selected = [r for r in [*frozen['rows'], *frozen.get('benchmark_rows', [])]
                if r['job']['run_id'] == run_id]
    if len(selected) != 1:
        raise ValueError('Run must be uniquely registered in the frozen plan')
    row = selected[0]
    job = row['job']
    verified = completed(runs,job,frozen['dataset_sha256'])
    if any(row[k] != v for k,v in verified.items()):
        raise ValueError('Checkpoint is not the frozen model')
    if not torch.cuda.is_available():
        raise RuntimeError('Use the common GPU test wrapper')
    genes, meta, control = inputs(dataset, frozen)
    index = {g:i for i,g in enumerate(genes)}
    path = runs/job['phase']/run_id
    graph = None
    if job['model'] == 'gears':
        graph = path/'train_graphs.npz'
        expected = read(path/'result.json')['model_init']['graphs_sha256']
        if digest(graph) != expected:
            raise ValueError('GEARS train graph hash mismatch')
    config = json.loads(NEURAL.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    reservation = output.with_suffix(output.suffix+'.running')
    reservation.mkdir()
    try:
        started = time.monotonic()
        with TemporaryDirectory(prefix='common-test-',dir=output.parent) as tmp:
            device = torch.device('cuda:0')
            model, predict = expression_model(SimpleNamespace(genes=genes), job['model'], job['mode'],
                                              Path(tmp), device, graph_path=graph)
            state = torch.load(path/'best_model.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            predictions = predict_conditions(predict, control, meta,
                                             lambda item:[index[g] for g in item['targets']],
                                             device, config['validation_control_chunk'])
        scored = evaluate(dataset,'test',predictions)
        artifact = output.with_suffix('.predictions.npz')
        prediction_hash = export(artifact,genes,{'native':predictions})
        immutable(output, dict(arm=row['arm'], seed=job['seed'], run_id=run_id,
                               freeze_sha256=digest(freeze_path),
                               dataset_sha256=frozen['dataset_sha256'],
                               checkpoint_sha256=row['checkpoint_sha256'],
                               prediction_sha256=prediction_hash,
                               gpu=torch.cuda.get_device_name(),seconds=time.monotonic()-started,
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                               metrics=scored))
    finally:
        reservation.rmdir()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.freeze,a.dataset,a.runs,a.run_id,a.output)


if __name__ == '__main__':
    main()
