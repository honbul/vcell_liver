"""F1-diag: regenerate *validation* expression predictions from saved best checkpoints.

The prior study saved only per-condition validation metrics for neural runs, not the
predicted expression vectors. A5 (retrieval, shrinkage, centered correlation) needs the
vectors, so this script reloads each run's ``best_model.pt`` read-only, predicts the 34
validation interventions on the 32 fixed evaluation controls with the original
``predict_conditions`` path, checks that the per-condition delta RMSE replays
``best_validation.json``, and exports predictions with the original ``export`` format.

No test outcome is read. Prior assets are never written: upstream sources are copied
into a private ``fake_root`` under the new assets directory first.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

SCGPT_RUNS = {
    'M0': 'C0/C0-Norman-v1-scgpt-M0-D0-s26-r0',
    'M1': 'M/M-Norman-v1-scgpt-M1-D0-s26-r0',
    'M2': 'M/M-Norman-v1-scgpt-M2-D0-s26-r0',
    'M3': 'M/M-Norman-v1-scgpt-M3-D0-s26-r0',
    'M-random': 'M/M-Norman-v1-scgpt-M-random-D0-s26-r0',
    'D1': 'D/D-Norman-v1-scgpt-M0-D1-s26-r0',
    'D2': 'D/D-Norman-v1-scgpt-M0-D2-s26-r0',
    'D0-seed1': 'F/F-Norman-v1-scgpt-M0-D0-s26-r1',
    'D0-seed2': 'F/F-Norman-v1-scgpt-M0-D0-s26-r2',
    'D1-seed1': 'F/F-Norman-v1-scgpt-M0-D1-s26-r1',
    'D1-seed2': 'F/F-Norman-v1-scgpt-M0-D1-s26-r2',
}
GEARS_RUNS = {'GEARS': 'C0/C0-Norman-v1-gears-M0-D0-s26-r0'}


def fake_root(prior_root, out):
    """Private copy of the upstream sources so imports never touch prior directories."""
    root = out / 'fake_root'
    up = root / '.upstream'
    up.mkdir(parents=True, exist_ok=True)
    for name in ('scgpt-paper.zip', 'gears002.tar.gz'):
        if not (up / name).exists():
            shutil.copy2(prior_root / '.upstream' / name, up / name)
    if not (up / 'gears-paper').exists():
        shutil.copytree(prior_root / '.upstream' / 'gears-paper', up / 'gears-paper', symlinks=True)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', choices=['scgpt', 'gears'], required=True)
    p.add_argument('--prior-root', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    import common_data
    import scgpt_author
    from common_eval import evaluate
    from common_prediction_io import export
    from common_runtime import expression_model, predict_conditions
    from types import SimpleNamespace

    a.out.mkdir(parents=True, exist_ok=True)
    froot = fake_root(a.prior_root, a.out)
    scgpt_author.ROOT = froot  # only used to locate the verified upstream archives
    if a.runtime == 'scgpt':
        import common_scgpt
        common_scgpt.PUBLIC = a.public
        runs = SCGPT_RUNS
    else:
        import common_gears
        common_gears.ROOT = froot
        runs = GEARS_RUNS
    common_data.PUBLIC = a.public

    config = json.loads((ROOT / 'protocols/common/neural-v1.json').read_text())
    genes = json.loads((a.dataset / 'genes.json').read_text())['output']
    index = {g: i for i, g in enumerate(genes)}
    meta = json.loads((a.dataset / 'val_conditions.json').read_text())
    control = np.load(a.dataset / 'eval_controls.npy', allow_pickle=False)
    device = torch.device('cuda:0')
    pred_dir = a.out / 'val_predictions'
    pred_dir.mkdir(exist_ok=True)
    receipt = {}
    for label, rel in runs.items():
        run = a.runs / rel
        started = time.monotonic()
        result = json.loads((run / 'result.json').read_text())
        work = a.out / 'model_build' / label
        work.mkdir(parents=True, exist_ok=True)
        graph = run / 'train_graphs.npz' if a.runtime == 'gears' else None
        model, predict = expression_model(SimpleNamespace(genes=genes), a.runtime, result['mode'],
                                          work, device, graph_path=graph)
        state = torch.load(run / 'best_model.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        preds = predict_conditions(predict, control, meta, lambda item: [index[g] for g in item['targets']],
                                   device, config['validation_control_chunk'])
        scored = evaluate(a.dataset, 'val', preds)
        saved = json.loads((run / 'best_validation.json').read_text())
        ref = {r['condition']: r['delta_rmse'] for r in saved['per_condition']}
        diffs = [abs(r['delta_rmse'] - ref[r['condition']]) for r in scored['per_condition']]
        path = pred_dir / f'{label}.npz'
        sha = export(path, genes, {'native': preds})
        receipt[label] = dict(
            run=str(rel), mode=result['mode'], data_mode=result['data_mode'], seed=result['seed'],
            best_epoch=result['best_epoch'], saved_val_delta_rmse=saved['metrics']['delta_rmse'],
            replay_val_delta_rmse=scored['metrics']['delta_rmse'],
            max_abs_condition_delta_rmse_diff=max(diffs), prediction_npz=str(path), prediction_sha256=sha,
            seconds=time.monotonic() - started)
        print('F1_DIAG_INFER', label, json.dumps(receipt[label]), flush=True)
        del model, state
        torch.cuda.empty_cache()
    out = a.out / f'infer_receipt_{a.runtime}.json'
    out.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
                                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                                   runs=receipt), indent=2) + '\n')


if __name__ == '__main__':
    main()
