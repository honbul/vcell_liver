"""Read a real M3 epoch checkpoint and verify adapter updates and frozen weights."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def fingerprint(handle):
    sha = hashlib.sha256()
    for block in iter(lambda: handle.read(8 << 20), b''):
        sha.update(block)
    handle.seek(0)
    return sha.hexdigest()


def inspect(run, reference):
    init = json.loads((run/'model_init.json').read_text())
    if init['mode'] != 'M3' or len(init['attention_projections']) != 12:
        raise ValueError('Expected the common scGPT M3 run')
    with reference.open('rb') as f:
        reference_hash = fingerprint(f)
        if reference_hash != init['checkpoint_sha256']:
            raise ValueError('Public checkpoint does not match run initialization')
        original = torch.load(f, map_location='cpu', weights_only=True)
    # Hold one descriptor so atomic checkpoint replacement cannot change the audited bytes.
    with (run/'resume.pt').open('rb') as f:
        checkpoint_hash = fingerprint(f)
        state = torch.load(f, map_location='cpu', weights_only=False)
    current = state['current']
    mounts = init['attention_projections']
    prefixes = {n:n.replace('.in_proj_weight', '.parametrizations.in_proj_weight')
                for n in mounts}
    for name in init['backbone_keys_loaded']:
        public_name = name.replace('in_proj_weight', 'Wqkv.weight').replace(
            'in_proj_bias', 'Wqkv.bias')
        trained_name = prefixes[name]+'.original' if name in prefixes else name
        if not torch.equal(current[trained_name], original[public_name]):
            raise ValueError(f'Frozen backbone tensor changed: {name}')
    adam = state['optimizer']
    names = list(init['trainable_parameters'])
    ids = [i for group in adam['param_groups'] for i in group['params']]
    if len(ids) != len(names) or set(ids) != set(adam['state']):
        raise ValueError('Optimizer parameters do not match initialization')
    by_name = {n:adam['state'][i] for n,i in zip(names,ids)}
    for name, item in by_name.items():
        if item['exp_avg'].shape != current[name].shape:
            raise ValueError(f'Unexpected optimizer shape: {name}')
    steps = sum(row['updates'] for row in state['history'])
    rows = []
    for name in mounts:
        row = {'projection': name}
        for direction in ('down','up'):
            param = prefixes[name]+'.0.'+direction
            item = by_name[param]
            norm = item['exp_avg_sq'].norm().item()
            if item['step'].item() != steps or not torch.isfinite(current[param]).all() or norm <= 0:
                raise ValueError(f'No finite optimizer update evidence: {param}')
            row[direction+'_adam_squared_gradient_norm'] = norm
        row['up_weight_norm'] = current[prefixes[name]+'.0.up'].norm().item()
        if row['up_weight_norm'] <= 0:
            raise ValueError(f'Adapter up weight is still zero: {name}')
        rows.append(row)
    return dict(run_id=run.name, epoch=state['epoch'], optimizer_steps=steps,
                checkpoint_sha256=checkpoint_hash, public_checkpoint_sha256=reference_hash,
                frozen_backbone_tensors_verified=len(init['backbone_keys_loaded']), projections=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(inspect(a.run, a.reference), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
