"""Read a local research checkpoint without changing training or exporting its weights."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from artifact_io import digest, write

ROOT = Path(__file__).resolve().parents[1]


def inspect(track, run_id):
    if not run_id.replace('-','').replace('_','').isalnum():
        raise ValueError('Invalid immutable run ID')
    run = ROOT/'assets/runs'/track/run_id
    settings_path = run/'settings.json'
    settings = json.loads(settings_path.read_text())
    checkpoint = run/'resume.pt'
    # One open descriptor pins the checkpoint inode across an atomic writer
    # replacement. The digest and deserialization describe exactly the same bytes.
    with checkpoint.open('rb') as f:
        state_hash = hashlib.sha256()
        for block in iter(lambda:f.read(8 << 20),b''):
            state_hash.update(block)
        fingerprint = state_hash.hexdigest()
        size = os.fstat(f.fileno()).st_size
        f.seek(0)
        payload = torch.load(f,map_location='cpu',weights_only=False)
    current, best = payload['current'], payload['best']
    if set(current) != set(best) or not current:
        raise ValueError('Current and best model layouts differ')
    if any(current[k].shape != best[k].shape or current[k].dtype != best[k].dtype
           for k in current):
        raise ValueError('Stored best model is not compatible with current model')
    opt = payload['optimizer']
    scheduler = payload['scheduler']
    groups = opt['param_groups']
    parameters = {p for g in groups for p in g['params']}
    if not opt['state'] or not set(opt['state']) <= parameters:
        raise ValueError('Missing optimizer state or invalid optimizer parameter IDs')
    rng = payload['rng']
    if set(rng) != {'python','numpy','torch','cuda'} or not rng['cuda'] or not len(rng['torch']):
        raise ValueError('Missing RNG state for restart')
    epoch = int(payload['epoch'])
    if epoch <= 0:
        raise ValueError('Checkpoint did not record a completed training epoch')
    value_key = {'R1':'min_val','R2':'score'}.get(track,'best_val')
    score = float(payload[value_key])
    if not math.isfinite(score):
        raise ValueError('Non-finite validation selection value')
    if track == 'R2' and not {'scaler','sampling_counts','patience','stopped'} <= set(payload):
        raise ValueError('scGPT AMP, sampling or early stop state is missing')
    if int(scheduler['last_epoch']) not in {epoch-1,epoch}:
        raise ValueError('Scheduler is incompatible with the completed epoch')
    return dict(
        status='checkpoint_structure_verified',track=track,run_id=run_id,epoch=epoch,
        checkpoint_sha256=fingerprint,checkpoint_bytes=size,settings_sha256=digest(settings_path),
        initial_job_id=settings['slurm_job_id'],model_state_tensors=len(current),
        optimizer_state_entries=len(opt['state']),scheduler_epoch=int(scheduler['last_epoch']),
        rng_domains=sorted(rng),selection_value=score,
        scope='CPU deserialization only; no resumed training was performed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track',choices=['R1','R2','C0','M','D','F'],required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write(a.output,inspect(a.track,a.run_id))


if __name__=='__main__':
    main()
