"""Extract compact execution evidence from complete paper-track Slurm runs."""
import csv
import json
import math
import re

from artifact_io import digest

CHECKPOINT = re.compile(r'^EPOCH_CHECKPOINT (\d+) (\S+)$',re.M)


def submissions(path):
    with path.open() as f:
        rows = dict((stage, job) for stage,job,*_ in csv.reader(f,delimiter='\t'))
    for stage in ('R1-seed1','R2','R1-seeds2-5'):
        if not rows[stage].split(';',1)[0].isdigit():
            raise ValueError(f'Invalid Slurm handle for {stage}')
    return {stage: job.split(';',1)[0] for stage,job in rows.items()}


def audit(log, end_marker, expected_epochs, resumes=()):
    points = []
    logs = []
    final_seen = 0
    sources = [(log,None),*resumes]
    for i,(path,resume_receipt) in enumerate(sources):
        if resume_receipt is not None:
            epoch = resume_receipt['epoch']
            if epoch == len(points) and not math.isclose(points[-1][1],
                    resume_receipt['selection_value'],rel_tol=1e-6,abs_tol=1e-10):
                raise ValueError(f'Restart score differs from last epoch: {path}')
            if epoch != len(points):
                # The writer atomically committed a checkpoint but was stopped
                # before emitting its stdout line. Bridge only the next epoch.
                if epoch != len(points)+1:
                    raise ValueError(f'Restart evidence does not join the epoch history: {path}')
                points.append((epoch,resume_receipt['selection_value']))
        raw = path.read_text(errors='replace').splitlines()
        found = [(int(e),float(v)) for e,v in CHECKPOINT.findall('\n'.join(raw))]
        for epoch,value in found:
            if epoch != len(points)+1 or not math.isfinite(value):
                raise ValueError(f'Unexpected epoch checkpoint history: {path}')
            points.append((epoch,value))
        markers = sum(line.startswith(end_marker+' ') for line in raw)
        if i != len(sources)-1 and markers:
            raise ValueError(f'An earlier attempt already exported the final run: {path}')
        final_seen += markers
        info = dict(name=path.name,sha256=digest(path),checkpoint_lines=len(found))
        if resume_receipt is not None:
            info['resumed_from'] = {k:resume_receipt[k] for k in
                                   ('epoch','checkpoint_sha256','receipt_sha256')}
        logs.append(info)
    if len(points) != expected_epochs or final_seen != 1:
        raise ValueError(f'Incomplete epoch history or final export marker: {log}')
    return dict(log_name=log.name,log_sha256=logs[0]['sha256'],logs=logs,epochs=len(points),
                checkpoints=[dict(epoch=e,selection_value=v) for e,v in points])


def resume_logs(ledger, logs, run_id, track, settings_sha, seed=None):
    if ledger is None or not ledger.is_file():
        return []
    entries = json.loads(ledger.read_text()).get(run_id,[])
    sources = []
    jobs = set()
    for entry in entries:
        job = str(entry['slurm_job_id'])
        if not job.isdigit() or job in jobs:
            raise ValueError(f'Invalid or duplicate resume job for {run_id}')
        jobs.add(job)
        receipt_file = (ledger.parent/entry['checkpoint_receipt']).resolve()
        if not receipt_file.is_relative_to(ledger.parent.resolve()):
            raise ValueError('Resume checkpoint receipt must be in the project log directory')
        receipt = json.loads(receipt_file.read_text())
        if (receipt['status'] != 'checkpoint_structure_verified' or receipt['run_id'] != run_id
                or receipt['track'] != track or receipt['settings_sha256'] != settings_sha
                or receipt['epoch']<=0 or not math.isfinite(receipt['selection_value'])):
            raise ValueError(f'Invalid resume checkpoint receipt for {run_id}')
        name = f'gears_fig2f_{job}_{seed}.log' if track=='R1' else f'scgpt_fig3a_{job}.log'
        sources.append((logs/name,dict(**receipt,receipt_sha256=digest(receipt_file))))
    return sources
