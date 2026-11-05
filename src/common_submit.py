"""Submit one registered common experiment stage through Slurm exactly once."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from artifact_io import digest, write
from common_selection import ROOT, PROTOCOL, declared_screen, read

TERMINAL = {'modified_reproduction_executed','quantitative_match','quantitative_mismatch',
            'asset_unavailable','execution_failed'}


def paper_status(path):
    status = read(path)
    for track in ('R1','R2'):
        if status[track]['status'] not in TERMINAL or not status[track]['evidence']:
            raise ValueError(f'Paper-track status is not recorded: {track}')
    return digest(path)


def training_command(job, dataset, paper, plan):
    wrapper = ['experiments/common_train.sbatch',job['model'],'--dataset',str(dataset),
               '--paper-status',str(paper)]
    if plan is not None:
        wrapper.extend(['--stage-plan',str(plan)])
    for name in ('phase','run_id','mode','data_mode','seed','lr'):
        wrapper.extend(['--'+name.replace('_','-'),str(job[name])])
    slurm = ['--gres=gpu:a40:1'] if job['model']=='gears' else []
    return slurm + wrapper


def tasks(a):
    cfg = read(PROTOCOL)
    stage, dataset = a.stage, a.dataset.resolve()
    split = read(dataset/'receipt.json')['split_seed']
    dsha = digest(dataset/'receipt.json')
    work = a.workspace.resolve()
    if stage != 'test':
        if a.paper_status is None:
            raise ValueError('Paper status evidence required')
        paper_status(a.paper_status)
    if stage in ('C0','M'):
        if split != cfg['initial_split'] or a.plan is not None:
            raise ValueError('Screening only uses the declared primary split')
        jobs = [j for j in declared_screen(cfg) if j['phase']==stage]
    else:
        if a.plan is None:
            raise ValueError(f'{stage} requires a frozen plan')
        plan = read(a.plan)
        if plan['protocol_sha256'] != digest(PROTOCOL) or plan['dataset_sha256'] != dsha:
            raise ValueError('Plan does not match the selected dataset or protocol')
        if stage == 'D':
            jobs = plan['jobs']
        elif stage == 'F':
            jobs = [r['job'] for r in plan['repeats'] if not r['reused_validation']]
        else:
            if plan['stage'] != 'frozen_test':
                raise ValueError('Test needs a frozen checkpoint bundle')
            jobs = [r['job'] for r in [*plan['rows'],*plan.get('benchmark_rows',[])]]
    entries = []
    if stage == 'C0':
        entries.append((f'C0-simple-s{split}', [
            'experiments/common_baselines.sbatch','--dataset',str(dataset),
            '--paper-status',str(a.paper_status),
            '--output',str(work/'simple_validation.json')]))
    if stage == 'test':
        if 'simple_baselines' in plan:
            entries.append((f'test-simple-s{split}', [
                'experiments/common_baselines.sbatch','--dataset',str(dataset),'--freeze',str(a.plan),
                '--output',str(work/'simple_test.json')]))
        for j in jobs:
            entries.append((f"test-{j['run_id']}", [
                'experiments/common_test.sbatch',j['model'],'--dataset',str(dataset),
                '--freeze',str(a.plan),'--runs',str(ROOT/'assets/runs'),'--run-id',j['run_id'],
                '--output',str(work/'scores'/f"{j['run_id']}.json")]))
    else:
        entries.extend((j['run_id'],training_command(j,dataset,a.paper_status,a.plan)) for j in jobs)
    return entries


def submit(key, args, journal):
    if not key.replace('-','').replace('_','').isalnum():
        raise ValueError(f'Unsafe job key: {key}')
    job_name = 'rf_' + hashlib.sha256(key.encode()).hexdigest()[:16]
    claim = journal/f'{key}.json'
    lock = claim.with_suffix('.lock')
    try:
        lock.mkdir()
    except FileExistsError:
        # May have been interrupted between sbatch and saving its handle.
        # Reconcile with scheduler; never automatically retry this uncertain case.
        if not claim.is_file():
            raise RuntimeError(f'Unresolved submission reservation: {lock}; lookup {job_name}')
        if read(claim)['args'] != args:
            raise RuntimeError(f'Registered command differs: {claim}')
        print(key, read(claim)['slurm_job_id'], 'existing')
        return
    acknowledged = False
    try:
        if claim.is_file():
            if read(claim)['args'] != args:
                raise RuntimeError(f'Registered command differs: {claim}')
            print(key, read(claim)['slurm_job_id'], 'existing')
            return
        # A failed or interrupted sbatch leaves the reservation for reconciliation.
        # A time-limited job is resumed separately with the original run ID.
        handle = subprocess.check_output(['sbatch','--parsable','--job-name='+job_name,*args],
                                         cwd=ROOT,text=True).strip()
        if not handle.split(';',1)[0].isdigit():
            raise RuntimeError(f'Unrecognized Slurm handle: {handle}')
        write(claim,dict(job=key,job_name=job_name,slurm_job_id=handle,args=args))
        acknowledged = True
        print(key, handle, flush=True)
    finally:
        if acknowledged or claim.is_file():
            lock.rmdir()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['C0','M','D','F','test'])
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--paper-status', type=lambda s:Path(s).resolve())
    p.add_argument('--plan', type=lambda s:Path(s).resolve())
    p.add_argument('--submit', action='store_true', help='Without this flag, only print commands')
    a = p.parse_args()
    jobs = tasks(a)
    if a.submit:
        journal = ROOT/'logs/common_submissions'
        journal.mkdir(parents=True,exist_ok=True)
        for key,args in jobs:
            submit(key,args,journal)
    else:
        for key,args in jobs:
            print(key,shlex.join(['sbatch',*args]))


if __name__ == '__main__':
    main()
