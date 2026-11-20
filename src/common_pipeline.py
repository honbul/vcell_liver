"""Advance common experiments after successful Slurm jobs and immutable validation evidence."""
import argparse
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

from artifact_io import digest
from common_effects import summarize
from common_reserve import make_plan, lock
from common_reports import report as write_tables
from common_selection import (
    PROTOCOL, ROOT, choose_confirmation, immutable, lock_test, read, select_model,
)
from common_submit import paper_status, submit, tasks

STAGES = ('data', 'confirmation', 'freeze', 'summary',
          'reserve-plan', 'reserve-freeze', 'reserve-summary', 'close')


def dependency(handles):
    numbers = sorted({int(h.split(';',1)[0]) for h in handles})
    return [f'--dependency=afterok:{":".join(map(str,numbers))}'] if numbers else []


def transition(a):
    work = a.primary_workspace.resolve()
    primary_data = a.primary_dataset.resolve()
    paper = work/'paper_status.json'
    paper_status(paper)
    cfg = read(PROTOCOL)
    runs = ROOT/'assets/runs'
    journal = ROOT/'logs/common_submissions'
    journal.mkdir(parents=True,exist_ok=True)
    primary_freeze, primary_report = work/'frozen_test.json', work/'confirmation_summary.json'
    model = work/'model_selection.json'
    plan = work/'confirmation_plan.json'

    def spawn(stage, split, after=()):
        args = ['experiments/common_pipeline.sbatch',stage,'--primary-workspace',str(work),
                '--primary-dataset',str(primary_data),'--split',str(split)]
        return submit(f'pipeline-{stage}-s{split}',dependency(after)+args,journal)

    def launch(stage, dataset, scope, selected):
        args = SimpleNamespace(stage=stage,dataset=dataset,workspace=scope,paper_status=paper,
                               plan=selected,afterok=None)
        return [submit(key,cmd,journal) for key,cmd in tasks(args)]

    stage = a.stage
    split = a.split
    main_split = cfg['initial_split']
    if read(primary_data/'receipt.json')['split_seed'] != main_split:
        raise ValueError('Primary dataset does not match the declared split')
    dsha = digest(primary_data/'receipt.json')
    if stage in ('data','confirmation','freeze','summary','close') and split != main_split:
        raise ValueError('Primary stage has the wrong split ID')
    if stage.startswith('reserve') and split not in cfg['confirmation']['reserve_splits']:
        raise ValueError('Reserve stage has an undeclared split')

    if stage == 'data':
        if read(model) != select_model(cfg,runs,dsha):
            raise ValueError('Model selection is not reproducible from validated runs')
        spawn('confirmation',split,launch('D',primary_data,work,model))
    elif stage in ('confirmation','freeze'):
        selected = choose_confirmation(cfg,read(model),runs,dsha)
        selected['model_selection_sha256'] = digest(model)
        if stage == 'confirmation':
            immutable(plan,json.loads(json.dumps(selected)))
            spawn('freeze',split,launch('F',primary_data,work,plan))
        else:
            if read(plan) != json.loads(json.dumps(selected)):
                raise ValueError('Confirmation plan no longer matches the validation screen')
            frozen = lock_test(selected,runs,work/'simple_validation.json')
            frozen['plan_sha256'] = digest(plan)
            immutable(primary_freeze,frozen)
            spawn('summary',split,launch('test',primary_data,work,primary_freeze))
    elif stage == 'summary':
        result = summarize(primary_freeze,work/'scores',work/'simple_test.json')
        immutable(primary_report,result)
        reserve = []
        for seed in result['followup_reserve_splits']:
            data = ROOT/'assets/common'/f'C0-Norman-v1-s{seed}'
            prep = submit(f'C0-prepare-reserve-s{seed}',
                          ['experiments/common_prepare.sbatch','--split-seed',str(seed),
                           '--output',str(data)],journal)
            # Queue the reserve pipeline in order; only the earliest transition
            # is ready until the artifacts it requires have been produced.
            reserve.append(spawn('reserve-plan',seed,[prep]))
        if not reserve:
            spawn('close',split)
    elif stage.startswith('reserve'):
        dataset = ROOT/'assets/common'/f'C0-Norman-v1-s{split}'
        scope = work/f'reserve-s{split}'
        reserve_plan = scope/'confirmation_plan.json'
        frozen_path = scope/'frozen_test.json'
        selected = make_plan(primary_freeze,primary_report,dataset)
        if stage == 'reserve-plan':
            immutable(reserve_plan,selected)
            spawn('reserve-freeze',split,launch('F',dataset,scope,reserve_plan))
        elif stage == 'reserve-freeze':
            if read(reserve_plan) != selected:
                raise ValueError('Reserve plan no longer matches the frozen primary comparison')
            frozen = lock(selected,runs)
            frozen['plan_sha256'] = digest(reserve_plan)
            immutable(frozen_path,frozen)
            spawn('reserve-summary',split,launch('test',dataset,scope,frozen_path))
        else:
            result = summarize(frozen_path,scope/'scores')
            immutable(scope/'confirmation_summary.json',result)
            # The final reserve dispatcher coordinates only after both report
            # artifacts exist; writes happen atomically before this check.
            expected = [work/f'reserve-s{s}'/'confirmation_summary.json'
                        for s in read(primary_report)['followup_reserve_splits']]
            with (work/'.reserve_close.lock').open('a') as latch:
                fcntl.flock(latch,fcntl.LOCK_EX)
                if all(p.exists() for p in expected):
                    spawn('close',main_split)
    else:
        report = read(primary_report)
        evidence = ['paper_status.json','model_selection.json','confirmation_plan.json',
                    'frozen_test.json','simple_validation.json','simple_test.json',
                    'confirmation_summary.json']
        for seed in report['followup_reserve_splits']:
            evidence.extend(f'reserve-s{seed}/{name}' for name in
                            ('confirmation_plan.json','frozen_test.json','confirmation_summary.json'))
        immutable(work/'common_evidence.json',dict(
            status='common_experiments_executed',dataset_sha256=dsha,
            primary_arm=report['primary_arm'],reserve_splits=report['followup_reserve_splits'],
            files={name:digest(work/name) for name in evidence}))
        write_tables(work,work/'tables')
    print('COMMON_TRANSITION_COMPLETE',stage,split,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=STAGES)
    p.add_argument('--primary-workspace',type=Path,required=True)
    p.add_argument('--primary-dataset',type=Path,required=True)
    p.add_argument('--split',type=int,required=True)
    transition(p.parse_args())


if __name__=='__main__':
    main()
