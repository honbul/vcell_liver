"""If the frozen primary candidate qualifies, lock the same pair on each reserve split."""
import argparse
from pathlib import Path

from artifact_io import digest
from common_selection import PROTOCOL, completed, immutable, read, spec


def make_plan(freeze, summary, dataset):
    original, report = read(freeze), read(summary)
    if (original['stage'] != 'frozen_test' or original.get('reserve')
            or original['protocol_sha256'] != digest(PROTOCOL)
            or report['freeze_sha256'] != digest(freeze)
            or report['primary_arm'] != original['primary_arm']):
        raise ValueError('Reserve plan requires primary test evidence')
    cfg = read(PROTOCOL)['confirmation']
    if report['followup_reserve_splits'] != cfg['reserve_splits']:
        raise ValueError('Additional split trigger did not pass')
    receipt = read(dataset/'receipt.json')
    split = receipt['split_seed']
    if split not in cfg['reserve_splits'] or receipt['version'] != 'C0-Norman-v1':
        raise ValueError('Undeclared reserve dataset')
    repeats = []
    for arm in ('D0M0',original['primary_arm']):
        source = next(r['job'] for r in original['rows'] if r['arm']==arm)
        job = spec('F',source['model'],source['mode'],source['data_mode'],split,
                   cfg['reserve_training_seed'],source['lr'])
        repeats.append(dict(arm=arm,job=job,reused_validation=False))
    return dict(dataset_sha256=digest(dataset/'receipt.json'),protocol_sha256=digest(PROTOCOL),
                primary_freeze_sha256=digest(freeze),primary_report_sha256=digest(summary),
                reserve_split=split,primary_arm=original['primary_arm'],primary_metric='delta_rmse',
                repeats=repeats)


def lock(plan, runs):
    return dict(stage='frozen_test',reserve=True,protocol_sha256=digest(PROTOCOL),
                dataset_sha256=plan['dataset_sha256'],primary_arm=plan['primary_arm'],
                primary_metric=plan['primary_metric'],reserve_split=plan['reserve_split'],
                rows=[dict(arm=r['arm'], **completed(runs,r['job'],plan['dataset_sha256']))
                      for r in plan['repeats']])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary-freeze', type=Path, required=True)
    p.add_argument('--primary-summary', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--runs', type=Path)
    p.add_argument('--plan', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    item = make_plan(a.primary_freeze,a.primary_summary,a.dataset)
    if a.plan is not None:
        if read(a.plan) != item:
            raise ValueError('Reserve plan no longer matches the frozen candidate')
        item = lock(item,a.runs)
        item['plan_sha256'] = digest(a.plan)
    immutable(a.output,item)


if __name__ == '__main__':
    main()
