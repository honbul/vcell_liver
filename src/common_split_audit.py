"""Audit the registered split and distinguish new conditions from new target genes."""
import argparse
from collections import Counter
from pathlib import Path

from artifact_io import digest
from common_selection import immutable, read

ROLES = ('train','val','test')


def audit(dataset):
    receipt = read(dataset/'receipt.json')
    files = ['genes.json','controls.json',*(f'{role}_conditions.json' for role in ROLES)]
    for name in files:
        if digest(dataset/name) != receipt['files'][name]:
            raise ValueError(f'Changed split manifest: {name}')
    genes = set(read(dataset/'genes.json')['output'])
    if len(genes) != receipt['shared_genes']:
        raise ValueError('Missing or duplicated output gene')
    controls = read(dataset/'controls.json')
    pool, eval_ids = (controls[k] for k in ('train_pool_ids','eval_ids'))
    if (len(set(pool)) != len(pool) or len(set(eval_ids)) != len(eval_ids)
            or len(eval_ids) != receipt['evaluation_controls'] or not set(eval_ids) <= set(pool)):
        raise ValueError('Control pool and fixed evaluation IDs do not agree')
    groups = {role:read(dataset/f'{role}_conditions.json') for role in ROLES}
    observed = set()
    treated = set()
    aliases = set()
    for role, table in groups.items():
        current = []
        for condition, row in table.items():
            targets = tuple(sorted(set(row['targets'])))
            if not (targets and set(targets) <= genes and '+'.join(targets)==condition):
                raise ValueError(f'Invalid shared target in {role}')
            if targets in observed or aliases.intersection(row['aliases']):
                raise ValueError(f'Intervention shared between roles: {condition}')
            observed.add(targets)
            aliases.update(row['aliases'])
            current.extend(row['treated_ids'])
        if (len(current)!=len(set(current)) or treated.intersection(current)
                or set(pool).intersection(current)):
            raise ValueError(f'Duplicate or non-treated cell in {role}')
        if receipt['roles'][role] != {'conditions':len(table),'treated_cells':len(current)}:
            raise ValueError(f'Split counts differ from preparation receipt: {role}')
        treated.update(current)
    train_targets = {g for row in groups['train'].values() for g in row['targets']}
    train_singles = {row['targets'][0] for row in groups['train'].values()
                     if len(row['targets'])==1}
    heldout = {}
    for role in ('val','test'):
        rows = {}
        for name,arity in [('single',1),('combination',2)]:
            conditions = [row['targets'] for row in groups[role].values() if len(row['targets'])==arity]
            rows[name] = dict(
                n=len(conditions),
                missing_train_single=dict(sorted(Counter(
                    sum(g not in train_singles for g in condition) for condition in conditions).items())),
                missing_any_train_target=dict(sorted(Counter(
                    sum(g not in train_targets for g in condition) for condition in conditions).items())))
        rows['additive_eligible'] = rows['combination']['missing_train_single'].get(0,0)
        heldout[role] = rows
    return dict(
        scope='Condition, target, cell and control metadata only; no outcome matrices opened',
        split_seed=receipt['split_seed'],dataset_receipt_sha256=digest(dataset/'receipt.json'),
        genes_sha256=receipt['files']['genes.json'], controls_sha256=receipt['files']['controls.json'],
        roles=receipt['roles'],n_distinct_train_targets=len(train_targets),n_train_singles=len(train_singles),
        heldout=heldout)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    immutable(a.output,audit(a.dataset))


if __name__=='__main__':
    main()
