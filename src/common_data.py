"""Build an isolated common Norman dataset without importing author DE rankings."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
from scipy import sparse

from artifact_io import digest, write

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'protocols/common/norman-v1.json'
PUBLIC = ROOT / 'assets/public'


def key(seed, domain, item):
    return hashlib.sha256(f'{seed}\t{domain}\t{item}'.encode()).digest()


def targets(condition):
    parts = condition.split('+')
    genes = sorted(set(parts) - {'ctrl'})
    if condition != 'ctrl' and (len(genes) not in (1, 2) or len(parts) != 2 or
                                len(parts) != len(set(parts)) or '' in genes):
        raise ValueError(f'Unexpected condition: {condition}')
    return tuple(genes)


def condition_split(conditions, seed):
    def partition(names, domain):
        order = sorted(names, key=lambda c: key(seed, domain, c))
        n = (15 * len(order) + 50) // 100
        return {c: ('test' if i < n else 'val' if i < 2*n else 'train')
                for i, c in enumerate(order)}

    split = partition((c for c, g in conditions.items() if len(g) == 1), 'single')
    singles = {g[0] for c, g in conditions.items() if split.get(c) == 'train'}
    strata = defaultdict(list)
    for c, gs in conditions.items():
        if len(gs) == 2:
            strata[sum(g in singles for g in gs)].append(c)
    for n, names in strata.items():
        split.update(partition(names, f'combo-{n}'))
    return split


def decode(group, name):
    ds = group[name]
    if ds.dtype.kind in ('O', 'S'):
        return ds.asstr()[:]
    cat = group['__categories'][name].asstr()[:]
    codes = ds[:]
    if np.any(codes < 0):
        raise ValueError(f'Missing categories in {name}')
    return cat[codes]


def fixed_control(pool_ids, treated_id, seed, epoch=None):
    domain = 'fixed-train-control' if epoch is None else f'resample-train-control-{epoch}'
    return int.from_bytes(key(seed, domain, treated_id)[:8], 'big') % len(pool_ids)


def prepare(dst, seed, public):
    contract = json.loads(PROTOCOL.read_text())
    if seed not in [contract['split']['seed'], *contract['split']['reserve_seeds']]:
        raise ValueError('Split seed is outside the common contract')
    receipt = {r['file']: r['sha256'] for r in json.loads(
        (ROOT / 'manifests/public_receipt.json').read_text())}
    files = ['norman/perturb_processed.h5ad', 'scgpt/vocab.json']
    for name in files:
        if digest(public / name) != receipt[name]:
            raise ValueError(f'Source hash mismatch: {name}')
    vocab = json.loads((public / files[1]).read_text())
    with h5py.File(public / files[0], 'r') as f:
        obs, var = f['obs'], f['var']
        cells = decode(obs, obs.attrs['_index'])
        genes = decode(var, 'gene_name')
        original = decode(obs, 'condition')
        control_mask = original == 'ctrl'
        if set(obs.attrs['column-order']) != {'condition', 'cell_type', 'dose_val',
                                             'control', 'condition_name'}:
            raise ValueError('Unexpected obs schema; recheck batch/control contract')
        if set(decode(obs, 'cell_type')) != {'A549'}:
            raise ValueError('Unexpected source cell-line annotation')
        if len(set(cells)) != len(cells) or len(set(genes)) != len(genes):
            raise ValueError('Source identifiers are not unique')
        if not np.array_equal(control_mask, obs['control'][:].astype(bool)):
            raise ValueError('Source control indicator disagrees with condition')
        keep = np.array([g in vocab for g in genes])
        shared = set(genes[keep])
        aliases = defaultdict(list)
        excluded = {}
        counts = Counter(original)
        for name in sorted(counts.keys() - {'ctrl'}):
            gs = targets(name)
            missing = sorted(set(gs) - shared)
            if missing:
                excluded[name] = {'missing_targets': missing, 'cells': counts[name]}
            else:
                aliases['+'.join(gs)].append(name)
        groups = {c: tuple(c.split('+')) for c in aliases}
        split = condition_split(groups, seed)
        group_ids = {raw: c for c, names in aliases.items() for raw in names}
        canonical = np.array([group_ids.get(n, '') for n in original], dtype=str)
        x = f['X']
        if x.attrs['encoding-type'] != 'csr_matrix':
            raise ValueError('Expected public Norman CSR expression')
        matrix = sparse.csr_matrix((x['data'][:], x['indices'][:], x['indptr'][:]),
                                   shape=tuple(x.attrs['shape']))[:, keep].tocsr()
    if not np.isfinite(matrix.data).all():
        raise ValueError('Non-finite normalized expression')
    matrix.sort_indices()
    pool = np.flatnonzero(control_mask)
    pool = np.array(sorted(pool, key=lambda i: cells[i]))
    eval_idx = np.array(sorted(pool, key=lambda i: key(seed, 'eval-control', cells[i]))[
        :contract['control']['eval_max_per_batch']])
    if not len(eval_idx):
        raise ValueError('Missing matched controls')
    control_ids = cells[pool]
    sparse.save_npz(dst / 'controls_train.npz', matrix[pool], compressed=False)
    np.save(dst / 'eval_controls.npy', matrix[eval_idx].toarray())
    write(dst / 'controls.json', {
        'train_pool_ids': control_ids.tolist(), 'eval_ids': cells[eval_idx].tolist(),
        'batch': 'not recorded'
    })
    train = np.flatnonzero(np.array([split.get(c) == 'train' for c in canonical]))
    sparse.save_npz(dst / 'treated_train.npz', matrix[train], compressed=False)
    np.savez(dst / 'train_index.npz', cell_ids=np.asarray(cells[train], dtype=str),
             condition=canonical[train],
             fixed_control_index=np.array([fixed_control(control_ids, cells[i], seed) for i in train]))
    for role in ('val', 'test'):
        names = sorted(c for c in groups if split[c] == role)
        row_groups = [np.flatnonzero(canonical == c) for c in names]
        # The outcomes are only available through role-specific evaluation inputs.
        means = np.stack([np.asarray(matrix[ix].mean(axis=0)).ravel() for ix in row_groups])
        np.savez(dst / f'outcomes_{role}.npz', condition=np.array(names), expression_mean=means)
    for role in ('train', 'val', 'test'):
        names = sorted(c for c in groups if split[c] == role)
        write(dst / f'{role}_conditions.json', {
            c: {'targets': list(groups[c]), 'aliases': aliases[c],
                'treated_ids': np.asarray(cells[canonical == c], dtype=str).tolist()}
            for c in names})
    write(dst / 'genes.json', {'output': genes[keep].tolist(), 'excluded': genes[~keep].tolist()})
    write(dst / 'exclusions.json', excluded)
    write(dst / 'receipt.json', {
        'version': contract['version'], 'split_seed': seed, 'contract_sha256': digest(PROTOCOL),
        'preparation_sha256': digest(Path(__file__)), 'source_sha256': {n: receipt[n] for n in files},
        'source_shape': list(matrix.shape[:1]) + [len(genes)], 'shared_genes': int(keep.sum()),
        'control_cells': len(pool), 'evaluation_controls': len(eval_idx),
        'excluded_source_labels': len(excluded), 'excluded_canonical_interventions':
            len({targets(c) for c in excluded}), 'excluded_cells': sum(x['cells'] for x in excluded.values()),
        'roles': {r: {'conditions': sum(v == r for v in split.values()),
                      'treated_cells': sum(counts[a] for c in groups if split[c] == r for a in aliases[c])}
                  for r in ('train', 'val', 'test')},
        'files': {f.name: digest(f) for f in sorted(dst.iterdir())}
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-seed', type=int, default=26)
    p.add_argument('--public', type=Path, default=PUBLIC)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    dst = a.output or ROOT / f'assets/common/C0-Norman-v1-s{a.split_seed}'
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Atomic reservation keeps preparation processes from overwriting each other.
    lock = dst.with_suffix('.prepare-lock')
    lock.mkdir()
    tmp = None
    try:
        if dst.exists():
            raise FileExistsError(dst)
        tmp = Path(tempfile.mkdtemp(prefix=f'.{dst.name}-', dir=dst.parent))
        prepare(tmp, a.split_seed, a.public)
        tmp.rename(dst)
        print(dst / 'receipt.json', flush=True)
    finally:
        if tmp is not None and tmp.exists():
            shutil.rmtree(tmp)
        lock.rmdir()


if __name__ == '__main__':
    main()
