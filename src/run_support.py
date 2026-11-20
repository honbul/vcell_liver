"""Small on-disk contracts shared by new run entry points."""
import hashlib, json, os, random, fcntl, subprocess
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'assets/public'
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
def claim_run(track, run_id, protocol, settings, resume):
    if not run_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Run ID must be a single safe name')
    out = ROOT / 'assets/runs' / track / run_id
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'result.json').exists():
        raise RuntimeError(f'Immutable finished run: {out}')
    record = out / 'settings.json'
    spec = dict(protocol_sha256=digest(protocol), assets_receipt_sha256=digest(PUBLIC/'receipt.json'), **settings)
    if record.exists():
        if not resume or json.loads(record.read_text())['contract'] != spec:
            raise RuntimeError('Existing run requires explicit resume and identical input/settings')
    else:
        write(record, dict(contract=spec, project_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(), slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    return out, lock
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
def checkpoint(path, obj):
    tmp = path.with_suffix('.pt.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)
def link_dataset(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    d = root / 'norman'
    d.mkdir(exist_ok=True)
    for name in ('perturb_processed.h5ad','go.csv'):
        f=d/name
        if not f.exists():
            f.symlink_to(PUBLIC/'norman'/name)
        if f.resolve() != (PUBLIC/'norman'/name).resolve():
            raise RuntimeError(f'Unexpected data source: {f}')
    return lock
def split_record(pert_data):
    d={k:sorted(map(str,v)) for k,v in pert_data.set2conditions.items()}
    assert not any(set(d[i])&set(d[j]) for i,j in [('train','val'),('train','test'),('val','test')])
    counts=pert_data.adata.obs.condition.astype(str).value_counts()
    return dict(conditions=d, n_cells={k:int(counts.reindex(v).sum()) for k,v in d.items()}, n_genes=pert_data.adata.n_vars, n_source=pert_data.adata.n_obs)
