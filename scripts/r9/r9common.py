"""R9 shared paths, provenance and donor-level statistics helpers."""
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('R9_ROOT', Path(__file__).resolve().parents[2]))
ASSETS = ROOT / 'assets'
REPORT = ROOT / 'reports/r9_rare_state'
MANIFEST = ROOT / 'manifests/r9_rare_state'
CACHE = ASSETS / 'cache'
RUNS = ASSETS / 'runs'

GEO149614 = Path('/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE149614')
GEO151530 = Path('/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE151530')
# Deterministic sparse conversion of the GSE149614 raw counts produced by the earlier [G1] work
# (scripts/01_prepare_hcc.py); reused read-only, hash recorded in manifests/local_artifacts.json.
PROCESSED149614 = Path('/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/processed/GSE149614.h5ad')
PROCESSED149614_SHA = 'b234f5ca91c5655ede829e7520c07d361b578f8d1b59bb09d60a49976f48be6d'

BROAD = ['B', 'Endothelial', 'Fibroblast', 'Hepatocyte', 'Myeloid', 'T/NK']
# Compartment parent used for the fallback output level (defined before any model run).
PARENT = {'B': 'Immune', 'T/NK': 'Immune', 'Myeloid': 'Immune',
          'Endothelial': 'Stromal', 'Fibroblast': 'Stromal', 'Hepatocyte': 'Parenchymal'}
# GSE151530 author 'Type' -> GSE149614 broad label; fixed in A0 before any R9 model touches GSE151530.
EXT_MAP = {'T cells': 'T/NK', 'B cells': 'B', 'TAMs': 'Myeloid', 'TECs': 'Endothelial', 'CAFs': 'Fibroblast',
           'Malignant cells': 'UNMAPPED_malignant', 'unclassified': 'UNMAPPED_unclassified'}
G1_TEST_DONORS = ['HCC02', 'HCC09']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def write_tsv(df, path, **kw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep='\t', index=False, na_rep='NA', float_format=kw.pop('float_format', '%.6g'), **kw)


class Timer:
    def __init__(self):
        self.t0 = time.monotonic()
        self.c0 = time.process_time()

    def finish(self):
        return dict(wall_seconds=round(time.monotonic() - self.t0, 2),
                    cpu_seconds=round(time.process_time() - self.c0, 2),
                    peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 3),
                    host=os.uname().nodename, slurm_job=os.environ.get('SLURM_JOB_ID'),
                    slurm_task=os.environ.get('SLURM_ARRAY_TASK_ID'))


def new_run_dir(run_id):
    """Immutable run directory: refuse to overwrite an existing completed run."""
    d = RUNS / run_id
    if (d / 'DONE').exists():
        raise SystemExit(f'immutable run already completed: {run_id}')
    d.mkdir(parents=True, exist_ok=True)
    return d


def auprc(y, s):
    """Average precision; NaN (undefined, not 0) when there are no positives."""
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return np.nan
    return float(average_precision_score(y, s))


def donor_paired_ci(diff, n_boot=10000, seed=0):
    """Mean of per-donor paired differences, donor-bootstrap 95% percentile CI and t-interval."""
    from scipy import stats
    d = np.asarray([x for x in diff if np.isfinite(x)], float)
    n = len(d)
    if n == 0:
        return dict(n_donors=0, mean=np.nan, boot_lo=np.nan, boot_hi=np.nan, t_lo=np.nan, t_hi=np.nan, n_pos=0)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, n, (n_boot, n))].mean(1) if n > 1 else np.array([d[0]])
    m = d.mean()
    if n > 1:
        se = d.std(ddof=1) / np.sqrt(n)
        tq = stats.t.ppf(0.975, n - 1)
        tlo, thi = m - tq * se, m + tq * se
    else:
        tlo = thi = np.nan
    return dict(n_donors=n, mean=m, boot_lo=np.quantile(bs, 0.025), boot_hi=np.quantile(bs, 0.975),
                t_lo=tlo, t_hi=thi, n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()))
