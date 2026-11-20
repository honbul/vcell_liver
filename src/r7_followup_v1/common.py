"""Paths and small helpers of the R7 follow-up (target attribution v1). Nothing here writes outside $R7F_ASSETS."""
import hashlib, json, os, time

TRACK = "r7_followup_v1"
VERSION = "followup-v1"
MAN = f"manifests/{TRACK}"
REP = f"reports/{TRACK}"
PRED = f"predictions/{TRACK}"
CFG = f"configs/{TRACK}/followup.json"
HCC = ("HUH7", "JHH5", "JHH7")
MODEL_SEEDS = (42, 43, 44)
NULL_SEEDS = (142, 143, 144)
N_FOLDS = 3


def cfg():
    return json.load(open(CFG))


def fruns(*p):
    return os.path.join(os.environ["R7F_RUNS"], *p)


def cache(*p):
    path = os.path.join(os.environ["R7F_CACHE"], *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def orig_runs(*p):
    """Original R7 runs: read only."""
    return os.path.join(os.environ["R7_ORIG_RUNS"], *p)


def pub(*p):
    path = os.path.join(os.environ["R7F_PUBLIC"], *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def pman(*p):
    return pub(MAN, *p)


def prep(*p):
    return pub(REP, *p)


def ppred(*p):
    return pub(PRED, *p)


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha_list(items):
    return hashlib.sha256("\n".join(map(str, items)).encode()).hexdigest()


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def new_run_dir(kind, run_id):
    """Immutable run id: an existing directory is an error."""
    out = fruns(kind, run_id)
    os.makedirs(out, exist_ok=False)
    return out


def run_record(run_id, kind, t0, **kw):
    return dict(run_id=run_id, kind=kind, seconds=round(time.time() - t0, 1), commit=os.environ.get("R7F_COMMIT", ""),
                slurm_job=os.environ.get("SLURM_JOB_ID", ""), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID", ""),
                host=os.uname().nodename, **kw)


def tag_cols(df, split_role, **kw):
    """Common provenance columns of every result table (plan section 13.1)."""
    df.insert(0, "track", "R7"); df.insert(1, "version", VERSION); df.insert(2, "split_role", split_role)
    for k, v in kw.items():
        df[k] = v
    return df
