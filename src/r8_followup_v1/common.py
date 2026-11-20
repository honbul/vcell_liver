"""Paths and small helpers of the R8 follow-up (observation-transfer analysis v1).

The original R8 study is read-only: its store (R8_PROC), checkpoints (R8_RUNS) and committed manifests are only read.
Every output of the follow-up goes under R8F_RUNS (immutable run directories) or R8F_PUBLIC (small tables that are
mirrored into manifests/r8_followup_v1 and reports/r8_followup_v1).
"""
import hashlib, json, os, subprocess, time

TRACK = "r8_followup_v1"
ORIG_TRACK = "replicate_calibrated_r8"
ORIG_MAN = f"manifests/{ORIG_TRACK}"
ORIG_CFG = f"configs/{ORIG_TRACK}/protocol.yaml"
CFG = f"configs/{TRACK}/followup.yaml"
MAN = f"manifests/{TRACK}"
N_GENES = 978
TARGETS = ("HUH7", "JHH5", "JHH7")
BOOT_SEED = 20261117
N_BOOT = 2000


def runs(*p):
    return os.path.join(os.environ["R8F_RUNS"], *p)


def public(*p):
    path = os.path.join(os.environ["R8F_PUBLIC"], *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def pub_man(name):
    return public("manifests", TRACK, name)


def man_file(name):
    """Follow-up manifest: server copy while a stage is developed, else the committed copy."""
    p = os.path.join(os.environ.get("R8F_PUBLIC", "/nonexistent"), "manifests", TRACK, name)
    return p if os.path.exists(p) else os.path.join(MAN, name)


def orig_man(name):
    """Original R8 manifest, committed copy only (read-only)."""
    return os.path.join(ORIG_MAN, name)


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def new_run_dir(kind, run_id):
    """Immutable run id: an existing directory is an error."""
    out = runs(kind, run_id)
    os.makedirs(out, exist_ok=False)
    return out


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "experiments", "tests"], capture_output=True, text=True).stdout.strip()
        return head, bool(dirty)
    except Exception:
        return "", True


def run_record(run_id, kind, t0, **kw):
    head, dirty = git_state()
    return dict(run_id=run_id, kind=kind, seconds=round(time.time() - t0, 1), commit=head, dirty_tree=dirty,
                slurm_job=os.environ.get("SLURM_JOB_ID", ""), host=os.uname().nodename, **kw)
