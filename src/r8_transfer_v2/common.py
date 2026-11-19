"""Paths and small helpers of R8 observation-transfer v2 (corrections P0 and follow-up E1 / E2).

Read-only inputs: the original R8 store (R8_PROC), the R8 follow-up v1 run tree (R8F_V1_RUNS) and the committed manifests of
both earlier studies. Every v2 output goes under R8V2_RUNS (immutable run directories) or R8V2_PUBLIC (small tables mirrored
into manifests/r8_transfer_v2 and reports/r8_transfer_v2).
"""
import hashlib, json, os, subprocess, time

TRACK = "r8_transfer_v2"
V1 = "r8_followup_v1"
CFG = f"configs/{TRACK}/run_spec.yaml"
MAN = f"manifests/{TRACK}"
V1_MAN = f"manifests/{V1}"
TARGETS = ("HUH7", "JHH5", "JHH7")


def runs(*p):
    return os.path.join(os.environ["R8V2_RUNS"], *p)


def v1_runs(*p):
    return os.path.join(os.environ["R8F_V1_RUNS"], *p)


def public(*p):
    path = os.path.join(os.environ["R8V2_PUBLIC"], *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def pub_man(name):
    return public("manifests", TRACK, name)


def v1_man(name):
    return os.path.join(V1_MAN, name)


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


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


def log(*a):
    print(*a, flush=True)
