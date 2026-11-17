"""Paths, hashing and small I/O helpers shared by every R2 module."""
import hashlib, json, os, time

TRACK = "episodic_residual_r2"
CFG = f"configs/{TRACK}/protocol.yaml"
MAN = f"manifests/{TRACK}"                     # committed manifests (read side)
B_MAN = "manifests/transigen_adaptation_b1"    # prior study, read-only
EXCLUDED_TARGETS = ("HUH7", "HUH751", "JHH5", "JHH7")


def proc():
    return os.environ["R2_PROC"]


def runs(*p):
    return os.path.join(os.environ["R2_RUNS"], *p)


def public(*p):
    """Server-side mirror of the repository tree for small, publishable outputs."""
    path = os.path.join(os.environ["R2_PUBLIC"], *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def new_run_dir(kind, run_id):
    """Immutable run id: an existing directory is an error."""
    out = runs(kind, run_id)
    os.makedirs(out, exist_ok=False)
    return out


def run_record(run_id, kind, t0, **kw):
    return dict(run_id=run_id, kind=kind, seconds=round(time.time() - t0, 1), commit=os.environ.get("R2_COMMIT", ""),
                slurm_job=os.environ.get("SLURM_JOB_ID", ""), host=os.uname().nodename, **kw)


def norm_name(s):
    return "".join(ch for ch in str(s).upper() if ch.isalnum())
