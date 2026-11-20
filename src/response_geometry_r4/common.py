"""Paths, hashing and small I/O helpers shared by every R4 module."""
import hashlib, json, os, time

TRACK = "response_geometry_r4"
CFG = f"configs/{TRACK}/protocol.yaml"
MAN = f"manifests/{TRACK}"                     # committed manifests (read side)
B_MAN = "manifests/transigen_adaptation_b1"    # prior study B, read-only
HCC = ("HUH7", "JHH5", "JHH7")
SPLIT_SEED = 20261117


def proc(*p):
    return os.path.join(os.environ["R4_PROC"], *p)


def runs(*p):
    return os.path.join(os.environ["R4_RUNS"], *p)


def tb(*p):
    return os.path.join(os.environ["TB_PROC"], *p)


def public(*p):
    """Server-side mirror of the repository tree for small, publishable outputs."""
    path = os.path.join(os.environ["R4_PUBLIC"], *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


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
    out = runs(kind, run_id)
    os.makedirs(out, exist_ok=False)
    return out


def run_record(run_id, kind, t0, **kw):
    return dict(run_id=run_id, kind=kind, seconds=round(time.time() - t0, 1), commit=os.environ.get("R4_COMMIT", ""),
                slurm_job=os.environ.get("SLURM_JOB_ID", ""), host=os.uname().nodename, **kw)


def norm_name(s):
    return "".join(ch for ch in str(s).upper() if ch.isalnum())
