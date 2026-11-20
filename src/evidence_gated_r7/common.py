"""Paths, hashing and small I/O helpers shared by every R7 module."""
import hashlib, json, os, time

TRACK = "evidence_gated_r7"
CFG = f"configs/{TRACK}/protocol.yaml"
MAN = f"manifests/{TRACK}"                     # committed manifests (read side)
R4_MAN = "manifests/response_geometry_r4"      # R4 split / families (read-only, committed in R4)
HCC = ("HUH7", "JHH5", "JHH7")


def ext(*p):
    return os.path.join(os.environ["R7_EXT"], *p)


def kg(*p):
    return os.path.join(os.environ["R7_KG"], *p)


def r4proc(*p):
    return os.path.join(os.environ["R4_PROC"], *p)


def tb(*p):
    return os.path.join(os.environ["TB_PROC"], *p)


def runs(*p):
    return os.path.join(os.environ["R7_RUNS"], *p)


def public(*p):
    """Server-side mirror of the repository tree for small, publishable outputs."""
    path = os.path.join(os.environ["R7_PUBLIC"], *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def pman(*p):
    return public(MAN, *p)


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
    return dict(run_id=run_id, kind=kind, seconds=round(time.time() - t0, 1), commit=os.environ.get("R7_COMMIT", ""),
                slurm_job=os.environ.get("SLURM_JOB_ID", ""), host=os.uname().nodename, **kw)
