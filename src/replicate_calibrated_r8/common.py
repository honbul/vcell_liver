"""Paths, hashing and small I/O helpers of study R8 (replicate-calibrated, state-preserving adaptation)."""
import hashlib, json, os, time

TRACK = "replicate_calibrated_r8"
CFG = f"configs/{TRACK}/protocol.yaml"
MAN = f"manifests/{TRACK}"
R2_MAN = "manifests/episodic_residual_r2"          # predecessor study, read-only
B_MAN = "manifests/transigen_adaptation_b1"        # study B, read-only
TARGETS = ("HUH7", "JHH5", "JHH7")
N_GENES = 978


def env(k):
    return os.environ[k]


def proc(*p):
    return os.path.join(os.environ["R8_PROC"], *p)


def runs(*p):
    return os.path.join(os.environ["R8_RUNS"], *p)


def public(*p):
    """Server-side mirror of the repository tree for small, publishable outputs."""
    path = os.path.join(os.environ["R8_PUBLIC"], *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def man_file(name, committed_only=False):
    """Committed manifest, or the not-yet-committed server copy while a stage is being developed."""
    c = os.path.join(MAN, name)
    if committed_only:
        return c
    p = os.path.join(os.environ["R8_PUBLIC"], "manifests", TRACK, name)
    return p if os.path.exists(p) else c


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
    return dict(run_id=run_id, kind=kind, seconds=round(time.time() - t0, 1), commit=os.environ.get("R8_COMMIT", ""),
                slurm_job=os.environ.get("SLURM_JOB_ID", ""), host=os.uname().nodename, **kw)
