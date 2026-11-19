"""Paths, hashing and small I/O helpers of the R7 XPert v2 track (private plan 2026-11-19 section 4).

Read-only inputs: the original R7 XPert inputs ($XP_WORK), the upstream XPert working copy ($XP_UPSTREAM), the original
XPert runs ($R7_ORIG_ASSETS/runs/xpert) and the R4 unit store (through evidence_gated_r7.data.Store).
Writes: only under $R7V2_ASSETS (runs, public mirror)."""
import hashlib, json, os, time

TRACK = "r7_xpert_followup_v2"
MAN = f"manifests/{TRACK}"
CFG = f"configs/{TRACK}"
HCC = ("HUH7", "JHH5", "JHH7")
SEEDS = (42, 43, 44)
UPSTREAM_REF = "d53ff497e465692c80fac7b899e3cc5cc8e7bfed"
ORIG_XPERT_RUNS = {"official": ["xpr_e100_s42", "xpr_e100_s43", "xpr_e100_s44"],
                   "ecfp": ["xpe_e100_s42", "xpe_e100_s43", "xpe_e100_s44"]}


def runs(*p):
    return os.path.join(os.environ["R7V2_RUNS"], *p)


def orig_xpert(*p):
    return os.path.join(os.environ["R7_ORIG_ASSETS"], "runs", "xpert", *p)


def xp_work(*p):
    return os.path.join(os.environ["XP_WORK"], *p)


def public(*p):
    path = os.path.join(os.environ["R7V2_PUBLIC"], *p)
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


def sha_array(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()).hexdigest()


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def stamp(**kw):
    return dict(commit=os.environ.get("R7V2_COMMIT", ""), slurm_job=os.environ.get("SLURM_JOB_ID", ""),
                host=os.uname().nodename, time=time.strftime("%Y-%m-%dT%H:%M:%S"), **kw)
