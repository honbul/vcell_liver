"""Shared paths, protocol access and small helpers for study R5."""
import hashlib, json, os, subprocess
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PROTOCOL = os.path.join(REPO, "configs", "perturbation_design_r5", "protocol.yaml")
MAN = os.path.join(REPO, "manifests", "perturbation_design_r5")
B_MAN = os.path.join(REPO, "manifests", "transigen_adaptation_b1")
TARGETS = ["HUH7", "JHH5", "JHH7"]


def env(name, default=None):
    v = os.environ.get(name, default)
    if v is None:
        raise RuntimeError(f"environment variable {name} is not set (source experiments/perturbation_design_r5/env.sh)")
    return v


def proc():
    return env("R5_PROC")


def runs():
    return env("R5_RUNS")


def public(*parts):
    p = os.path.join(env("R5_PUBLIC"), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def protocol():
    with open(PROTOCOL) as f:
        return yaml.safe_load(f)


def sha_hex(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


def nproc():
    return int(os.environ.get("R5_NPROC", "4"))
