"""Model definitions as (candidate configurations, selection mode) and development summaries.

A *model* is a rule that, from the inner-fold loss table of one training set, picks a configuration
(per drug, or one for all drugs). The same rule is used in development (inner losses only) and in the
frozen nested evaluation (the picked configuration is then refitted on the whole training set).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_models as M  # noqa: E402

RIDGE_RATIO_NAME = {0.0: "whole", 1.0: "r1", 10.0: "r10", 100.0: "r100"}


def parse_key(s: str) -> tuple:
    p = s.split("|")
    if p[0] == "ridge":
        return ("ridge", p[1], float(p[2]))
    return ("nuc", float(p[1]), float(p[2]), float(p[3]))


def strength(key: tuple) -> tuple:
    """Sort key from weakest to strongest regularisation: lambda, then eta/lambda, then tau."""
    if key[0] == "ridge":
        return (key[2], M.KERNEL_RATIO.get(key[1], 0.0), 0.0)
    return (key[2], key[1], key[3])


def model_spec(cfg: dict, name: str) -> dict:
    """Candidate filter and selection mode for a named model (see configs: `models`)."""
    return cfg["models"][name]


def candidates(spec: dict, keys: list) -> list:
    """Row indices of the configurations that belong to a model, ordered weakest -> strongest."""
    out = []
    for i, k in enumerate(keys):
        if k[0] == "ridge":
            ok = k[1] in spec.get("ridge_kernels", []) and (spec.get("lambda") is None or k[2] in spec["lambda"])
        else:
            ok = (k[1] in spec.get("nuc_ratios", []) and k[3] in spec.get("tau_fractions", [])
                  and (spec.get("lambda") is None or k[2] in spec["lambda"]))
        if ok:
            out.append(i)
    assert out, f"no candidate configuration for {spec}"
    return sorted(out, key=lambda i: strength(keys[i]))


def choose(spec: dict, keys: list, loss: np.ndarray) -> np.ndarray:
    """Chosen row index per drug (constant across drugs for a global model)."""
    order = candidates(spec, keys)
    if spec["select"] == "per_drug":
        return M.select_per_drug(loss, order)
    return np.full(loss.shape[1], M.select_global(loss, order))


def load_inner(run_dirs, prior: str = "real", prefix: str = "r") -> dict:
    """{set id: dict(keys, loss, fam_loss?, families?, meta, records)} merged over run directories
    (later grids add configurations; a configuration may not appear twice)."""
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]
    out = {}
    for rd in run_dirs:
        for f in sorted(glob.glob(os.path.join(rd, "inner", f"{prefix}*__{prior}.npz"))):
            sid = os.path.basename(f).split("__")[0]
            z = np.load(f, allow_pickle=False)
            keys = [parse_key(s) for s in z["configs"]]
            item = dict(keys=keys, loss=z["loss"], meta=[json.loads(str(z["meta"]))],
                        records=json.loads(str(z["records"])))
            if "fam_loss" in z.files:
                item["fam_loss"], item["families"] = z["fam_loss"].astype(float), list(z["families"])
            if sid not in out:
                out[sid] = item
                continue
            cur = out[sid]
            assert not set(cur["keys"]) & set(keys), f"configuration repeated across runs for {sid}"
            cur["keys"] += keys
            cur["loss"] = np.vstack([cur["loss"], item["loss"]])
            cur["meta"] += item["meta"]
            cur["records"] += item["records"]
            if "fam_loss" in cur and "fam_loss" in item:
                assert cur["families"] == item["families"]
                cur["fam_loss"] = np.concatenate([cur["fam_loss"], item["fam_loss"]], axis=0)
            else:
                cur.pop("fam_loss", None)
    return out


def dev_score(inner: dict, spec: dict) -> tuple:
    """Development score of a model: mean over training sets of the macro (over drugs) sqrt of the
    inner loss at the chosen configuration. Also returns the chosen keys per set."""
    per_set, chosen = {}, {}
    for sid, it in inner.items():
        idx = choose(spec, it["keys"], it["loss"])
        per_set[sid] = float(np.sqrt(it["loss"][idx, np.arange(it["loss"].shape[1])]).mean())
        chosen[sid] = [it["keys"][i] for i in idx]
    return float(np.mean(list(per_set.values()))), per_set, chosen


def dev_boot(inner: dict, spec: dict, all_families: list, W: np.ndarray) -> np.ndarray:
    """Family-bootstrap replicates of the development score with the chosen configurations held fixed.
    W: [boot, family] resampling counts over `all_families`. A family's inner-fold errors in every
    training set that contains it move together."""
    pos = {f: i for i, f in enumerate(all_families)}
    total = np.zeros(W.shape[0])
    for sid, it in inner.items():
        idx = choose(spec, it["keys"], it["loss"])
        L = it["fam_loss"][idx, :, np.arange(len(idx))]            # [drug, family]
        w = W[:, [pos[f] for f in it["families"]]]                 # [boot, family]
        num = w @ L.T                                              # [boot, drug]
        total += np.sqrt(num / w.sum(axis=1, keepdims=True)).mean(axis=1)
    return total / len(inner)
