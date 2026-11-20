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
    if p[0] == "res":
        return ("res", float(p[1]), float(p[2]), float(p[3]), float(p[4]))
    return ("nuc", float(p[1]), float(p[2]), float(p[3]))


def strength(key: tuple) -> tuple:
    """Sort key from weakest to strongest regularisation: lambda, then eta/lambda, then tau."""
    if key[0] == "ridge":
        return (key[2], M.KERNEL_RATIO.get(key[1], 0.0), 0.0)
    if key[0] == "res":
        return (key[2], key[1], key[3], key[4])
    return (key[2], key[1], key[3])


def group_of(key: tuple) -> tuple:
    """Hybrid selection: the part of a configuration shared by all drugs. Plain whole-space Ridge is the
    group with no shared fit (its members are the per-drug lambda values)."""
    if key[0] == "res":
        return (key[2], key[1], key[3])
    assert key[0] == "ridge" and key[1] == "whole"
    return (-1.0, 0.0, 0.0)


def model_spec(cfg: dict, name: str) -> dict:
    """Candidate filter and selection mode for a named model (see configs: `models`)."""
    spec = dict(cfg["models"][name])
    spec.setdefault("lambda", cfg["objective"]["lambda_grid"])
    return spec


def candidates(spec: dict, keys: list) -> list:
    """Row indices of the configurations that belong to a model, ordered weakest -> strongest."""
    out = []
    for i, k in enumerate(keys):
        if k[0] == "ridge":
            ok = k[1] in spec.get("ridge_kernels", []) and (spec.get("lambda") is None or k[2] in spec["lambda"])
        elif k[0] == "res":
            ok = (k[1] in spec.get("res_ratios", []) and k[3] in spec.get("res_tau_fractions", [])
                  and (spec.get("lambda") is None or k[2] in spec["lambda"]))
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
    if spec["select"] == "hybrid":
        gid = sorted({group_of(keys[i]) for i in order})
        groups = [[i for i in order if group_of(keys[i]) == g] for g in gid]
        return M.select_hybrid(loss, groups)
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
                item["fam_loss"], item["families"] = z["fam_loss"].astype(float), [str(f) for f in z["families"]]
                item["fam_fold"] = z["family_inner_fold"]
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
                assert cur["families"] == item["families"] and (cur["fam_fold"] == item["fam_fold"]).all()
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


def choose_from(spec: dict, keys: list, loss: np.ndarray) -> np.ndarray:
    return choose(spec, keys, loss)


def dev_xsel_per_drug(inner: dict, spec: dict) -> np.ndarray:
    """[training set, drug] sqrt of the selection-honest inner loss (see dev_xsel)."""
    out = []
    for sid, it in inner.items():
        FL, ff = it["fam_loss"], it["fam_fold"]
        D = FL.shape[2]
        sel = np.empty((FL.shape[1], D))
        for j in np.unique(ff):
            idx = choose(spec, it["keys"], FL[:, ff != j].mean(axis=1))
            sel[ff == j] = FL[idx][:, ff == j, :][np.arange(D), :, np.arange(D)].T
        out.append(np.sqrt(sel.mean(axis=0)))
    return np.array(out)


def dev_xsel(inner: dict, spec: dict, all_families: list | None = None, W: np.ndarray | None = None):
    """Selection-honest development score. For each inner fold j the configuration is chosen from the
    inner losses of the *other* inner folds (families disjoint from fold j) and scored on fold j; the
    per-drug squared errors of all families are then pooled as in the inner loss. This removes the
    optimism of reporting the minimum of the same numbers that were used to choose, so models with
    different numbers of candidates (per-drug vs global selection) can be compared.
    Returns the score, or bootstrap replicates if W is given (choices held fixed)."""
    pos = {f: i for i, f in enumerate(all_families)} if W is not None else None
    total = 0.0 if W is None else np.zeros(W.shape[0])
    for sid, it in inner.items():
        FL, ff = it["fam_loss"], it["fam_fold"]                    # [config, family, drug]
        D = FL.shape[2]
        sel = np.empty((FL.shape[1], D))                           # family x drug errors under held-out-fold selection
        for j in np.unique(ff):
            idx = choose(spec, it["keys"], FL[:, ff != j].mean(axis=1))
            sel[ff == j] = FL[idx][:, ff == j, :][np.arange(D), :, np.arange(D)].T
        if W is None:
            total += float(np.sqrt(sel.mean(axis=0)).mean())
        else:
            w = W[:, [pos[f] for f in it["families"]]]
            total += np.sqrt((w @ sel) / w.sum(axis=1, keepdims=True)).mean(axis=1)
    return total / len(inner)
