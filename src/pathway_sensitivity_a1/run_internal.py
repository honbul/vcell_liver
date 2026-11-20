#!/usr/bin/env python3
"""Nested family-group evaluation of one representation on LIMORE (stages A2/A3).

Writes one out-of-fold prediction table per repeat into an immutable run directory. Tables hold
predictions and training statistics only; observed responses are joined later by ID.

Usage: run_internal.py --run-dir DIR --kind A3 [--perm-seed 101] [--fraction 0.5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
import nested  # noqa: E402


def features(cfg: dict, study: core.Study, kind: str, perm_seed: int | None) -> np.ndarray:
    if kind == "A0":
        return np.zeros((len(study.models), 1))
    if kind == "A1":
        return study.expr
    if kind in ("A1r", "A2"):
        return study.rank
    if kind == "A3":
        return study.hallmark
    if kind == "A4":
        return core.load_perm_scores(cfg, study.name, perm_seed, study.models, study.sets)
    raise ValueError(kind)


def tag(kind: str, perm_seed, fraction: float) -> str:
    t = kind
    if kind == "A4":
        t += f"_perm{perm_seed:03d}"
    if fraction < 1.0:
        t += f"_frac{int(round(fraction * 100)):03d}"
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--kind", required=True, choices=["A0", "A1", "A1r", "A2", "A3", "A4"])
    ap.add_argument("--perm-seed", type=int)
    ap.add_argument("--fraction", type=float, default=1.0)
    args = ap.parse_args()
    cfg = core.load_config()
    assert (args.kind == "A4") == (args.perm_seed is not None)
    if args.kind == "A4":
        assert args.perm_seed in cfg["a4"]["seeds"]
    assert args.fraction in cfg["learning_curve"]["fractions"]

    study = core.load_study(cfg, "LIMORE")
    reg = core.load_registry().set_index("model_id")
    splits = core.load_outer_splits()
    long = core.load_limore_response(cfg)
    drugs = sorted(long.drug_id.unique())
    row_of = {m: i for i, m in enumerate(study.models)}
    X_all = features(cfg, study, args.kind, args.perm_seed)
    K = len(study.sets)
    name = tag(args.kind, args.perm_seed, args.fraction)
    os.makedirs(os.path.join(args.run_dir, "oof"), exist_ok=True)
    meta = dict(kind=args.kind, perm_seed=args.perm_seed, fraction=args.fraction, K=K,
                cache_key=core.cache_key(cfg), n_features=int(X_all.shape[1]), folds=[])
    t0 = time.time()
    for rep, sp in splits.groupby("repeat"):
        out = os.path.join(args.run_dir, "oof", f"{name}__rep{rep}.tsv.gz")
        if os.path.exists(out):
            raise FileExistsError(f"refusing to overwrite {out}")
        models = list(sp.model_id)
        assert all(reg.loc[m, "included"] and reg.loc[m, "family_id"] == f
                   for m, f in zip(models, sp.family_id)), "split table disagrees with registry"
        idx = np.array([row_of[m] for m in models])
        assert [study.models[i] for i in idx] == models
        X = X_all[idx]
        Y = core.response_matrix(long, models, drugs, "y")
        fam = sp.family_id.values
        fold = sp.outer_fold.values
        frames = []

        def subset(k, tr_idx, rep=rep, fam=fam):
            if args.fraction >= 1.0:
                return tr_idx
            fams = np.array(sorted(set(fam[tr_idx])))
            rs = np.random.RandomState(cfg["learning_curve"]["seed_base"] + 100 * int(rep) + int(k))
            keep = set(rs.choice(fams, size=int(np.ceil(args.fraction * len(fams))), replace=False))
            return np.array([i for i in tr_idx if fam[i] in keep])

        base = cfg["split"]["inner_seed_base"] + 100 * int(rep)
        for k, tr_idx, te_idx, pred, info in nested.outer_folds(
                X, Y, fam, fold, args.kind, cfg["model"]["alpha_grid"], cfg["split"]["inner_folds"],
                lambda k, base=base: base + int(k), K, cfg["endpoint"]["constant_response_sd"], subset):
            te = fold == k
            assert list(np.where(te)[0]) == list(te_idx)
            te_models = [models[i] for i in np.where(te)[0]]
            df = pd.DataFrame(pred, index=te_models, columns=drugs).rename_axis("model_id").reset_index()
            df = df.melt(id_vars="model_id", var_name="drug_id", value_name="pred")
            per_drug = pd.DataFrame(dict(drug_id=drugs, train_mean=info["mu"], s_train=info["sd"],
                                         alpha=info["alpha"], constant=info["constant"]))
            df = df.merge(per_drug, on="drug_id", validate="many_to_one")
            df.insert(0, "outer_fold", int(k))
            df.insert(0, "repeat", int(rep))
            frames.append(df)
            dims = sorted({d for d in info["pca_dim"] if d is not None})
            meta["folds"].append(dict(repeat=int(rep), outer_fold=int(k), n_train_models=int(len(tr_idx)),
                                      n_train_families=int(len(set(fam[tr_idx]))), n_test_models=int(te.sum()),
                                      pca_dim=dims))
        res = pd.concat(frames, ignore_index=True)
        assert not res.duplicated(["model_id", "drug_id"]).any() and res.model_id.nunique() == len(models)
        res.to_csv(out + ".tmp.gz", sep="\t", index=False, float_format="%.8g", compression="gzip")
        os.replace(out + ".tmp.gz", out)
    meta["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.run_dir, "oof", f"{name}__meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"{name}: done in {meta['seconds']} s")


if __name__ == "__main__":
    main()
