#!/usr/bin/env python3
"""Inner-CV loss tables (and, only with --outer-predict, held-out predictions) for a configuration grid.

One unit of work is (training set, prior). Training sets:
    r{rep}f{fold}   outer-training families of that Track A outer fold (internal)
    x_{family}      all LIMORE families except that one (LCCL cross-study leave-family-out)
    x_ALL           all LIMORE families (LCCL new lineages)
Priors: real | permNNN (whole-axis gene-label permutation of the Hallmark sets).

Development runs never pass --outer-predict: they only produce inner-fold losses, which are computed
from training families alone. Held-out predictions are written without labels; scoring is a separate
step (evaluate.py).

Usage: run_grid.py --run-dir DIR --grid NAME --sets internal|external --priors real[,perm101,...]
                   [--task-index I --n-tasks N] [--outer-predict]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import r1_models as M  # noqa: E402


def grid_configs(cfg: dict, name: str) -> list:
    g = cfg["grids"][name]
    out = []
    if g.get("ridge_kernels"):
        out += M.ridge_configs(g["lambda"], g["ridge_kernels"])
    if g.get("tau_fractions"):
        out += M.nuc_configs(g["lambda"], g["nuc_ratios"], g["tau_fractions"])
    assert len(set(out)) == len(out)
    return out


def training_sets(cfg: dict, data: R.Data, which: str, ext: R.External | None) -> list:
    """(set id, train row mask, rows to predict (features), predicted model ids, inner seed)"""
    out = []
    if which == "internal":
        for rep in sorted(data.splits.repeat.unique()):
            fold = data.outer_fold(rep)
            for k in sorted(set(fold.tolist())):
                tr, te = fold != k, fold == k
                assert not set(data.fam[tr]) & set(data.fam[te])
                out.append((f"r{rep}f{k}", tr, data.rank[te], [data.models[i] for i in np.where(te)[0]],
                            R.inner_seed(cfg, rep, k)))
    else:
        seed = cfg["split"]["external_inner_seed"]
        new = ext.eval_type == "new_lineage"
        assert not set(ext.fam[new]) & set(data.fam)
        out.append(("x_ALL", np.ones(len(data.models), bool), ext.rank[new],
                    [m for m, n in zip(ext.models, new) if n], seed))
        for f in sorted(set(ext.fam[~new])):
            tr = data.fam != f
            assert (~tr).sum() >= 1
            rows = ext.fam == f
            out.append((f"x_{f}", tr, ext.rank[rows], [m for m, r in zip(ext.models, rows) if r], seed))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--sets", required=True, choices=["internal", "external"])
    ap.add_argument("--priors", default="real")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--outer-predict", action="store_true")
    args = ap.parse_args()
    cfg = R.load_config()
    data = R.load_data(cfg)
    ext = R.load_external(cfg, data) if args.sets == "external" else None
    configs = grid_configs(cfg, args.grid)
    keys = [M.key_str(c) for c in configs]
    priors = args.priors.split(",")
    units = [(s, p) for s in training_sets(cfg, data, args.sets, ext) for p in priors]
    units = [u for i, u in enumerate(units) if i % args.n_tasks == args.task_index]
    os.makedirs(os.path.join(args.run_dir, "inner"), exist_ok=True)
    if args.outer_predict:
        os.makedirs(os.path.join(args.run_dir, "pred"), exist_ok=True)
    n_inner = cfg["split"]["inner_folds"]
    for (sid, tr, X_te, te_models, seed), prior in units:
        out = os.path.join(args.run_dir, "inner", f"{sid}__{prior}.npz")
        if os.path.exists(out):
            raise FileExistsError(f"refusing to overwrite {out}")
        t0 = time.time()
        sets = data.prior_sets(prior)
        X, Y, fam = data.rank[tr], data.Y[tr], data.fam[tr]
        loss, records = M.inner_losses(X, Y, fam, sets, configs, n_inner, seed)
        meta = dict(set=sid, prior=prior, grid=args.grid, n_train_models=int(tr.sum()),
                    n_train_families=int(len(set(fam))), inner_seed=int(seed),
                    n_not_converged=int(sum(not r["converged"] for r in records)),
                    max_iterations=int(max([r["iterations"] for r in records], default=0)))
        extra = {}
        if args.outer_predict:
            need_reduced = any(c[0] == "nuc" for c in configs)
            prep = M.Prep(X, Y, fam, X_te, sets, need_reduced=need_reduced)
            pred = np.empty((len(configs), len(te_models), Y.shape[1]))
            for c, key in enumerate(configs):
                Pz, info = prep.predict_z(key)
                pred[c] = prep.to_raw(Pz)
                if info is not None:
                    records.append(dict(inner_fold=-1, config=keys[c], converged=info["converged"],
                                        iterations=info["iterations"], rank=info["rank"]))
            meta.update(pathway_rank=int(prep.r), par_fraction=prep.par_fraction,
                        tau_max=float(getattr(prep, "tau_max", np.nan)),
                        n_not_converged=int(sum(not r["converged"] for r in records)))
            np.savez(os.path.join(args.run_dir, "pred", f"{sid}__{prior}.tmp.npz"), configs=np.array(keys), pred=pred,
                     models=np.array(te_models), y_mu=prep.y_mu, y_sd=prep.y_sd, constant=prep.constant)
            os.replace(os.path.join(args.run_dir, "pred", f"{sid}__{prior}.tmp.npz"),
                       os.path.join(args.run_dir, "pred", f"{sid}__{prior}.npz"))
        meta["seconds"] = round(time.time() - t0, 1)
        np.savez(out[:-4] + ".tmp.npz", configs=np.array(keys), loss=loss, meta=json.dumps(meta),
                 records=json.dumps(records), **extra)
        os.replace(out[:-4] + ".tmp.npz", out)
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
