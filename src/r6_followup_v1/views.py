"""The fixed training-view cache shared by every arm of the minimal comparison (plan 6.1).

For a training seed s, every control pool gets `reps` views per keep-rate p in {1, 0.5, 0.25}:
    view = 64 distinct fit_input cells of the pool (chosen by the view's own generator)
           + exact binomial thinning of their integer counts at p (p = 1: identity)
    generator = numpy PCG64([20261118, s, pool_id, round(1000 p), rep])
The cache depends on control cells only (no response, no drug, no fold), so the same file serves
every source fold and the final setting. S, SD, the mean-RNA MLP and the Ridge all read this one
file: the same barcodes and the same thinning draws, a finite set of views for every arm.
Evaluation input cells and reference cells never enter it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.depth_robust_r6 import data as D
from src.depth_robust_r6.count_view import lognorm, thin_np

from . import common as C

REPS = 8
BASE = 20261118


def build_views(d: D.R6Data, seed: int, reps: int = REPS) -> dict:
    n_pool, G = len(d.pools), d.cnt.shape[1]
    assert (d.pools.pool_id.to_numpy() == np.arange(n_pool)).all()
    V = n_pool * len(D.PS) * reps
    cells = np.empty((V, D.EVAL_N), np.int32); cnt = np.empty((V, D.EVAL_N, G), np.int16)
    crest = np.empty((V, D.EVAL_N), np.int32); pool = np.empty(V, np.int32); pv = np.empty(V, np.float32)
    v = 0
    for r in d.pools.itertuples():
        for p in D.PS:
            for j in range(reps):
                rng = np.random.Generator(np.random.PCG64([BASE, int(seed), int(r.pool_id), int(round(p * 1000)), j]))
                idx = int(r.fit_start) + np.sort(rng.choice(int(r.n_fit_input), D.EVAL_N, replace=False))
                c, cr = thin_np(d.cnt[idx], d.crest[idx], p, rng)
                cells[v], cnt[v], crest[v], pool[v], pv[v] = idx, c, cr, r.pool_id, p
                v += 1
    return dict(cells=cells, cnt=cnt, crest=crest, pool=pool, p=pv, reps=np.int32(reps))


def view_means(v: dict) -> np.ndarray:
    """Mean log-normalised profile of every view [V, G] (the mean-RNA input of M and of the Ridge)."""
    out = np.empty((len(v["pool"]), v["cnt"].shape[2]))
    for i in range(0, len(out), 256):
        c = v["cnt"][i:i + 256].astype(np.float64)
        out[i:i + 256] = lognorm(c, c.sum(-1) + v["crest"][i:i + 256]).mean(1)
    return out


def path(seed: int) -> Path:
    return C.assets() / "cache" / "views_v1" / f"views_s{seed}.npz"


def load_views(seed: int) -> dict:
    z = np.load(path(seed))
    return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    a = ap.parse_args()
    d = C.load(eval_views=False)
    for s in a.seeds:
        f = path(s)
        f.parent.mkdir(parents=True, exist_ok=True)
        assert not f.exists(), f"{f} exists; view caches are written once"
        v = build_views(d, s)
        np.savez(f, **v)
        full = d.cnt[v["cells"].ravel()].reshape(v["cnt"].shape)
        rec = dict(seed=s, reps_per_pool_and_p=REPS, keep_rates=list(D.PS), n_views=int(len(v["pool"])), n_pools=int(len(d.pools)),
                   n_cells_per_view=D.EVAL_N, generator=f"numpy PCG64([{BASE}, seed, pool_id, round(1000*p), rep])",
                   cells_role="fit_input only", sha256_cells=D.sha256_array(v["cells"]), sha256_counts=D.sha256_array(v["cnt"]),
                   sha256_rest=D.sha256_array(v["crest"]), file=str(f), bytes=f.stat().st_size, sha256_file=D.sha256_file(f),
                   kept_fraction_panel={str(p): float(v["cnt"][v["p"] == np.float32(p)].sum() / full[v["p"] == np.float32(p)].sum()) for p in D.PS},
                   distinct_fit_cells_used=int(len(np.unique(v["cells"]))), commit=C.git_commit())
        C.dump_json(rec, f"manifests/r6_followup_v1/view_cache_s{s}.json")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
