"""Score every method on a setting's evaluation wells with one protocol.

For each evaluation well: observed = its first n real cells (barcode-hash order), generated = n
cells, n = min(observed, control, 64). Variants written per method:

    native     the method's own output (model outputs clamped at zero; fraction negative recorded)
    pre        centred on the shared predicted mean mu_hat, no restoration
    anchored   centred on mu_hat, non-negative restoration v1 (plan 5.6)
    anchored2  centred on mu_hat, non-negative restoration v2 (cycle 1: control zero fraction)

`mu_hat = dmso_mean[plate, line] + m_psi(line, drug)`; no observed treated statistic of the
evaluation well is used to generate, anchor or restore.

Methods: control, meanpoint, shifted, shifted_pool, gauss_diag, gauss_lr<k>, ddpm:<run_id>, cells:<npz path>
(the last reads externally generated cells, e.g. CellFlow, keyed by well gid).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D
from .baselines import GaussianBank, TreatedPool, control_cells
from .ddpm import Context, load_model, sample
from .metrics import restore_nonneg, restore_zero_masked, shift_to_mean
from .setting import build, run_dir


def mu_hat_table(st, kind="delta_hat"):
    """kind: delta_hat (m_psi), delta_linemean (per-line mean response) or zero (no change)."""
    dh = np.load(run_dir("mean", st.name) / "delta_hat.npy")
    dh = np.zeros_like(dh) if kind == "zero" else np.load(run_dir("mean", st.name) / f"{kind}.npy")
    return {int(r.gid): st.d.dmso_mean[r.plate_index, r.line] + dh[r.line, r.drug] for r in st.eval_wells.itertuples()}


def score_all(st, name, pops, mu, variants, tseed, sseed, dev, save=None):
    S = st.scorer
    rows, keep, pi0 = [], {}, {}
    for r, P in zip(st.eval_wells.itertuples(), pops):
        key = (r.plate_index, r.line)
        if key not in pi0:
            pi0[key] = S.t((st.d.ctrl_cells(*key) <= 0).mean(0))
        n = int(r.n_eval)
        T = S.t(st.d.cells(r, n))
        m = S.t(mu[int(r.gid)])
        base = dict(method=name, train_seed=tseed, sample_seed=sseed, gid=r.gid, line=r.line, drug=r.drug, n=n)
        for v in variants:
            if v == "native":
                neg = float((P < 0).float().mean())
                X = P.clamp(min=0)
                rec = S.score(X, T, line=r.line); rec["neg_frac"] = neg
            elif v == "pre":
                X = shift_to_mean(P, m)
                rec = S.score(X, T, line=r.line)
            else:
                Xs = shift_to_mean(P, m)
                X = restore_nonneg(Xs, m) if v == "anchored" else restore_zero_masked(Xs, m, pi0[key])
                rec = S.score(X, T, line=r.line)
                rec["restore_mean_err"] = float((X.mean(0) - m.clamp(min=0)).abs().max())
            rows.append({**base, "variant": v, **rec})
            if save is not None and v == save:
                keep[str(r.gid)] = X.cpu().numpy().astype(np.float16)
    return rows, keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="dev")
    ap.add_argument("--methods", required=True, help="comma-separated")
    ap.add_argument("--sample-seeds", default="0,1,2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-cells", action="store_true", help="store generated cells of sample seed 0 (server only)")
    ap.add_argument("--mean-kind", default="delta_hat")
    ap.add_argument("--save-variant", default="anchored2")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = run_dir("eval", a.setting)
    dst = out / f"{a.tag}.tsv.gz"
    if dst.exists():
        raise SystemExit(f"{dst} exists; evaluation tags are immutable")
    t0 = time.time()
    st = build(a.setting, dev)
    W = st.eval_wells
    mu = mu_hat_table(st, a.mean_kind)
    n_each = W.n_eval.to_numpy()
    seeds = [int(s) for s in a.sample_seeds.split(",")]
    ctx = Context(st, dev)
    bank = None
    rows, timing = [], {}
    print(f"setting={a.setting} eval_wells={len(W)} conds={W.groupby(['line','drug']).ngroups} drugs={W.drug.nunique()} "
          f"lines={W.line.nunique()} dropped_small={W.attrs.get('n_dropped_small')} setup={time.time()-t0:.0f}s", flush=True)
    mus = {a.mean_kind: mu}
    for meth_full in a.methods.split(","):
        meth, _, mk = meth_full.partition("@")
        mk = mk or a.mean_kind
        if mk not in mus:
            mus[mk] = mu_hat_table(st, mk)
        t1 = time.time()
        tseed = -1
        for ss in seeds:
            if meth in ("control", "shifted"):
                pops = [control_cells(st, r, int(r.n_eval), ss, dev) for r in W.itertuples()]
                variants = ["native"] if meth == "control" else ["pre", "anchored", "anchored2"]
            elif meth == "shifted_pool":
                tp = TreatedPool(st)
                pops = [tp.sample(r, int(r.n_eval), ss, dev) for r in W.itertuples()]
                variants = ["pre", "anchored", "anchored2"]
            elif meth == "meanpoint":
                if ss != seeds[0]:
                    continue
                pops = [torch.zeros(int(r.n_eval), st.d.X.shape[1], device=dev) for r in W.itertuples()]
                variants = ["pre"]
            elif meth.startswith("gauss"):
                if bank is None:
                    bank = GaussianBank(st, 100, dev)
                kind, k = ("diag", 0) if meth == "gauss_diag" else ("lowrank", int(meth.replace("gauss_lr", "")))
                pops = [bank.sample(int(r.line), int(r.n_eval), ss, r.gid, kind, k) for r in W.itertuples()]
                variants = ["pre", "anchored", "anchored2"]
            elif meth.startswith("ddpm:"):
                rid = meth.split(":", 1)[1]
                model, cfg = load_model(run_dir("ddpm", rid) / "model.pt", dev)
                tseed = cfg["seed"]
                torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
                pops = sample(model, cfg, ctx, W, n_each, 7919 * (ss + 1) + tseed, dev)
                variants = ["native", "anchored", "anchored2"] if cfg["target"] == "abs" else ["pre", "anchored", "anchored2"]
            elif meth.startswith("cells:"):
                z = np.load(meth.split(":", 1)[1].format(seed=ss))
                pops = [torch.as_tensor(z[str(g)][:n].astype(np.float32), device=dev) for g, n in zip(W.gid, n_each)]
                assert all(len(p) == n for p, n in zip(pops, n_each))
                variants = ["native", "anchored", "anchored2"]
            else:
                raise ValueError(meth)
            save = a.save_variant if (a.save_cells and ss == seeds[0] and a.save_variant in variants) else None
            r, keep = score_all(st, meth_full, pops, mus[mk], variants, tseed, ss, dev, save)
            rows += r
            if keep:
                gp = Path(os.environ["R3_ASSETS"]) / "gen" / a.setting
                gp.mkdir(parents=True, exist_ok=True)
                np.savez(gp / f"{a.tag}__{meth_full.replace(':', '_').replace('/', '_')}__ss{ss}.npz", **keep)
        timing[meth_full] = time.time() - t1
        print(meth_full, f"{timing[meth_full]:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(dst, sep="\t", index=False, float_format="%.6g")
    D.dump_json(dict(tag=a.tag, setting=a.setting, methods=a.methods.split(","), sample_seeds=seeds,
                     n_wells=int(len(W)), timing_s=timing, mean_kind=a.mean_kind,
                     peak_vram_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else None,
                     sha256=D.sha256_file(dst), commit=os.popen("git rev-parse HEAD").read().strip()),
                out / f"{a.tag}.json")
    s = R.groupby(["method", "variant"])[["ed", "ed_shape", "ed_pca50", "mean_rmse", "var_ratio", "zero_frac_mae"]].mean()
    print(s.to_string(), flush=True)


if __name__ == "__main__":
    main()
