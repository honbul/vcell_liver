#!/usr/bin/env python
"""G4 task 1: State ST-HVG-Replogle *zeroshot* inference on the held-out line, controls only.

Runs in assets/envs/state (arc-state 0.11.1). Per held-out line L (one official zeroshot checkpoint
per line; its TOML puts L's perturbed cells only in `test`, L's controls in train):
 1. Input = L's non-targeting control cells from the raw h5ad (same files as F3), each cell
    normalised to 1e4 by its row sum over the author panel, log1p, restricted to the model's 2,000
    HVGs (var_dims gene_names, matched by symbol). This is checked against the authors' own
    eval adata_real (control rows): exact row-hash match rate for row-sum vs UMI_count scaling.
 2. Targets = F3 condition-A targets of L (perturbed in >=1 F3 source line with >=20 cells) that are
    in the checkpoint's perturbation one-hot map. No treated cell of L is read.
 3. Basal pool = a fixed random set of n_pool L controls (seeded); every target is predicted from
    the same pool (n_pool/64 sets of 64), each basal cell carrying its own gem_group batch token
    (shared gem_group ids across lines, as in the checkpoint's batch map). Forward = the model's
    predict_step (padded sets), outputs clipped to [0, 14] as in `state tx infer`.
    Predicted delta_t = mean(pred cells) - mean(pool basal), on the log1p(CP10k) HVG scale.
 4. Checkpoints: final.ckpt (step 100k, primary: no selection) and best.ckpt (min val_loss; for
    these TOMLs val falls back to L's test cells -> target-selected, reference only). A second pool
    seed with final.ckpt gives the basal-sampling noise.
 5. Reproduction check: authors' eval_best.ckpt adata_pred/adata_real -> per-target delta
    (mean pred of the target's rows - mean real control rows) vs our best.ckpt delta.
Outputs: assets/g4/state/pred/<L>.npz, reports/feedback/G4/state_infer_<L>.json.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_pseudobulk import LINES as RAW, _dec, read_cat  # noqa: E402

FOLD = {"hepg2": "HepG2", "jurkat": "Jurkat", "k562": "K562", "rpe1": "RPE1"}


def load_controls(path, hvg, block=8000):
    with h5py.File(path, "r") as f:
        gn_codes = f["var/gene_name"][:]
        var_names = _dec(f["var/__categories/gene_name"][:])[gn_codes]
        g_codes, g_cats = read_cat(f, "gene")
        tsym = g_cats[g_codes]
        gem = f["obs/gem_group"][:].astype(np.int64)
        umi = f["obs/UMI_count"][:].astype(np.float64)
        ctrl = np.where(tsym == "non-targeting")[0]
        pos = {}
        for j, g in enumerate(var_names):
            pos.setdefault(g, j)
        col = np.array([pos.get(g, -1) for g in hvg])
        X = f["X"]
        rows_rs, rows_umi = [], []
        for s in range(0, X.shape[0], block):
            sel = ctrl[(ctrl >= s) & (ctrl < s + block)]
            if len(sel) == 0:
                continue
            B = X[s:min(s + block, X.shape[0])][sel - s].astype(np.float64)
            rs = B.sum(1, keepdims=True)
            H = np.zeros((len(sel), len(hvg)))
            ok = col >= 0
            H[:, ok] = B[:, col[ok]]
            rows_rs.append(np.log1p(H / rs * 1e4).astype(np.float32))
            rows_umi.append(np.log1p(H / umi[sel][:, None] * 1e4).astype(np.float32))
    return dict(X_rs=np.vstack(rows_rs), X_umi=np.vstack(rows_umi), gem=gem[ctrl], n_ctrl=len(ctrl),
                n_hvg_missing=int((col < 0).sum()), idx=ctrl)


def row_hash(X, dec=3):
    return {np.round(r, dec).tobytes() for r in X}


def authors_eval(adir, n_sample=None):
    """Per-target delta from the authors' zeroshot eval files (pred rows of target - real control rows)."""
    out = {}
    with h5py.File(adir / "adata_real.h5ad", "r") as fr, h5py.File(adir / "adata_pred.h5ad", "r") as fp:
        g = fr["obs"]["gene"]
        gene = g["categories"][:].astype(str)[g["codes"][:]]
        gp = fp["obs"]["gene"]
        gene_p = gp["categories"][:].astype(str)[gp["codes"][:]]
        assert (gene == gene_p).all()
        Xr, Xp = fr["X"], fp["X"]
        ctrl = np.where(gene == "non-targeting")[0]
        real_ctrl = Xr[np.sort(ctrl)]
        out["real_ctrl"] = real_ctrl
        cm_real = real_ctrl.mean(0)
        cm_pred = Xp[np.sort(ctrl)].mean(0)
        order = np.argsort(gene, kind="stable")
        gs = gene[order]
        bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
        Xp_all = Xp[:]  # ~1-1.5 GB float32
        Xr_all = Xr[:]
        dp, dr, names, n = [], [], [], []
        for a, b in zip(bounds[:-1], bounds[1:]):
            t = gs[a]
            if t == "non-targeting":
                continue
            idx = np.sort(order[a:b])
            dp.append(Xp_all[idx].mean(0) - cm_real)
            dr.append(Xr_all[idx].mean(0) - cm_real)
            names.append(t)
            n.append(b - a)
        out.update(targets=np.array(names), delta_pred=np.array(dp), delta_real=np.array(dr), n=np.array(n),
                   ctrl_pred_shift=cm_pred - cm_real)
    return out


def predict(model, pool_X, pool_b, pert_vecs, set_len, device, chunk_sets=256):
    """pert_vecs: [T, pert_dim]. Returns mean predicted expression per target [T, G]."""
    n_pool = len(pool_X)
    assert n_pool % set_len == 0
    sets_per_t = n_pool // set_len
    tpc = max(1, chunk_sets // sets_per_t)  # targets per forward chunk
    X = torch.tensor(pool_X, dtype=torch.float32, device=device)
    bidx = torch.tensor(pool_b, dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for s in range(0, len(pert_vecs), tpc):
            pv = torch.tensor(pert_vecs[s:s + tpc], dtype=torch.float32, device=device)
            k = len(pv)
            batch = {"ctrl_cell_emb": X.repeat(k, 1),
                     "pert_emb": pv.repeat_interleave(n_pool, 0),
                     "pert_name": ["x"] * (k * n_pool),
                     "batch": bidx.repeat(k)}
            o = model.predict_step(batch, batch_idx=0, padded=True)
            P = o["preds"].float().clamp_(0.0, 14.0).reshape(k, n_pool, -1).mean(1)
            out.append(P.cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=sorted(FOLD))
    ap.add_argument("--hf", default="assets/g4/state/hf/ST-HVG-Replogle/zeroshot")
    ap.add_argument("--f3", default="assets/f3")
    ap.add_argument("--out", default="assets/g4/state/pred")
    ap.add_argument("--report", default="reports/feedback/G4")
    ap.add_argument("--n-pool", type=int, default=512)
    ap.add_argument("--pool-seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    t0 = time.time()
    L = FOLD[a.fold]
    mdir = Path(a.hf) / a.fold
    vd = pickle.load(open(mdir / "var_dims.pkl", "rb"))
    hvg = [str(g) for g in vd["gene_names"]]
    pmap = torch.load(mdir / "pert_onehot_map.pt", weights_only=False)
    pmap = {str(k): v for k, v in pmap.items()}
    bmap = pickle.load(open(mdir / "batch_onehot_map.pkl", "rb"))
    bidx_of = {str(k): int(np.argmax(np.asarray(v))) for k, v in bmap.items()}
    meta = dict(fold=a.fold, line=L, n_hvg=len(hvg), pert_dim=int(vd["pert_dim"]), batch_dim=int(vd["batch_dim"]),
                device=a.device, n_pool=a.n_pool, pool_seeds=a.pool_seeds, torch=torch.__version__)

    # 1) controls of L only
    C = load_controls(RAW[L], hvg)
    meta.update(n_ctrl=C["n_ctrl"], n_hvg_missing_in_raw=C["n_hvg_missing"])
    print(f"[g4s] {L}: {C['n_ctrl']} controls, hvg missing {C['n_hvg_missing']} ({time.time()-t0:.0f}s)", flush=True)
    # verify input space against the authors' adata_real control rows
    adir = mdir / "eval_best.ckpt"
    A = authors_eval(adir) if (adir / "adata_pred.h5ad").is_file() else None
    if A is not None:
        hr = row_hash(A["real_ctrl"])
        for k in ("X_rs", "X_umi"):
            hk = row_hash(C[k])
            meta[f"ctrl_rowhash_match_{k}"] = dict(authors_rows=len(hr), ours=len(hk), matched=len(hr & hk))
        m_a, m_o = A["real_ctrl"].mean(0), C["X_rs"].mean(0)
        meta["ctrl_mean_corr_rs"] = float(np.corrcoef(m_a, m_o)[0, 1])
        meta["ctrl_mean_maxabs_rs"] = float(np.abs(m_a - m_o).max())
        print("[g4s] input check", {k: v for k, v in meta.items() if k.startswith("ctrl_")}, flush=True)
    Xc = C["X_rs"]
    b_all = np.array([bidx_of.get(str(g), -1) for g in C["gem"]])
    meta["gem_missing_in_batch_map"] = int((b_all < 0).sum())
    b_all = np.where(b_all < 0, 0, b_all)

    # 2) targets: F3 condition A of L in the pert map
    fa = sorted((Path(a.f3) / "loco").glob(f"pred_{L}_A_*.npz"))[0]
    tA = np.load(fa, allow_pickle=True)["targets"].astype(str)
    targets = np.array([t for t in tA if t in pmap])
    meta.update(n_f3_A_targets=len(tA), n_targets=len(targets),
                f3_A_not_in_pert_map=[t for t in tA if t not in pmap][:50])
    pv = np.stack([np.asarray(pmap[t], dtype=np.float32).reshape(-1) for t in targets])
    nt_vec = np.asarray(pmap["non-targeting"], dtype=np.float32).reshape(1, -1) if "non-targeting" in pmap else None

    # 3) inference
    res = dict(targets=targets, genes=np.array(hvg))
    for ck in ("final", "best"):
        model = StateTransitionPerturbationModel.load_from_checkpoint(str(mdir / "checkpoints" / f"{ck}.ckpt"),
                                                                      map_location=a.device)
        model.eval().to(a.device)
        set_len = int(getattr(model, "cell_sentence_len", 64))
        meta["cell_set_len"] = set_len
        seeds = a.pool_seeds if ck == "final" else a.pool_seeds[:1]
        for ps in seeds:
            rng = np.random.default_rng(ps)
            pool = np.sort(rng.choice(len(Xc), a.n_pool, replace=False))
            pX, pb = Xc[pool], b_all[pool]
            t1 = time.time()
            P = predict(model, pX, pb, pv, set_len, a.device)
            res[f"delta_{ck}_p{ps}"] = (P - pX.mean(0)).astype(np.float32)
            if nt_vec is not None:
                res[f"ntshift_{ck}_p{ps}"] = (predict(model, pX, pb, nt_vec, set_len, a.device)[0] - pX.mean(0)).astype(np.float32)
            meta[f"seconds_{ck}_p{ps}"] = round(time.time() - t1, 1)
            print(f"[g4s] {ck} pool{ps}: {len(targets)} targets in {time.time()-t1:.0f}s", flush=True)
        del model
    k0, k1 = f"delta_final_p{a.pool_seeds[0]}", f"delta_final_p{a.pool_seeds[-1]}"
    if k0 != k1:
        r = [np.corrcoef(x, y)[0, 1] for x, y in zip(res[k0], res[k1])]
        meta["pool_seed_delta_pearson_median"] = float(np.median(r))
    # 4) reproduction vs the authors' best.ckpt eval predictions
    if A is not None:
        ai = {t: i for i, t in enumerate(A["targets"])}
        both = [i for i, t in enumerate(targets) if t in ai]
        ours = res[f"delta_best_p{a.pool_seeds[0]}"][both]
        theirs = A["delta_pred"][[ai[targets[i]] for i in both]]
        rr = np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(ours, theirs)])
        meta["repro_vs_authors"] = dict(n_targets=len(both), pearson_median=float(np.nanmedian(rr)),
                                        pearson_q10=float(np.nanquantile(rr, 0.1)),
                                        norm_ratio_median=float(np.median(np.linalg.norm(ours, axis=1) /
                                                                          np.maximum(np.linalg.norm(theirs, axis=1), 1e-9))))
        res["authors_targets"] = A["targets"]
        res["authors_delta_pred"] = A["delta_pred"].astype(np.float32)
        res["authors_delta_real"] = A["delta_real"].astype(np.float32)
        res["authors_n"] = A["n"]
        print("[g4s] repro", meta["repro_vs_authors"], flush=True)
    res["ctrl_mean_all"] = Xc.mean(0)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / f"{L}.npz", **res)
    meta["seconds_total"] = round(time.time() - t0, 1)
    Path(a.report).mkdir(parents=True, exist_ok=True)
    (Path(a.report) / f"state_infer_{a.fold}.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
