"""Freeze the R6 data contract from the raw-count cache: control roles, truth, split, eval views.

Writes (server, git-ignored)  $R6_ASSETS/cache/contract_v1/
    pools.tsv, input_cells.tsv, input_counts.npz, ref_mean.npy, all_mean.npy, Y_r6.npy,
    valid_r6.npy, gene_stats.npz, eval_views.npz
and (Git) manifests/depth_robust_r6/
    control_input_reference_split.tsv   one row per pool: sizes and barcode-list hashes per role
    input_cells.tsv.gz                  barcode-level roles of the input cells
    thinning_manifest.json              keep-rates, seeds, generator, hashes of every evaluation view
    drug_split.json                     drug groups per role and per development fold
    contract_record.json                counts and hashes of everything above
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import data as D
from .count_view import lognorm, thin_np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True, help="output dir of build_count_cache")
    ap.add_argument("--v2-data", default=os.environ.get("R6_V2DATA"))
    ap.add_argument("--manifest-dir", default="manifests/chemcpa_moe_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--git-out", default="manifests/depth_robust_r6")
    a = ap.parse_args()
    out, gout, v2 = Path(a.out), Path(a.git_out), Path(a.v2_data)
    out.mkdir(parents=True, exist_ok=False)
    gout.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(Path(a.counts) / "cells.tsv", sep="\t")       # sorted by plate_index, line, u
    C = sp.load_npz(Path(a.counts) / "panel_counts.npz").tocsr()
    G = C.shape[1]
    assert C.data.min() >= 1 and C.data.max() <= np.iinfo(np.int16).max, C.data.max()
    lib = meta.library_size.to_numpy().astype(np.float64)
    Ln = C.astype(np.float64).tocsr()
    Ln.data = np.log1p(Ln.data * np.repeat(1e4 / lib, np.diff(Ln.indptr)))   # full-observation log1p(CP10k)

    Yv2 = np.load(v2 / "Y.npy"); valid = np.load(v2 / "valid.npy")
    plate = np.load(v2 / "plate.npy"); pw = np.load(v2 / "plate_weights.npy")
    nP, nL = pw.shape[2], Yv2.shape[0]

    rows, in_idx, cell_rows = [], [], []
    ref_mean = np.zeros((nP, nL, G)); all_mean = np.zeros((nP, nL, G)); have = np.zeros((nP, nL), bool)
    k = 0
    for (p, l), g in meta.groupby(["plate_index", "line"], sort=True):
        idx = g.index.to_numpy()
        assert np.all(np.diff(g.u.to_numpy()) >= 0)
        n = len(idx)
        all_mean[p, l] = np.asarray(Ln[idx].mean(axis=0)).ravel(); have[p, l] = True
        roles = D.pool_roles(n)
        bc = g.barcode.to_numpy()
        h = lambda x: hashlib.sha256("\n".join(x).encode()).hexdigest()
        if roles is None:
            ref_mean[p, l] = all_mean[p, l]
            rows.append(dict(pool_id=-1, plate_index=p, line=l, cell_line_id=g.cell_line_id.iloc[0], plate=g.plate.iloc[0],
                             n_total=n, n_eval_input=0, n_fit_input=0, n_reference=n, eligible=False,
                             eval_start=-1, eval_end=-1, fit_start=-1, fit_end=-1,
                             sha256_eval_barcodes="", sha256_fit_barcodes="", sha256_reference_barcodes=h(bc)))
            continue
        ne, nf, nr = roles
        ref_mean[p, l] = np.asarray(Ln[idx[ne + nf:]].mean(axis=0)).ravel()
        s = sum(len(x) for x in in_idx)
        in_idx.append(idx[: ne + nf])
        rows.append(dict(pool_id=k, plate_index=p, line=l, cell_line_id=g.cell_line_id.iloc[0], plate=g.plate.iloc[0],
                         n_total=n, n_eval_input=ne, n_fit_input=nf, n_reference=nr, eligible=True,
                         eval_start=s, eval_end=s + ne, fit_start=s + ne, fit_end=s + ne + nf,
                         sha256_eval_barcodes=h(bc[:ne]), sha256_fit_barcodes=h(bc[ne:ne + nf]),
                         sha256_reference_barcodes=h(bc[ne + nf:])))
        cell_rows.append(pd.DataFrame(dict(barcode=bc[: ne + nf], pool_id=k, plate=g.plate.iloc[0],
                                           cell_line_id=g.cell_line_id.iloc[0], sample=g["sample"].to_numpy()[: ne + nf],
                                           role=["eval_input"] * ne + ["fit_input"] * nf,
                                           library_size=g.library_size.to_numpy()[: ne + nf],
                                           panel_sum=g.panel_sum.to_numpy()[: ne + nf])))
        k += 1
    pools_all = pd.DataFrame(rows)
    pools = pools_all[pools_all.eligible].reset_index(drop=True)
    ii = np.concatenate(in_idx)
    cells = pd.concat(cell_rows, ignore_index=True)
    assert cells.barcode.is_unique                                  # no cell holds two roles
    cnt = np.asarray(C[ii].todense()).astype(np.int16)
    crest = meta.c_rest.to_numpy()[ii].astype(np.int32)
    assert (cnt.sum(1) + crest == meta.library_size.to_numpy()[ii]).all()

    # ---- truth with the barcode-disjoint reference ----------------------------------------------
    shift = np.einsum("ldp,plg->ldg", pw, all_mean - ref_mean)
    Y = (Yv2 + shift).astype(np.float32)
    elig = np.zeros((nP, nL), bool); elig[pools.plate_index, pools.line] = True
    ok_primary = (plate >= 0) & elig[np.clip(plate, 0, None), np.arange(nL)[:, None]]
    ok_all = np.einsum("ldp,pl->ld", (pw > 0).astype(float), (~have).astype(float)) == 0
    valid6 = valid & ok_primary & ok_all
    Y[~valid6] = 0.0

    # ---- gene statistics from fit_input control cells only --------------------------------------
    fit_rows = np.concatenate([np.arange(r.fit_start, r.fit_end) for r in pools.itertuples()])
    xf = lognorm(cnt[fit_rows].astype(np.float64), (cnt[fit_rows].sum(1) + crest[fit_rows]).astype(np.float64))
    x_mean = xf.mean(0); x_sd = np.maximum(xf.std(0), np.quantile(xf.std(0), 0.10))
    order = np.argsort(-x_mean, kind="stable")
    tb = np.empty(G); tb[order] = 0.49 * (1.0 - np.arange(G) / G)

    # ---- fixed evaluation views -----------------------------------------------------------------
    ev_rows = np.concatenate([np.arange(r.eval_start, r.eval_end) for r in pools.itertuples()])
    views, vman = {}, []
    for p in D.PS[1:]:
        for s in D.THIN_SEEDS:
            rng = np.random.Generator(np.random.PCG64([20261117, int(round(p * 1000)), s]))
            c2, r2 = thin_np(cnt[ev_rows], crest[ev_rows], p, rng)
            views[f"c_p{p}_s{s}"] = c2; views[f"r_p{p}_s{s}"] = r2
            vman.append(dict(p=p, seed=s, generator=f"numpy PCG64([20261117, {int(round(p*1000))}, {s}])",
                             sha256_panel=D.sha256_array(c2), sha256_rest=D.sha256_array(r2),
                             kept_fraction_panel=float(c2.sum() / cnt[ev_rows].sum()),
                             kept_fraction_library=float((c2.sum() + r2.sum()) / (cnt[ev_rows].sum() + crest[ev_rows].sum())),
                             median_library=float(np.median(c2.sum(1) + r2))))

    pools.to_csv(out / "pools.tsv", sep="\t", index=False)
    cells.to_csv(out / "input_cells.tsv", sep="\t", index=False)
    np.savez(out / "input_counts.npz", cnt=cnt, crest=crest)
    np.save(out / "ref_mean.npy", ref_mean.astype(np.float32)); np.save(out / "all_mean.npy", all_mean.astype(np.float32))
    np.save(out / "Y_r6.npy", Y); np.save(out / "valid_r6.npy", valid6)
    np.savez(out / "gene_stats.npz", x_mean=x_mean.astype(np.float32), x_sd=x_sd.astype(np.float32), tiebreak=tb.astype(np.float32))
    np.savez(out / "eval_views.npz", **views, eval_rows=ev_rows)

    pools_all.drop(columns=["eval_start", "eval_end", "fit_start", "fit_end"]).to_csv(
        gout / "control_input_reference_split.tsv", sep="\t", index=False)
    cells.to_csv(gout / "input_cells.tsv.gz", sep="\t", index=False, compression="gzip")

    splits = pd.read_csv(Path(a.manifest_dir) / "splits.tsv", sep="\t").sort_values("group_index")
    fold = {int(r.group_index): int(D.bhash("r3-devfold|" + str(r.group_smiles)) * D.N_FOLDS)
            for r in splits.itertuples() if r.split == "train"}
    role = {s: [int(x) for x in splits.group_index[splits.split == s]] for s in ["train", "val", "test"]}
    nv = lambda ds: int(valid6[:, ds].sum())
    D.dump_json(dict(
        source="manifests/chemcpa_moe_v2/splits.tsv (seed 20261112), unchanged: one role per drug structure group across all cell lines",
        n_groups={k: len(v) for k, v in role.items()}, groups=role,
        dev_fold_rule="blake2b('r3-devfold|' + group SMILES) * 3, TRAIN drugs only (same folds as R3)",
        dev_fold={str(k): v for k, v in fold.items()},
        dev=dict(fit_folds=[0, 1], eval_fold=D.DEV_EVAL_FOLD,
                 n_fit_drugs=sum(1 for v in fold.values() if v != D.DEV_EVAL_FOLD),
                 n_eval_drugs=sum(1 for v in fold.values() if v == D.DEV_EVAL_FOLD),
                 n_fit_conditions=nv([g for g, v in fold.items() if v != D.DEV_EVAL_FOLD]),
                 n_eval_conditions=nv([g for g, v in fold.items() if v == D.DEV_EVAL_FOLD])),
        final=dict(n_fit_drugs=len(role["train"]), n_eval_drugs=len(role["test"]),
                   n_fit_conditions=nv(role["train"]), n_eval_conditions=nv(role["test"])),
        validation_role="the 70 historical VALIDATION drug groups are not used by any step of R6",
        group_smiles={int(r.group_index): str(r.group_smiles) for r in splits.itertuples()}),
        gout / "drug_split.json")
    D.dump_json(dict(
        rule="c' ~ Binomial(c, p) per panel gene; c_rest' ~ Binomial(c_rest, p); library' = sum(c') + c_rest'; p = 1 is the identity",
        normalisation="log1p(1e4 * c' / library') at every p",
        keep_rates=list(D.PS), thinning_seeds=list(D.THIN_SEEDS), n_cells_per_set=D.EVAL_N,
        evaluation_views=vman, n_eval_cells=int(len(ev_rows)),
        sha256_eval_full_panel=D.sha256_array(cnt[ev_rows]), sha256_eval_full_rest=D.sha256_array(crest[ev_rows]),
        training_views="drawn on the fly on the GPU (torch.binomial) from fit_input cells; seeded by the run's training seed; never stored",
        target="never thinned: the treated wells and the reference controls are full observation at every p",
        not_a_replicate="a thinned view is the same cell observed less deeply, not a new cell or a new well"),
        gout / "thinning_manifest.json")
    rec = dict(
        n_pools_seen=int(len(pools_all)), n_pools_eligible=int(len(pools)),
        ineligible_pools=pools_all[~pools_all.eligible][["plate", "cell_line_id", "n_total"]].to_dict("records"),
        pool_size=pools.n_total.describe().to_dict(), n_fit_input=pools.n_fit_input.describe().to_dict(),
        n_reference=pools.n_reference.describe().to_dict(),
        n_input_cells=int(len(cnt)), n_eval_cells=int(len(ev_rows)), n_fit_cells=int(len(fit_rows)),
        roles_rule=f"eval_input = ranks [0,{D.EVAL_N}); fit_input = next min({D.FIT_MAX}, n-{D.EVAL_N}-{D.REF_MIN}); reference = rest; pool needs n >= {2*D.EVAL_N + D.REF_MIN}",
        valid_conditions=dict(v2=int(valid.sum()), r6=int(valid6.sum()),
                              lost_primary_pool=int((valid & ~ok_primary).sum()), lost_missing_secondary_pool=int((valid & ok_primary & ~ok_all).sum())),
        n_multi_plate_conditions=int((valid6 & ((pw > 0).sum(-1) > 1)).sum()),
        truth_shift=dict(note="Y_r6 - Y_v2: effect of removing the input cells from the reference control",
                         rms=float(np.sqrt((shift[valid6] ** 2).mean())), max_abs=float(np.abs(shift[valid6]).max()),
                         truth_rms=float(np.sqrt((Y[valid6].astype(np.float64) ** 2).mean()))),
        reference_noise=dict(note="per-gene standard error of the reference mean relative to truth RMS, median over pools",
                             median_se=float(np.median([np.sqrt(np.asarray(Ln[g.index.to_numpy()[D.pool_roles(len(g))[0] + D.pool_roles(len(g))[1]:]].toarray()).var(0).mean() / D.pool_roles(len(g))[2])
                                                        for _, g in list(meta.groupby(["plate_index", "line"]))[:40] if D.pool_roles(len(g))]))),
        files={f: dict(bytes=os.path.getsize(out / f), sha256=D.sha256_file(out / f))
               for f in ["pools.tsv", "input_cells.tsv", "input_counts.npz", "ref_mean.npy", "Y_r6.npy", "valid_r6.npy", "gene_stats.npz", "eval_views.npz"]},
        contract_dir=str(out))
    D.dump_json(rec, gout / "contract_record.json")
    print(json.dumps(rec, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
