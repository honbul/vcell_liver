"""Branch F2: plate transfer on TRAIN drugs (leave one plate out).

For every plate P the models were fitted on TRAIN drugs that have no condition on P, so neither the
drugs nor any control pool of P was seen. Scored: the conditions of the held drugs whose primary
plate is P. A response-mean baseline cannot use a pool mean here (no response of that pool is
available): the transfer baseline is the cell-line mean of the fitting drugs.

Reference rows that are NOT transfer (they use responses measured on the held plate):
    poolmean_inplate   shrunk pool mean from the other drug-fold's TRAIN drugs (Phase C baseline)
    SD_inplate         the Phase C SD models (drug folds; pools seen in training)
gap_closed = (linemean_transfer - arm) / (linemean_transfer - poolmean_inplate): the share of the
pool-level gain that an arm recovers without any response from the held plate.
The plate is the unit of generalisation and there are only five; drug-group bootstrap intervals
describe drug sampling only, so the per-plate rows are part of the result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import metrics as M
from .baselines import MeanBaseline
from .scoring import DEF_TOTAL, DEF_WITHIN, Scorer

PS = (1.0, 0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/r6_followup_v1/plate_transfer.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    seeds = spec["seeds"]
    d = C.load(eval_views=False)
    plates = sorted(int(p) for p in d.pools.plate_index.unique())
    kappa = json.loads(Path("manifests/r6_followup_v1/kappa_selection.json").read_text())["chosen_kappa"]
    G = d.Y.shape[2]
    Ls, Ds, Ps, idxs, n_fit = [], [], [], {}, {}
    for P in plates:
        fit, held = C.plate_split(d, P)
        L, Dg, ix = C.plate_conditions(d, held, P)
        Ls.append(L); Ds.append(Dg); Ps.append(np.full(len(L), P)); idxs[P] = ix; n_fit[P] = len(fit)
    L, Dg, plate = np.concatenate(Ls), np.concatenate(Ds), np.concatenate(Ps)
    pool = C.pool_of(d, L, Dg)
    masks = {f"plate{P}": plate == P for P in plates}
    sc = Scorer(d.Y[L, Dg], L, Dg, pool, plate, masks=masks, n_boot=a.n_boot, seed=0)
    # baselines
    lm = np.zeros((len(L), G), np.float32); pm = np.zeros((len(L), G), np.float32)
    for P in plates:
        fit, _ = C.plate_split(d, P)
        Lf, Df = d.conditions(fit)
        mb = MeanBaseline(d.Y, Lf, Df, C.pool_of(d, Lf, Df), d.Y.shape[0], len(d.pools))
        q = plate == P
        pr, level, _ = mb.predict(L[q], Dg[q], pool[q], kappa)
        assert (level != "pool").all(), "a pool of the held plate was seen in fitting"
        lm[q] = pr
    for k in C.FOLDS:
        fit, _ = C.source_split(d, k)
        Lf, Df = d.conditions(fit)
        mb = MeanBaseline(d.Y, Lf, Df, C.pool_of(d, Lf, Df), d.Y.shape[0], len(d.pools))
        q = d.fold[Dg] == k
        pm[q] = mb.predict(L[q], Dg[q], pool[q], kappa)[0]
    names = ["linemean_transfer", "poolmean_inplate"]
    for p in PS:
        sc.add(f"linemean_transfer@{p}", [lm]); sc.add(f"poolmean_inplate@{p}", [pm])

    def cond_index(Lq, Dq):
        key = {(int(l), int(g)): i for i, (l, g) in enumerate(zip(Lq, Dq))}
        return np.array([key[(int(l), int(g))] for l, g in zip(L, Dg)])

    for arm, item in spec["arms"].items():
        preds = {p: [] for p in PS}
        for s in item.get("seeds", seeds):
            per = {}
            if item.get("kind") == "inplate":                          # Phase C drug-fold models, same conditions
                for k in C.FOLDS:
                    held = C.source_split(d, k)[1]
                    Lk, Dk = d.conditions(held)
                    z = C.load_pred(C.assets() / "runs" / item["stage"] / item["pattern"].format(fold=k, seed=s) / "pred.npz", Lk, Dk)
                    q = d.fold[Dg] == k
                    key = {(int(l), int(g)): i for i, (l, g) in enumerate(zip(Lk, Dk))}
                    ix = np.array([key[(int(l), int(g))] for l, g in zip(L[q], Dg[q])])
                    per[k] = (q, {kk: v[ix] for kk, v in z.items()})
            else:
                for P in plates:
                    held = C.plate_split(d, P)[1]
                    Lh, Dh = d.conditions(held)
                    z = C.load_pred(C.assets() / "runs" / item["stage"] / item["pattern"].format(plate=P, seed=s) / "pred.npz", Lh, Dh)
                    per[P] = (plate == P, {kk: v[idxs[P]] for kk, v in z.items()})
            for p in PS:
                for key_ in C.PKEYS[p]:
                    A = np.zeros((len(L), G), np.float32)
                    for q, z in per.values():
                        A[q] = z[key_]
                    preds[p].append(A)
        for p in PS:
            sc.add(f"{arm}@{p}", preds[p])
        names.append(arm); print("loaded", arm, flush=True)
    rows = []
    std = C.std_fields(split_role="source_plate_transfer", fold="held plate")
    for n in names:
        for p in PS:
            for mk in sc.masks:
                s = sc.summary(f"{n}@{p}", mk)
                r = dict(std, row_type="arm", arm=n, comparator="", p=p, model_seed=",".join(map(str, seeds)), **s)
                if mk == "all":
                    ret = {x["candidate_pool"]: x for x in sc.retrieval_rows(f"{n}@{p}")}
                    r.update(pool_top1=ret["pool_centred"]["top1"], pool_chance_top1=ret["pool_centred"]["chance_top1"])
                rows.append(r)
    for c, b in spec["pairs"]:
        for p in PS:
            for mk in sc.masks:
                pr = sc.pair(f"{c}@{p}", f"{b}@{p}", mk); pr.update(candidate=c, comparator=b)
                rows.append(dict(std, row_type="pair", arm=c, p=p, model_seed=",".join(map(str, seeds)), **pr))
    for n in names:
        if n in ("linemean_transfer", "poolmean_inplate"):
            continue
        for p in PS:
            for mk in sc.masks:
                A = sc.A[mk]
                pt = {x: M.macro_point(sc.arms[f"{x}@{p}"]["v"], A)[0] for x in (n, "linemean_transfer", "poolmean_inplate")}
                bs = {x: sc.arms[f"{x}@{p}"]["bs"][mk]["total"] for x in (n, "linemean_transfer", "poolmean_inplate")}
                gc = (pt["linemean_transfer"] - pt[n]) / (pt["linemean_transfer"] - pt["poolmean_inplate"])
                gb = (bs["linemean_transfer"] - bs[n]) / (bs["linemean_transfer"] - bs["poolmean_inplate"])
                lo, hi = M.ci(gb)
                rows.append(dict(std, row_type="gap_closed", arm=n, comparator="linemean_transfer -> poolmean_inplate", candidate=n, p=p, subset=mk,
                                 model_seed=",".join(map(str, seeds)), gap_closed=gc, gap_lo=lo, gap_hi=hi,
                                 total_candidate=pt[n], linemean_transfer=pt["linemean_transfer"], poolmean_inplate=pt["poolmean_inplate"]))
    out = pd.DataFrame(rows)
    # The in-plate references are leave-drug-fold-out predictions, so inside one (plate, pool) group they are NOT
    # constant across drugs (each drug fold gets a different pool mean). Centring by (plate, pool) therefore gives
    # them a spurious within-pool contrast / retrieval value. Those fields are blanked; their within-pool numbers
    # under the correct (drug fold, pool) grouping are in input_capacity_controls.tsv.
    inpl = out.arm.astype(str).str.contains("inplate") | out.get("comparator", "").astype(str).str.contains("inplate")
    inpl &= out.row_type.isin(["arm", "pair"])
    for c in out.columns:
        if c.startswith("within") or c.startswith("pool_"):
            out.loc[inpl, c] = np.nan
    out["note"] = ""
    out.loc[inpl, "note"] = "within-pool and retrieval fields blank: in-plate reference is not constant inside a (plate, pool) group"
    out["metric_definition"] = f"total: {DEF_TOTAL} | within: {DEF_WITHIN}"
    out["n_fit_drugs_by_plate"] = json.dumps(n_fit)
    out.to_csv(spec["out"], sep="\t", index=False, float_format="%.6f")
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    al = out[out.subset == "all"]
    print("n_fit", n_fit, {k: int(v.sum()) for k, v in masks.items()}, {k: len(set(Dg[v])) for k, v in masks.items()})
    print(al[al.row_type == "arm"][["arm", "p", "n_drugs", "n_units", "total_rmse", "total_lo", "total_hi", "within_rmse", "pool_top1", "pool_chance_top1"]].to_string())
    print(al[al.row_type == "pair"][["candidate", "comparator", "p", "total_diff", "total_lo", "total_hi", "total_rel", "total_category", "within_diff", "within_lo", "within_hi", "within_category"]].to_string())
    g = out[out.row_type == "gap_closed"]
    print(g[["arm", "p", "subset", "gap_closed", "gap_lo", "gap_hi", "total_candidate", "linemean_transfer", "poolmean_inplate"]].to_string())


if __name__ == "__main__":
    main()
