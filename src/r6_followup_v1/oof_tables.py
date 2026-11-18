"""Source out-of-fold comparison of retrained arms (plan 6.2 controls and branch tables).

Every TRAIN drug is predicted by the models of the fold that held it out. A spec file names the
arms (run-id pattern with {fold} and {seed}) and the pairs to report. Per arm and keep rate p the
predictions of all training seeds x thinning seeds are scored and averaged per condition before
the drug-group bootstrap; within-pool groups are (fold, pool).

Response-mean baselines (pool mean with the Phase-A kappa, line mean) are refit per fold from the
fold's fitting drugs and added as arms `poolmean` and `linemean`.
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

PS = (1.0, 0.5, 0.25)


def oof_conditions(d):
    Ls, Ds, Fs = [], [], []
    for k in C.FOLDS:
        _, held = C.source_split(d, k)
        L, Dg = d.conditions(held)
        Ls.append(L); Ds.append(Dg); Fs.append(np.full(len(L), k))
    return np.concatenate(Ls), np.concatenate(Ds), np.concatenate(Fs)


def mean_baselines(d, L, Dg, fold, pool, kappa):
    out = {k: np.zeros((len(L), d.Y.shape[2]), np.float32) for k in ("poolmean", "linemean")}
    for k in C.FOLDS:
        fit, _ = C.source_split(d, k)
        Lf, Df = d.conditions(fit)
        mb = MeanBaseline(d.Y, Lf, Df, C.pool_of(d, Lf, Df), d.Y.shape[0], len(d.pools))
        q = fold == k
        out["poolmean"][q] = mb.predict(L[q], Dg[q], pool[q], kappa)[0]
        out["linemean"][q] = mb.predict(L[q], Dg[q], pool[q], float("inf"))[0]
    return out


def load_arm(d, stage, pattern, seeds, L, Dg, fold, folds=C.FOLDS):
    """{p: [arrays over seeds x thinning seeds]}; conditions outside `folds` stay NaN-free only if all folds are given."""
    out = {p: [] for p in PS}
    recs = []
    for s in seeds:
        per = {}
        for k in folds:
            run = C.assets() / "runs" / stage / pattern.format(fold=k, seed=s)
            q = fold == k
            per[k] = C.load_pred(run / "pred.npz", L[q], Dg[q])
            recs.append(json.loads((run / "record.json").read_text()))
        for p in PS:
            for key in C.PKEYS[p]:
                A = np.zeros((len(L), d.Y.shape[2]), np.float32)
                for k in folds:
                    A[fold == k] = per[k][key]
                out[p].append(A)
    return out, recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    seeds = spec.get("seeds", list(C.SEEDS))
    folds = tuple(spec.get("folds", C.FOLDS))
    d = C.load(eval_views=False)
    L, Dg, fold = oof_conditions(d)
    keep = np.isin(fold, folds)
    L, Dg, fold = L[keep], Dg[keep], fold[keep]
    pool = C.pool_of(d, L, Dg)
    kappa = json.loads(Path("manifests/r6_followup_v1/kappa_selection.json").read_text())["chosen_kappa"]
    masks = {f"fold{k}": fold == k for k in folds} if len(folds) > 1 else {}
    sc = Scorer(d.Y[L, Dg], L, Dg, pool, fold, masks=masks, n_boot=a.n_boot, seed=0)
    role = spec.get("split_role", "source_oof")
    base = mean_baselines(d, L, Dg, fold, pool, kappa)
    names, run_rows = [], []
    for bn, P in base.items():
        for p in PS:
            sc.add(f"{bn}@{p}", [P])
        names.append(bn)
    for arm, item in spec["arms"].items():
        preds, recs = load_arm(d, item["stage"], item["pattern"], item.get("seeds", seeds), L, Dg, fold, folds)
        for p in PS:
            sc.add(f"{arm}@{p}", preds[p])
        if spec.get("per_seed"):
            nth = len(C.PKEYS[0.25])
            for i, s in enumerate(item.get("seeds", seeds)):
                sc.add(f"{arm}_s{s}@0.25", preds[0.25][i * nth:(i + 1) * nth], retrieval=False)
        names.append(arm)
        for r in recs:
            run_rows.append(dict(arm=arm, run_id=r["run_id"], stage=r["stage"], fold=r["fold"], seed=r["seed"], lr=r.get("lr"),
                                 n_params=(r.get("params") or {}).get("total"), updates=r.get("updates"), train_seconds=r.get("train_seconds", r.get("seconds")),
                                 peak_vram_MB=r.get("peak_vram_MB"), gpu=r.get("gpu"), job=r.get("job"), alphas=json.dumps(r.get("alphas")) if r.get("alphas") else "",
                                 anchor_kappa=r.get("anchor_kappa"), pair_lambda=r.get("pair_lambda"), distinct_views_used=r.get("distinct_views_used"),
                                 heldfold_p025_s0=(r.get("scores") or {}).get("p0.25_s0"), commit=r.get("commit"), pred_sha256=r.get("pred_sha256")))
        print("loaded", arm, flush=True)
    rows = []
    std = C.std_fields(split_role=role, fold=",".join(str(k) for k in folds))
    for n in names:
        for p in PS:
            for mk in sc.masks:
                s = sc.summary(f"{n}@{p}", mk)
                r = dict(std, row_type="arm", arm=n, comparator="", p=p, model_seed=",".join(map(str, seeds)) if n not in base else "", **s)
                if mk == "all":
                    ret = {x["candidate_pool"]: x for x in sc.retrieval_rows(f"{n}@{p}")}
                    r.update(pool_top1=ret["pool_centred"]["top1"], pool_top1_lo=ret["pool_centred"]["top1_lo"], pool_top1_hi=ret["pool_centred"]["top1_hi"],
                             pool_mrr=ret["pool_centred"]["mrr"], pool_chance_top1=ret["pool_centred"]["chance_top1"], pool_chance_mrr=ret["pool_centred"]["chance_mrr"],
                             line_top1=ret["line_all"]["top1"], line_chance_top1=ret["line_all"]["chance_top1"], line_mrr=ret["line_all"]["mrr"])
                rows.append(r)
    for c, b in spec["pairs"]:
        for p in PS:
            for mk in sc.masks:
                pr = sc.pair(f"{c}@{p}", f"{b}@{p}", mk)
                pr.update(candidate=c, comparator=b)
                r = dict(std, row_type="pair", arm=c, p=p, model_seed=",".join(map(str, seeds)), **pr)
                if mk == "all":
                    rp = {x["candidate_pool"]: x for x in sc.retrieval_pair(f"{c}@{p}", f"{b}@{p}")}["pool_centred"]
                    r.update(pool_top1_diff=rp["top1_diff"], pool_top1_diff_lo=rp["top1_lo"], pool_top1_diff_hi=rp["top1_hi"], pool_mrr_diff=rp["mrr_diff"])
                rows.append(r)
        if spec.get("per_seed") and c not in base and b not in base:
            for s in seeds:
                pr = sc.pair(f"{c}_s{s}@0.25", f"{b}_s{s}@0.25")
                pr.update(candidate=c, comparator=b)
                rows.append(dict(std, row_type="pair_single_seed", arm=c, p=0.25, model_seed=str(s), **pr))
    # depth interaction of pairs: (L_comp - L_cand at p=0.25) - (same at p=1); positive = candidate's advantage grows at low depth
    for c, b in spec.get("depth_pairs", []):
        for key in ("total", "within"):
            col = "v" if key == "total" else "lw"
            g = {}
            for p in (0.25, 1.0):
                ca, cb = sc.arms[f"{c}@{p}"], sc.arms[f"{b}@{p}"]
                g[p] = (M.macro_point(cb[col], sc.A["all"])[0] - M.macro_point(ca[col], sc.A["all"])[0], cb["bs"]["all"][key] - ca["bs"]["all"][key])
            lo, hi = M.ci(g[0.25][1] - g[1.0][1])
            rows.append(dict(std, row_type="depth_interaction", arm=c, comparator=b, candidate=c, subset="all", metric=key,
                             gain_p025=g[0.25][0], gain_p1=g[1.0][0], I_depth=g[0.25][0] - g[1.0][0], I_lo=lo, I_hi=hi,
                             sign="I = {L_comp(0.25) - L_cand(0.25)} - {L_comp(1) - L_cand(1)}; positive = candidate gains more at low depth"))
    out = pd.DataFrame(rows)
    out["metric_definition"] = f"total: {DEF_TOTAL} | within: {DEF_WITHIN}"
    out["drug_group"] = "structure group; bootstrap unit"
    Path(spec["out"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(spec["out"], sep="\t", index=False, float_format="%.6f")
    if run_rows:
        pd.DataFrame(run_rows).to_csv(spec["out"].replace(".tsv", "_runs.tsv"), sep="\t", index=False)
    if spec.get("per_condition"):
        pc = pd.DataFrame(dict(line=L, drug_group=Dg, fold=fold, pool=pool))
        for n in names:
            for p in (1.0, 0.25):
                pc[f"rmse_{n}_p{p}"] = sc.arms[f"{n}@{p}"]["v"]; pc[f"within_{n}_p{p}"] = sc.arms[f"{n}@{p}"]["lw"]
        pc.to_csv(spec["per_condition"], sep="\t", index=False, float_format="%.5f", compression="gzip")
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    show = out[(out.subset == "all")]
    print(show[show.row_type == "arm"][["arm", "p", "n_drugs", "n_units", "total_rmse", "total_lo", "total_hi", "within_rmse", "within_lo", "within_hi",
                                        "centred_magnitude_ratio", "pool_top1", "pool_chance_top1", "line_top1"]].to_string())
    print(show[show.row_type.isin(["pair", "pair_single_seed"])][["row_type", "candidate", "comparator", "p", "model_seed", "total_diff", "total_lo", "total_hi", "total_rel", "total_category",
                                                                  "within_diff", "within_lo", "within_hi", "within_rel", "within_category"]].to_string())
    if (out.row_type == "depth_interaction").any():
        print(out[out.row_type == "depth_interaction"][["candidate", "comparator", "metric", "gain_p025", "gain_p1", "I_depth", "I_lo", "I_hi"]].to_string())
    fm = out[(out.row_type == "pair") & (out.subset != "all") & (out.p == 0.25)]
    if len(fm):
        print(fm[["candidate", "comparator", "subset", "total_diff", "total_rel", "within_diff", "within_rel"]].to_string())


if __name__ == "__main__":
    main()
