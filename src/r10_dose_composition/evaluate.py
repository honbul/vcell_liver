"""Score predictors on eligible triplets of one role (train / development / test) and one line.

Every predictor sees only `Context` (endpoints of the endpoint replicate, its vehicle cells, the
requested dose). The query well is built separately and handed to the scorer only. The query cell
number is used by the scorer (n of the observed sample) and for eligibility, nowhere else.
`oracle_replicate` (the query dose of the endpoint replicate) is a diagnostic of the observational
limit, flagged `is_oracle=1`, and never part of a performance table.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib

import numpy as np
import pandas as pd
import torch

from .core import Cloud, LineData, build_context, build_query, query_diagnostics, score
from .methods import Head, baseline, predict_learned

BASELINES = ["control", "nearest_low", "nearest_high", "mixture", "global_moment", "state_moment", "ot_displacement"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--atlas_K", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--line", required=True)
    ap.add_argument("--methods", default=",".join(BASELINES + ["oracle_replicate"]))
    ap.add_argument("--heads_dir", default="")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--directions", default="cross,same")
    ap.add_argument("--episodes", default="E100,E1000")
    ap.add_argument("--endpoint_budget", type=int, default=0)
    ap.add_argument("--budget_draw", type=int, default=0)
    ap.add_argument("--t_star_json", default="", help="C3: train-fitted mixing weight per episode")
    ap.add_argument("--drugs", default="", help="optional comma list (pilot only)")
    # v2 contract keys (R10-E1): every row carries the training contract of the arm that produced it
    ap.add_argument("--train_contract", default="none",
                    help="replicate contract the learned heads in --heads_dir were trained/selected on (cross/same)")
    ap.add_argument("--assembly_version", default="v1_soft_empirical_residual")
    ap.add_argument("--atlas_version", default="r10_v1_trainonly_pca20_K4")
    ap.add_argument("--max_triplets", type=int, default=0, help="pilot only: first n triplets")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    tstar_contract = "none"
    if args.t_star_json:
        tj = json.load(open(args.t_star_json))
        cfg["mixture_t_star"] = tj["t_star"]
        tstar_contract = tj.get("direction", "cross")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    assert not os.path.exists(args.out), "immutable run output exists"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    Kmap = json.load(open(args.atlas_K))
    K_pad = max(Kmap.values())
    D = LineData(cfg, args.line, Kmap[args.line])
    trip = pd.read_csv(args.triplets, sep="\t")
    t = trip[(trip.role == args.role) & (trip.line == args.line) & trip.eligible
             & trip.direction.isin(args.directions.split(",")) & trip.episode.isin(args.episodes.split(","))]
    if args.drugs:
        t = t[t.drug.isin(args.drugs.split(","))]
    if args.max_triplets:
        t = t.iloc[:args.max_triplets]
    methods = args.methods.split(",")
    learned = [m for m in methods if m in ("D10", "D01", "D11", "W")]
    if learned:
        assert args.train_contract in ("cross", "same"), "learned arms need --train_contract"
    seeds = [int(s) for s in args.seeds.split(",")]
    heads = {}
    for s in seeds if learned else []:
        h = {}
        for kind in ("pi", "Q", "W"):
            p = f"{args.heads_dir}/{kind}_seed{s}.pt"
            if os.path.exists(p):
                h[kind] = Head.load(p)
                tc = torch.load(p, weights_only=False).get("train_contract")
                assert tc in (None, args.train_contract), f"{p}: train_contract {tc} != {args.train_contract}"
        heads[s] = h
    log(args.role, args.line, "triplets", len(t), "methods", methods, "K", D.K)
    rows = []
    t0 = time.time()
    for i, r in enumerate(t.itertuples()):
        e = cfg["episodes"][r.episode]
        ctx = build_context(D, r.drug, r.episode, e, r.rep_endpoint)
        if args.endpoint_budget:
            rng = np.random.default_rng(zlib.crc32(f"{r.line}|{r.drug}|{r.episode}|{r.rep_endpoint}|{args.budget_draw}".encode()))
            for a, b in (("ZL", "SL"), ("ZH", "SH")):
                Z = getattr(ctx, a)
                if len(Z) > args.endpoint_budget:
                    k = rng.choice(len(Z), args.endpoint_budget, replace=False)
                    setattr(ctx, a, Z[k]); setattr(ctx, b, getattr(ctx, b)[k])
        q = build_query(D, r.drug, e["query"], r.rep_query)
        base = dict(line=r.line, drug=r.drug, role=r.role, episode=r.episode, rep_endpoint=r.rep_endpoint,
                    rep_query=r.rep_query, direction=r.direction, n_low=len(ctx.ZL), n_high=len(ctx.ZH),
                    n_control=len(ctx.ZC), endpoint_budget=args.endpoint_budget, budget_draw=args.budget_draw,
                    **query_diagnostics(D, q))
        seed = zlib.crc32(f"{r.line}|{r.drug}|{r.episode}|{r.rep_endpoint}".encode())
        for m in methods:
            for s in (seeds if m in learned else [None]):
                tm = time.time()
                if m == "oracle_replicate":
                    if r.direction != "cross":
                        continue
                    c = D.cells(r.drug, e["query"], r.rep_endpoint)
                    if c is None or len(c) < cfg["min_cells_well"]:
                        continue
                    P = Cloud.uniform(D.Z[c])
                elif m in learned:
                    P = predict_learned(m, ctx, cfg, heads[s], K_pad)
                else:
                    P = baseline(m, ctx, cfg, seed)
                sc = score(D, P, q, cfg["shrink"]["min_state_neff_score"])
                tcon = args.train_contract if m in learned else (tstar_contract if m == "mixture_t" else "none")
                rows.append(dict(base, method=m, seed=-1 if s is None else s, is_oracle=int(m == "oracle_replicate"),
                                 train_contract=tcon, assembly_version=args.assembly_version,
                                 atlas_version=args.atlas_version,
                                 n_points=len(P.X), pred_neff=float(1 / (P.w ** 2).sum()), sec=time.time() - tm,
                                 **{k: v for k, v in P.meta.items() if k.startswith("diag_")}, **sc))
        if i % 50 == 0:
            log(i, "/", len(t), f"{time.time() - t0:.0f}s")
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    log("done", len(rows), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
