"""C3: fit one mixing weight t* per episode on TRAIN drugs (cross-replicate triplets).

The re-read composition of the empirical mixture is exactly (1-t) pi_L + t pi_H (fixed memberships),
so the drug-balanced train composition TV is evaluated on a grid t in {0, 0.025, ..., 1}. Development
and test drugs are not used."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .core import LINES, LineData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--atlas_K", required=True)
    ap.add_argument("--direction", default="cross")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert not os.path.exists(args.out), "immutable output exists"
    cfg = json.load(open(args.config))
    Kmap = json.load(open(args.atlas_K))
    trip = pd.read_csv(args.triplets, sep="\t")
    t = trip[(trip.role == "train") & (trip.direction == args.direction) & trip.eligible]
    grid = np.round(np.arange(0, 1.0001, 0.025), 3)
    rows = []
    for line in LINES:
        D = LineData(cfg, line, Kmap[line])
        for r in t[t.line == line].itertuples():
            e = cfg["episodes"][r.episode]
            pL = D.S[D.cells(r.drug, e["low"], r.rep_endpoint)].mean(0)
            pH = D.S[D.cells(r.drug, e["high"], r.rep_endpoint)].mean(0)
            pQ = D.S[D.cells(r.drug, e["query"], r.rep_query)].mean(0)
            for g in grid:
                rows.append(dict(line=line, drug=r.drug, episode=r.episode, t=g,
                                 tv=0.5 * np.abs((1 - g) * pL + g * pH - pQ).sum()))
    df = pd.DataFrame(rows)
    curve = (df.groupby(["episode", "t", "line", "drug"]).tv.mean().groupby(["episode", "t", "drug"]).mean()
             .groupby(["episode", "t"]).mean().reset_index())
    best = curve.loc[curve.groupby("episode").tv.idxmin()]
    out = dict(direction=args.direction, t_star={r.episode: float(r.t) for r in best.itertuples()},
               train_tv_at_t_star={r.episode: float(r.tv) for r in best.itertuples()},
               train_tv_at_half={ep: float(curve[(curve.episode == ep) & (curve.t == 0.5)].tv.iloc[0])
                                 for ep in curve.episode.unique()},
               curve=curve.round(5).to_dict(orient="records"))
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}))


if __name__ == "__main__":
    main()
