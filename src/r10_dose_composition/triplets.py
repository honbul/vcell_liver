"""B0: endpoint-query triplets (24 h) with eligibility and exclusion reasons.

One row per line x drug x episode x (endpoint replicate, query replicate). Cross-replicate rows
(endpoints from one culture replicate, query from the other) are the primary contract; same-
replicate rows are a secondary, separately reported condition. Eligibility uses only QC cell
counts of the wells (no RNA); the query count is used for eligibility/scoring, never as a
prediction input.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    w = pd.read_csv(f"{cfg['cache_dir']}/wells.tsv", sep="\t")
    split = pd.read_csv(args.split, sep="\t").set_index("drug")
    nmin, cmin = cfg["min_cells_well"], cfg["min_cells_control"]
    nv = w[~w.vehicle]
    assert not nv.duplicated(["line", "drug", "dose", "replicate"]).any()
    key = {(r.line, r.drug, float(r.dose), r.replicate): (int(r.n_qc), r.culture_plate) for r in nv.itertuples()}
    ctrl = w[w.vehicle].groupby(["line", "replicate"]).n_qc.sum()
    rows = []
    for line in sorted(w.line.unique()):
        for drug in split.index:
            for ep, e in cfg["episodes"].items():
                for re_ in ("rep1", "rep2"):
                    for rq in ("rep1", "rep2"):
                        def get(dose, rep):
                            return key.get((line, drug, float(dose), rep), (0, ""))
                        nl, pl = get(e["low"], re_)
                        nh, ph = get(e["high"], re_)
                        nq, pq = get(e["query"], rq)
                        nc = int(ctrl.get((line, re_), 0))
                        why = []
                        if nl < nmin:
                            why.append(f"low_endpoint_n<{nmin}")
                        if nh < nmin:
                            why.append(f"high_endpoint_n<{nmin}")
                        if nq < nmin:
                            why.append(f"query_n<{nmin}")
                        if nc < cmin:
                            why.append(f"control_n<{cmin}")
                        rows.append(dict(line=line, drug=drug, product_name=split.loc[drug, "product_name"],
                                         role=split.loc[drug, "role"], episode=ep, low_nM=e["low"],
                                         high_nM=e["high"], query_nM=e["query"], rep_endpoint=re_, rep_query=rq,
                                         direction="cross" if re_ != rq else "same", n_low=nl, n_high=nh,
                                         n_query=nq, n_control=nc, plate_low=pl, plate_high=ph, plate_query=pq,
                                         endpoints_same_plate=pl == ph, query_same_plate_as_endpoints=(pq == pl == ph),
                                         eligible=not why, exclusion_reason=";".join(why)))
    t = pd.DataFrame(rows)
    t.to_csv(args.out, sep="\t", index=False)
    c = t[t.direction == "cross"]
    print(c.groupby(["line", "episode", "role"]).eligible.agg(["sum", "size"]).to_string())
    d = c[c.eligible].groupby(["line", "episode", "role"]).drug.nunique()
    print("drugs with >=1 eligible cross triplet\n", d.to_string())


if __name__ == "__main__":
    main()
