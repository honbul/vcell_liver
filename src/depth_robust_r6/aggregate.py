"""Tables from stored predictions: main table, paired differences, guard, cell lines, SNU-423.

Spec (json): {"setting": "dev"|"final", "arms": {arm: [prediction npz paths relative to
$R6_ASSETS/runs/<setting>/]}, "pairs": [[a, b], ...], "out": "<report table dir>", "tag": "..."}.
Each npz holds line, drug and one array per view "p{p}_s{seed}". Per condition, RMSE is averaged
over the arm's training seeds (files) and the thinning seeds; uncertainty is a paired bootstrap
over drug groups. Nothing is fitted or selected here.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import evaluate as E

PKEYS = {1.0: ["p1.0_s0"], 0.5: [f"p0.5_s{s}" for s in D.THIN_SEEDS], 0.25: [f"p0.25_s{s}" for s in D.THIN_SEEDS]}
SEC = ["pearson", "centred_pearson", "top20_direction", "magnitude_ratio", "retrieval_rank_score", "retrieval_top1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    setting, tag = spec["setting"], spec.get("tag", spec["setting"])
    root = Path(os.environ["R6_ASSETS"]) / "runs" / setting
    out = Path(spec["out"]); out.mkdir(parents=True, exist_ok=True)
    d = D.load(eval_views=False)
    _, eval_drugs = d.split_spec(setting)
    L, Dg = d.conditions(eval_drugs)
    T = d.Y[L, Dg].astype(np.float64)
    cid = d.lines.cell_line_id.to_numpy()[L]
    rm, sec, files = {}, {}, {}
    for arm, paths in spec["arms"].items():
        rm[arm] = {p: np.zeros(len(L)) for p in PKEYS}
        sec[arm] = {p: {k: np.zeros(len(L)) for k in SEC} for p in PKEYS}
        files[arm] = []
        for rel in paths:
            z = np.load(root / rel)
            assert (z["line"] == L).all() and (z["drug"] == Dg).all(), rel
            files[arm].append(dict(path=rel, sha256=D.sha256_file(root / rel)))
            for p, keys in PKEYS.items():
                for k in keys:
                    P = z[k].astype(np.float32)
                    w = 1.0 / (len(paths) * len(keys))
                    rm[arm][p] += w * E.cond_rmse(P, T)
                    s = E.secondaries(P, T.astype(np.float32), L, Dg)
                    for q in SEC:
                        sec[arm][p][q] += w * np.nan_to_num(s[q], nan=0.5)
    arms = list(spec["arms"])
    main_rows, pair_rows, line_rows, snu_rows = [], [], [], []
    for arm in arms:
        for p in PKEYS:
            b = E.bootstrap_mean(rm[arm][p], Dg)
            row = dict(tag=tag, arm=arm, p=p, n_seeds=len(spec["arms"][arm]), delta_rmse=b["mean"], lo=b["lo"], hi=b["hi"])
            for q in SEC:
                row[q] = E.drug_macro(sec[arm][p][q], Dg)
            main_rows.append(row)
    for x, y in spec.get("pairs", []):
        if x not in rm or y not in rm:
            continue
        for p in PKEYS:
            b = E.paired_bootstrap(rm[x][p], rm[y][p], Dg)
            pair_rows.append(dict(tag=tag, a=x, b=y, p=p, diff=b["diff"], lo=b["lo"], hi=b["hi"], rel=b["rel"],
                                  frac_drugs_a_better=b["frac_drugs_a_better"], n_drugs=b["n_drugs"],
                                  ci_excludes_0=bool(b["lo"] > 0 or b["hi"] < 0),
                                  better_by_rule=bool(b["hi"] < 0 and b["rel"] <= -0.01),
                                  worse_by_rule=bool(b["lo"] > 0 and b["rel"] >= 0.01)))
    for l in np.unique(L):
        m = L == l
        for arm in arms:
            line_rows.append(dict(tag=tag, cell_line_id=d.lines.cell_line_id[l], n_drugs=int(m.sum()), arm=arm,
                                  **{f"p{p}": float(rm[arm][p][m].mean()) for p in PKEYS}))
    snu = np.flatnonzero(d.lines.cell_line_id.to_numpy() == E.SNU423)
    if len(snu):
        m = L == snu[0]
        for arm in arms:
            for p in PKEYS:
                b = E.bootstrap_mean(rm[arm][p][m], Dg[m])
                snu_rows.append(dict(tag=tag, kind="arm", a=arm, b="", p=p, value=b["mean"], lo=b["lo"], hi=b["hi"], n_drugs=int(m.sum())))
        for x, y in spec.get("pairs", []):
            if x in rm and y in rm:
                for p in PKEYS:
                    b = E.paired_bootstrap(rm[x][p][m], rm[y][p][m], Dg[m])
                    snu_rows.append(dict(tag=tag, kind="pair", a=x, b=y, p=p, value=b["diff"], lo=b["lo"], hi=b["hi"], n_drugs=b["n_drugs"]))
    pd.DataFrame(main_rows).to_csv(out / f"{tag}_main.tsv", sep="\t", index=False, float_format="%.6f")
    pd.DataFrame(pair_rows).to_csv(out / f"{tag}_pairs.tsv", sep="\t", index=False, float_format="%.6f")
    pd.DataFrame(line_rows).to_csv(out / f"{tag}_by_line.tsv", sep="\t", index=False, float_format="%.6f")
    pd.DataFrame(snu_rows).to_csv(out / f"{tag}_snu423.tsv", sep="\t", index=False, float_format="%.6f")
    D.dump_json(dict(tag=tag, setting=setting, n_conditions=int(len(L)), n_drugs=int(len(np.unique(Dg))), files=files,
                     commit=os.popen("git rev-parse HEAD").read().strip()), out / f"{tag}_inputs.json")
    # per-condition metric table (IDs + metadata + every arm's RMSE at every p)
    pc = pd.DataFrame(dict(cell_line_id=cid, line=L, drug_group=Dg, plate_index=d.plate[L, Dg],
                           split=d.splits.split.to_numpy()[Dg], truth_rms=np.sqrt((T ** 2).mean(1))))
    for arm in arms:
        for p in PKEYS:
            pc[f"{arm}_p{p}"] = rm[arm][p]
    pdir = Path(spec.get("pred_out", "predictions/depth_robust_r6")); pdir.mkdir(parents=True, exist_ok=True)
    pc.to_csv(pdir / f"{tag}_per_condition_rmse.tsv.gz", sep="\t", index=False, float_format="%.5f", compression="gzip")
    print(pd.DataFrame(main_rows)[["arm", "p", "n_seeds", "delta_rmse", "lo", "hi", "centred_pearson", "top20_direction", "magnitude_ratio", "retrieval_rank_score"]].to_string(index=False))
    print(pd.DataFrame(pair_rows).to_string(index=False))


if __name__ == "__main__":
    main()
