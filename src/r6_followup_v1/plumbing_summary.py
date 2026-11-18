"""Phase B summary: inner-development scores of the plumbing runs (fold 2, seed 42) next to the
response-mean baselines refit on the same inner-fit drugs. Used only to fix the learning rate of
each arm for the confirmation and to check that every arm trains."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import metrics as M
from .baselines import MeanBaseline
from .scoring import DEF_TOTAL, DEF_WITHIN


def main() -> None:
    d = C.load(eval_views=False)
    fit, dev = C.source_split(d, 2, inner=True)
    Lf, Df = d.conditions(fit); L, Dg = d.conditions(dev)
    pool = C.pool_of(d, L, Dg)
    T = d.Y[L, Dg].astype(np.float64)
    groups = M.groups_of(np.zeros(len(L), int), pool)
    kappa = json.loads(Path("manifests/r6_followup_v1/kappa_selection.json").read_text())["chosen_kappa"]
    mb = MeanBaseline(d.Y, Lf, Df, C.pool_of(d, Lf, Df), d.Y.shape[0], len(d.pools))

    def score(P):
        return M.drug_macro(M.cond_rmse(P, T), Dg), M.drug_macro(M.within_point(M.grams(P - T, groups), groups, len(L)), Dg)

    rows = []
    for name, k in (("poolmean", kappa), ("linemean", float("inf"))):
        t, w = score(mb.predict(L, Dg, pool, k)[0])
        for p in (1.0, 0.25):
            rows.append(dict(arm=name, run_id="", lr="", p=p, total_rmse=t, within_rmse=w))
    for rd in sorted((C.assets() / "runs" / "plumbing").iterdir()):
        if not (rd / "record.json").exists():
            rows.append(dict(arm="", run_id=rd.name, lr="", p="", note="no record: run failed (see RESULTS, failures)"))
            continue
        r = json.loads((rd / "record.json").read_text())
        if r.get("eval_role") != "inner_dev":
            continue
        pr = C.load_pred(rd / "pred.npz", L, Dg)
        for p in (1.0, 0.25):
            sc = [score(pr[k]) for k in C.PKEYS[p]]
            rows.append(dict(arm=r["arm"], run_id=r["run_id"], lr=r["lr"], p=p, total_rmse=np.mean([s[0] for s in sc]), within_rmse=np.mean([s[1] for s in sc]),
                             n_params=r["params"]["total"], updates=r["updates"], train_seconds=r["train_seconds"], peak_vram_MB=r["peak_vram_MB"],
                             distinct_views_used=r["distinct_views_used"], final_train_loss=r["curve"][-1]["train_supervised"]))
    out = pd.DataFrame(rows)
    for k, v in C.std_fields(split_role="source_inner_dev", fold=2, model_seed=42, n_drugs=len(dev), n_units=len(L)).items():
        out.insert(0, k, v)
    out["metric_definition"] = f"total: {DEF_TOTAL} | within: {DEF_WITHIN} (groups = pools, inner-dev drugs only); mean over thinning seeds"
    out.to_csv("reports/r6_followup_v1/plumbing_summary.tsv", sep="\t", index=False, float_format="%.6f")
    pd.set_option("display.width", 250)
    print(out.drop(columns=["metric_definition", "track", "version"]).to_string())


if __name__ == "__main__":
    main()
