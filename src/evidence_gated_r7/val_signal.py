"""Validation signal (pre-registered rule) on the source panel only: 3-seed means, paired drug-group bootstrap on the
seed-averaged unit RMSE, annotation-status strata and the shrinkage check. No HCC unit is scored."""
import argparse, json
import numpy as np
import pandas as pd

from .common import runs, pman, dump
from .data import Store, Knowledge
from .evaluate import unit_metrics, summarise, paired_bootstrap, family_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help='JSON {"name": ["kind/run_id", ...]}')
    ap.add_argument("--out", default="val_signal")
    a = ap.parse_args()
    models = json.loads(a.models)
    S = Store(); K = Knowledge(len(S.groups))
    val = S.val
    src = S.source(val)
    v = val[src]
    y = S.delta(v)
    uv = S.units.iloc[v].reset_index(drop=True)
    st = K.status.set_index("group_idx")
    uv["status"] = st.status.reindex(uv.group_idx).to_numpy()
    cells = uv.cell.to_numpy(); fam = uv.family.to_numpy()
    unit_rmse, seed_rows = {}, []
    for name, lst in models.items():
        per = []
        for kr in lst:
            kind, rid = kr.split("/")
            assert (np.load(runs(kind, rid, "val_idx.npy")) == val).all()
            p = np.load(runs(kind, rid, "val_pred.npy"))[src]
            rec = json.load(open(runs(kind, rid, "record.json")))
            m = unit_metrics(p, y, cells); per.append(m.rmse.to_numpy())
            s = summarise(m, uv, "source")
            seed_rows.append(dict(model=name, run=rid, seed=rec["config"]["seed"], rmse=s["rmse"], centred_pearson=s["centred_pearson"], mrr=s["mrr"],
                                  corr_rms=rec["best"]["val"].get("corr_rms"), chem_rms=rec["best"]["val"].get("chem_rms"), best_step=rec["best"]["step"],
                                  **{f"rmse_{lev}": family_mean(m.rmse.to_numpy()[uv.status == lev], fam[uv.status == lev]) for lev in ["complete", "partial", "missing"]}))
        unit_rmse[name] = np.mean(per, 0)
    sr = pd.DataFrame(seed_rows)
    sr.to_csv(pman(f"{a.out}_by_seed.tsv"), sep="\t", index=False)
    mean = sr.groupby("model")[["rmse", "centred_pearson", "mrr", "corr_rms", "chem_rms", "rmse_complete", "rmse_partial", "rmse_missing"]].mean()
    mean["n_seeds"] = sr.groupby("model").size()
    mean.reset_index().to_csv(pman(f"{a.out}_mean.tsv"), sep="\t", index=False)
    crow = []
    names = list(models)
    for x in names:
        for b in names:
            if x == b:
                continue
            c = paired_bootstrap(unit_rmse[x], unit_rmse[b], uv, "source")
            crow.append(dict(a=x, b=b, stratum="all", **c))
            for lev in ["complete", "partial", "missing"]:
                s = (uv.status == lev).to_numpy()
                crow.append(dict(a=x, b=b, stratum=lev, **paired_bootstrap(unit_rmse[x][s], unit_rmse[b][s], uv[s].reset_index(drop=True), "source")))
    ct = pd.DataFrame(crow)
    ct.to_csv(pman(f"{a.out}_contrasts.tsv"), sep="\t", index=False)
    res = dict(mean=mean.to_dict(orient="index"))
    if {"B11", "chem_only", "B01", "B10", "fixed_graph"} <= set(names):
        g = lambda n: float(mean.loc[n, "rmse"])
        c = ct[(ct.a == "B11") & (ct.b == "chem_only") & (ct.stratum == "all")].iloc[0]
        cond = dict(b11_minus_chem_le_m0p001=g("B11") - g("chem_only") <= -0.001, below_B01=g("B11") < g("B01"), below_B10=g("B11") < g("B10"),
                    below_fixed=g("B11") < g("fixed_graph"), ci_b11_chem_below0=bool(c.hi < 0))
        res.update(conditions=cond, validation_signal=bool(all(cond.values())), b11_minus_chem=dict(diff=float(c["diff"]), lo=float(c.lo), hi=float(c.hi)),
                   shrinkage_check=dict(b11_corr_rms=float(mean.loc["B11", "corr_rms"]), b11_chem_rms=float(mean.loc["B11", "chem_rms"]),
                                        ratio=float(mean.loc["B11", "corr_rms"] / mean.loc["B11", "chem_rms"])))
    dump(res, pman(f"{a.out}.json"))
    print(mean.round(5).to_string())
    print(ct[ct.stratum == "all"].round(5).to_string())
    print(json.dumps({k: v for k, v in res.items() if k != "mean"}, indent=1))


if __name__ == "__main__":
    main()
