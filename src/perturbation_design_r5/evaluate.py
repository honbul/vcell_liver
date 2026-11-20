"""Aggregation and contrasts. Episode -> AUC over K; mean over seeds -> family -> equal mean over families.
Source contrasts: family bootstrap. HCC contrasts: per line paired drug-group bootstrap (see hcc.py)."""
import json, os
import numpy as np
import pandas as pd

from .common import runs, public, dump_json

BOOT = dict(n=2000, seed=20261117)


def episode_table(scores):
    """One row per (role, cell, family, seed, policy, readout): AUC, K-wise errors."""
    out = []
    for key, g in scores.groupby(["role", "cell", "family", "seed", "policy", "readout"], sort=True):
        g = g.sort_values("K")
        rec = dict(zip(["role", "cell", "family", "seed", "policy", "readout"], key))
        ks, er = g.K.to_numpy(), g.rmse.to_numpy()
        rec["auc"] = float(np.trapz(er, ks) / (ks[-1] - ks[0])) if len(ks) > 1 else np.nan
        rec["ks"] = ",".join(map(str, ks))
        for k, e, p, cp in zip(ks, er, g.pearson, g.centred_pearson):
            rec[f"rmse_K{k}"] = e
            rec[f"pearson_K{k}"] = p
            rec[f"cpearson_K{k}"] = cp
        out.append(rec)
    return pd.DataFrame(out)


def family_table(ept, metric="auc"):
    """Mean over seeds per (family, policy, readout)."""
    return ept.groupby(["family", "policy", "readout"])[metric].mean().unstack("policy")


def summary(ept, metrics=("auc", "rmse_K4", "rmse_K8", "rmse_K16", "rmse_K32", "pearson_K16", "cpearson_K16")):
    rows = []
    for (ro, pol), g in ept.groupby(["readout", "policy"]):
        rec = dict(readout=ro, policy=pol, families=g.family.nunique(), episodes=len(g))
        for m in metrics:
            if m in g:
                rec[m] = g.groupby("family")[m].mean().mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def contrasts(ept, ref, metric="auc", readout="main", others=None):
    ft = family_table(ept[ept.readout == readout], metric).droplevel("readout")
    rs = np.random.RandomState(BOOT["seed"])
    fams = ft.index.to_numpy()
    idx = rs.randint(0, len(fams), size=(BOOT["n"], len(fams)))
    out = []
    for pol in (others or [c for c in ft.columns if c != ref]):
        if pol == ref or pol not in ft:
            continue
        d = (ft[pol] - ft[ref]).to_numpy()
        ok = ~np.isnan(d)
        dd = d[ok]
        bidx = rs.randint(0, len(dd), size=(BOOT["n"], len(dd)))
        bs = dd[bidx].mean(1)
        out.append(dict(readout=readout, metric=metric, policy=pol, reference=ref, diff=float(dd.mean()),
                        lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                        families_lower=int((dd < 0).sum()), families=int(len(dd))))
    return pd.DataFrame(out)


def summarize_run(tag, refs=("random", "diversity", "fixed_vr")):
    d = os.path.join(runs(), tag)
    sc = pd.read_csv(os.path.join(d, "scores.tsv"), sep="\t")
    ept = episode_table(sc)
    ept.to_csv(os.path.join(d, "episodes.tsv"), sep="\t", index=False)
    summ = summary(ept)
    con = []
    for ref in refs:
        if ref in set(ept.policy):
            for ro in ("main", "mean_offset"):
                for m in ("auc", "rmse_K16"):
                    con.append(contrasts(ept, ref, m, ro))
    con = pd.concat(con) if con else pd.DataFrame()
    pubd = public("manifests", "perturbation_design_r5", "runs", tag)
    os.makedirs(os.path.dirname(pubd + "/x"), exist_ok=True)
    summ.to_csv(os.path.join(pubd, "summary.tsv"), sep="\t", index=False, float_format="%.5f")
    con.to_csv(os.path.join(pubd, "contrasts.tsv"), sep="\t", index=False, float_format="%.5f")
    ft = family_table(ept[ept.readout == "main"], "auc").droplevel("readout")
    ft.to_csv(os.path.join(pubd, "family_auc_main.tsv"), sep="\t", float_format="%.5f")
    sq = pd.read_json(os.path.join(d, "sequences.jsonl"), lines=True)
    cost = sq.groupby("policy")[["trt_wells", "plates", "ctl_wells_overhead", "wall_s"]].mean()
    cost.to_csv(os.path.join(pubd, "cost_mean_per_episode.tsv"), sep="\t", float_format="%.3f")
    sq[["role", "cell", "family", "seed", "policy", "sequence"]].to_json(os.path.join(pubd, "sequences.jsonl"), orient="records", lines=True)
    info = json.load(open(os.path.join(d, "info.json")))
    dump_json(info, os.path.join(pubd, "info.json"))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        if len(con):
            print(con[(con.metric == "auc")].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return summ, con
