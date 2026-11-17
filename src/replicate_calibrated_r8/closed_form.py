"""Source-only selection of the shared anchor (kappa_f, kappa_b per K and observation model) and of the untrained residual
regression C1 (lambda, eta). Selection uses source-train dev episodes (leave-family-out f0); confirmation families are only reported."""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import yaml

from .common import CFG, public, dump, new_run_dir, run_record, man_file, proc
from .data import Store, Features
from . import methods as M
from src.episodic_residual_r2.evalcore import family_mean, paired_family


def rm(pred, ep):
    return torch.sqrt(((pred - ep.y_q) ** 2).mean(1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="closed_form_v1")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    t0 = time.time()
    out = new_run_dir("closed_form", a.run_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = Store(dev)
    fpath = proc("feature_basis.npz")
    if not os.path.exists(fpath):
        Features.fit(st).save(fpath)
    fmap = Features.load(fpath, dev)
    cov = json.load(open(man_file("covariance_model.json")))
    ec, ac, mc = cfg["episodes"], cfg["anchor"], cfg["models"]
    KF = [float(x) for x in ac["kappa_f_grid"]]
    KB = [float(x) for x in ac["kappa_b_grid"]]
    sets = dict(dev=st.episodes("train", ec["dev_support_seeds"], ec["dev_query_cap"]), confirmation=st.episodes("val", ec["confirmation_support_seeds"]))
    obs_models = dict(iid=None, cov=cov)
    base = lambda ep, K: dict(cell=ep.name, family=ep.family, support_seed=ep.seed, K=K)

    # ---- stage A: anchor grid
    rows, keffs = [], []
    with torch.no_grad():
        for sname, eps in sets.items():
            for ep in eps:
                for K in ec["k_values"]:
                    if K > len(ep.g_s):
                        continue
                    inp = ep.inputs(K)
                    for obs, om in obs_models.items():
                        for kf in KF:
                            anc = M.anchor(inp, kf, 0.0, om)
                            if kf == KF[0]:
                                keffs.append(dict(set=sname, obs=obs, **base(ep, K), k_eff=anc["keff"], max_weight=float(anc["w"].max()) * K,
                                                  cond=float(torch.linalg.cond(anc["Om"]))))
                            for kb in KB:
                                sh = 0.0 if np.isinf(kb) else anc["keff"] / (anc["keff"] + kb)
                                rows.append(dict(set=sname, obs=obs, kappa_f=kf, kappa_b=kb, delta_rmse=rm(anc["fq"] + anc["ebar"] * sh, ep), **base(ep, K)))
    df = pd.DataFrame(rows)
    grid = []
    for (sname, obs), g in df.groupby(["set", "obs"]):
        agg, _ = family_mean(g, by=("K", "kappa_f", "kappa_b"))
        agg["set"], agg["obs"] = sname, obs
        grid.append(agg)
    grid = pd.concat(grid)
    pub = lambda n: public("manifests", "replicate_calibrated_r8", n)
    grid.to_csv(pub("anchor_grid.tsv"), sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(keffs).groupby(["set", "obs", "K"]).agg(k_eff_mean=("k_eff", "mean"), k_eff_min=("k_eff", "min"), k_eff_max=("k_eff", "max"),
                                                          max_rel_weight=("max_weight", "max"), cond_max=("cond", "max"), n=("k_eff", "size")).reset_index().to_csv(
        pub("anchor_k_eff.tsv"), sep="\t", index=False, float_format="%.5g")
    look = lambda sname, obs, K, kf, kb: float(grid[(grid.set == sname) & (grid.obs == obs) & (grid.K == K) & (grid.kappa_f == kf) & (grid.kappa_b == kb)].delta_rmse.iloc[0])
    sel = dict(anchor=dict(iid={}, cov={}), shrunk_same_drug_mean={}, c1=dict(iid={}, cov={}), covariance_model=cov)
    for K in ec["k_values"]:
        d = grid[(grid.set == "dev") & (grid.obs == "iid") & (grid.K == K)]
        b = d.sort_values(["delta_rmse", "kappa_f", "kappa_b"]).iloc[0]
        sel["anchor"]["iid"][str(K)] = dict(kappa_f=float(b.kappa_f), kappa_b=float(b.kappa_b) if np.isfinite(b.kappa_b) else "inf", dev=float(b.delta_rmse),
                                            at_grid_edge=bool(b.kappa_f in (KF[0], KF[-1]) or b.kappa_b in (KB[0], KB[-1])))
        dc = grid[(grid.set == "dev") & (grid.obs == "cov") & (grid.K == K) & (grid.kappa_f == b.kappa_f)]
        bc = dc.sort_values(["delta_rmse", "kappa_b"]).iloc[0]
        sel["anchor"]["cov"][str(K)] = dict(kappa_f=float(b.kappa_f), kappa_b=float(bc.kappa_b) if np.isfinite(bc.kappa_b) else "inf", dev=float(bc.delta_rmse),
                                            at_grid_edge=bool(bc.kappa_b in (KB[0], KB[-1])))
        bf = d[np.isinf(d.kappa_b)].sort_values(["delta_rmse", "kappa_f"]).iloc[0]
        sel["shrunk_same_drug_mean"][str(K)] = dict(kappa_f=float(bf.kappa_f), dev=float(bf.delta_rmse))

    # ---- stage B: C1 (lambda, eta) on the selected anchor; then the report rows of both sets
    lams, etas = [float(x) for x in mc["c1_lambda_grid"]], [float(x) for x in mc["c1_eta_grid"]]
    rows2, rep = [], []
    with torch.no_grad():
        for sname, eps in sets.items():
            for ep in eps:
                for K in ec["k_values"]:
                    if K > len(ep.g_s):
                        continue
                    inp = ep.inputs(K)
                    rep.append(dict(set=sname, method="no_change", delta_rmse=rm(M.no_change(inp), ep), **base(ep, K)))
                    rep.append(dict(set=sname, method="same_drug_mean", delta_rmse=rm(M.same_drug_mean(inp), ep), **base(ep, K)))
                    rep.append(dict(set=sname, method="mean_offset", delta_rmse=rm(M.mean_offset(inp), ep), **base(ep, K)))
                    rep.append(dict(set=sname, method="shrunk_same_drug_mean", delta_rmse=rm(M.anchor(inp, sel["shrunk_same_drug_mean"][str(K)]["kappa_f"], "inf")["fq"], ep), **base(ep, K)))
                    for obs, om in obs_models.items():
                        e = sel["anchor"][obs][str(K)]
                        anc = M.anchor(inp, e["kappa_f"], e["kappa_b"], om)
                        rep.append(dict(set=sname, method=f"C0_{obs}", delta_rmse=rm(M.c0(anc), ep), **base(ep, K)))
                        ft = fmap.build(inp, st.FP, anc["fs"], anc["fq"])
                        for lam in lams:
                            corr = M.krr_correction(anc, ft, lam)
                            for eta in etas:
                                rows2.append(dict(set=sname, obs=obs, lam=lam, eta=eta, delta_rmse=rm(anc["fq"] + anc["b"] + eta * corr, ep), **base(ep, K)))
    d2 = pd.DataFrame(rows2)
    g2 = []
    for (sname, obs), g in d2.groupby(["set", "obs"]):
        agg, _ = family_mean(g, by=("K", "lam", "eta"))
        agg["set"], agg["obs"] = sname, obs
        g2.append(agg)
    g2 = pd.concat(g2)
    g2.to_csv(pub("c1_grid.tsv"), sep="\t", index=False, float_format="%.6g")
    for obs in obs_models:
        for K in ec["k_values"]:
            d = g2[(g2.set == "dev") & (g2.obs == obs) & (g2.K == K)]
            b = d.sort_values(["delta_rmse", "eta", "lam"], ascending=[True, True, False]).iloc[0]      # ties -> weaker correction
            sel["c1"][obs][str(K)] = dict(lam=float(b.lam), eta=float(b.eta), dev=float(b.delta_rmse), at_grid_edge=bool(b.lam in (lams[0], lams[-1])))
            m = (d2.obs == obs) & (d2.K == K) & np.isclose(d2.lam, b.lam) & np.isclose(d2.eta, b.eta)
            t = d2[m].copy()
            t["method"] = f"C1_{obs}"
            rep.append(t[["set", "method", "delta_rmse", "cell", "family", "support_seed", "K"]])
    rep = pd.concat([pd.DataFrame([r for r in rep if isinstance(r, dict)])] + [r for r in rep if not isinstance(r, dict)], ignore_index=True)
    rep.to_csv(pub("closed_form_episodes.tsv"), sep="\t", index=False, float_format="%.6g")
    summ, cons = [], []
    for sname, g in rep.groupby("set"):
        agg, fam = family_mean(g)
        agg["set"] = sname
        summ.append(agg)
        for x, y in (("C1_iid", "C0_iid"), ("C1_cov", "C0_cov"), ("C0_cov", "C0_iid"), ("C1_cov", "C1_iid"), ("C0_iid", "mean_offset"), ("C0_iid", "same_drug_mean"),
                     ("C0_iid", "shrunk_same_drug_mean")):
            c = paired_family(fam, x, y)
            c["set"] = sname
            cons.append(c)
    summ, cons = pd.concat(summ), pd.concat(cons)
    summ.to_csv(pub("closed_form_summary.tsv"), sep="\t", index=False, float_format="%.6g")
    cons.to_csv(pub("closed_form_contrasts.tsv"), sep="\t", index=False, float_format="%.6g")
    sel.update(rule="per K, lowest family-equal query delta RMSE on source-train dev episodes; no confirmation family, no HCC value", run_id=a.run_id,
               commit=os.environ.get("R8_COMMIT", ""))
    dump(sel, pub("anchor_selection.json"))
    dump(run_record(a.run_id, "closed_form", t0, n_dev_episodes=len(sets["dev"]), n_confirmation_episodes=len(sets["confirmation"]),
                    dev_families=int(rep[rep.set == "dev"].family.nunique()), confirmation_families=int(rep[rep.set == "confirmation"].family.nunique()),
                    selected={k: sel[k] for k in ("anchor", "shrunk_same_drug_mean", "c1")}), os.path.join(out, "result.json"))
    pd.set_option("display.width", 250, "display.max_rows", 400)
    print(json.dumps({k: sel[k] for k in ("anchor", "shrunk_same_drug_mean", "c1")}, indent=1))
    print(summ.pivot_table(index=["set", "method"], columns="K", values="delta_rmse").to_string())
    print(cons.to_string())


if __name__ == "__main__":
    main()
