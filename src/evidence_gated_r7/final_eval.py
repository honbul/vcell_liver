"""One-time R7 test evaluation. Refuses unless the freeze manifest is committed and the worktree is clean (Store.test_delta).
With --mock it runs the identical code on VALIDATION units with the uncommitted freeze draft (wiring check; test sealed)."""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch

from .common import MAN, runs, pman, public, dump, sha256, HCC
from .data import Store, TrainStats, Knowledge
from .evaluate import unit_metrics, summarise, paired_bootstrap, family_mean
from .models import R7Model
from .train import SeedBank, load_graph, tensors, predict
from src.response_geometry_r4.baselines import fit_all

PANELS = ["source", "HUH7", "JHH5", "JHH7", "hcc_mean"]
MASK_SEED = 20261117


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    S = Store(); st = TrainStats(S); K = Knowledge(len(S.groups))
    if a.mock:
        assert os.environ.get("R7_FINAL_EVAL") != "1"
        fr = json.load(open(pman("freeze_manifest.json")))
        out = runs("test_eval", f"mock_{int(t0)}"); tag = "mock"
        te = S.val; y = S.delta(te)
        dst = lambda *p: pman("mock", *p)
    else:
        assert os.environ.get("R7_FINAL_EVAL") == "1"
        fr = json.load(open(os.path.join(MAN, "freeze_manifest.json")))
        out = runs("test_eval", "test_eval_v1"); tag = "test"
        te = S.test; y = S.test_delta(te)
        dst = lambda *p: pman(*p)
    os.makedirs(out, exist_ok=False)
    ut = S.units.iloc[te].reset_index(drop=True)
    gst = K.status.set_index("group_idx")
    ut["status"] = gst.status.reindex(ut.group_idx).to_numpy()
    ut["n_target_genes"] = gst.n_target_genes.reindex(ut.group_idx).to_numpy()
    ut["m_available"] = gst.m_available.reindex(ut.group_idx).to_numpy()
    cells = ut.cell.to_numpy()
    dev = torch.device("cuda")
    T = tensors(S, st, dev)
    preds, stress = {}, {}
    bl = fit_all(S, st)
    sel = fr["baseline_selection"]["selected"]
    preds["no_change"] = [bl["no_change"](te)]
    preds["train_cell_mean"] = [bl["train_cell_mean"](te)]
    preds["ecfp_x0_ridge"] = [bl[sel["ecfp_x0_ridge"]["model"]](te)]
    preds["cellmean_ecfp_ridge"] = [bl[sel["cellmean_ecfp_ridge"]["model"]](te)]
    corr_rms = {}
    for name, lst in fr["models"].items():
        preds[name] = []; stress[name] = {0.5: [], 1.0: []}; corr_rms[name] = []
        for r in lst:
            assert sha256(r["checkpoint"]) == r["checkpoint_sha256"], r["run_id"]
            c = r["config"]
            arm = c["arm"]
            bank = SeedBank(K, c.get("evidence", "on")) if arm != "chem" else None
            m = R7Model(arm, load_graph(K, c.get("graph", "real")) if arm != "chem" else None).to(dev)
            m.load_state_dict(torch.load(r["checkpoint"], map_location=dev))
            p, rr, _ = predict(m, T, bank, te, dev, parts=True)
            preds[name].append(p); corr_rms[name].append(float(np.sqrt((rr ** 2).mean())))
            if name in fr["stress_models"]:
                for fr_ in (0.5, 1.0):
                    stress[name][fr_].append(predict(m, T, bank, te, dev, mask_frac=fr_, mask_seed=MASK_SEED))
    for x in fr["xpert"]:
        rec = json.load(open(pman("runs", f"xpert_{x['run_id']}_pred_{'validation' if a.mock else 'test'}.json")))
        assert rec["checkpoint_sha256"] == x["checkpoint_sha256"]
        p = np.load(runs("xpert", x["run_id"], f"pred_{'validation' if a.mock else 'test'}.npy"))
        assert sha256(runs("xpert", x["run_id"], f"pred_{'validation' if a.mock else 'test'}.npy")) == rec["pred_sha256"]
        preds.setdefault("xpert", []).append(p)
    hashes, rows, unit_rmse, unit_cp = {}, [], {}, {}
    for name, ps in preds.items():
        per = []
        for k, p in enumerate(ps):
            f = os.path.join(out, f"{name}_{k}.npy"); np.save(f, p.astype(np.float32)); hashes[f"{name}_{k}.npy"] = sha256(f)
            met = unit_metrics(p, y, cells); per.append(met)
            for pan in PANELS:
                rows.append(dict(model=name, seed_index=k, panel=pan, **summarise(met, ut, pan)))
        unit_rmse[name] = np.mean([m.rmse.to_numpy() for m in per], 0)
        unit_cp[name] = np.mean([m.centred_pearson.to_numpy() for m in per], 0)
    tab = pd.DataFrame(rows)
    tab.to_csv(dst("test_by_model_seed.tsv"), sep="\t", index=False)
    num = [c for c in tab.columns if c not in ("model", "seed_index", "panel")]
    summ = tab.groupby(["model", "panel"])[num].agg(["mean", "std"])
    summ.columns = [f"{x}_{s}" for x, s in summ.columns]
    summ.reset_index().to_csv(dst("test_summary.tsv"), sep="\t", index=False)
    src = ut.is_hcc.to_numpy() == 0
    fam = ut.family.to_numpy()
    # strata (source panel): annotation status, target-count tertiles among annotated, true-RMS tertiles
    ann = ut.m_available.to_numpy() == 1
    tert = np.full(len(ut), "none", object)
    tert[ann] = pd.qcut(ut.n_target_genes.to_numpy()[ann].astype(float) + 1e-6 * np.arange(ann.sum()), 3, labels=["few", "mid", "many"]).astype(str)
    strength = pd.qcut(np.sqrt((y ** 2).mean(1)), 3, labels=["weak", "mid", "strong"]).astype(str)
    srows = []
    for name, ur in unit_rmse.items():
        for col, vals in [("status", ut.status.to_numpy()), ("target_count", tert), ("strength", strength)]:
            for v in sorted(set(vals)):
                s = src & (vals == v)
                if s.any():
                    srows.append(dict(model=name, stratum=col, level=v, rmse=family_mean(ur[s], fam[s]),
                                      centred_pearson=family_mean(unit_cp[name][s], fam[s]), n_units=int(s.sum()), n_groups=int(ut.drug_group[s].nunique())))
    pd.DataFrame(srows).to_csv(dst("test_by_stratum.tsv"), sep="\t", index=False)
    # annotation stress on annotated units: 100 / 50 / 0 % of the annotation kept
    strow = []
    sub = ut[ann].reset_index(drop=True)
    for name in fr["stress_models"]:
        levels = {"kept_100": preds[name], "kept_50": stress[name][0.5], "kept_0": stress[name][1.0]}
        for lev, ps in levels.items():
            ur = np.mean([unit_metrics(p, y, cells).rmse.to_numpy() for p in ps], 0)
            for pan, s in [("source", src & ann)] + [(h, (cells == h) & ann) for h in HCC]:
                if s.any():
                    v = family_mean(ur[s], fam[s]) if pan == "source" else float(ur[s].mean())
                    strow.append(dict(model=name, annotation=lev, panel=pan, rmse=v, n_units=int(s.sum())))
            if lev != "kept_100":
                c = paired_bootstrap(ur[ann], unit_rmse[name][ann], sub, "source")
                strow.append(dict(model=name, annotation=f"{lev}_minus_kept_100", panel="source", rmse=c["diff"], lo=c["lo"], hi=c["hi"],
                                  n_units=int((src & ann).sum())))
    pd.DataFrame(strow).to_csv(dst("annotation_stress.tsv"), sep="\t", index=False)
    # contrasts (seed-averaged unit RMSE, paired drug-group bootstrap)
    crow = []
    for x, b in fr["contrasts"]:
        if x not in unit_rmse or b not in unit_rmse:
            crow.append(dict(a=x, b=b, panel="source", diff=np.nan, lo=np.nan, hi=np.nan, note="model missing")); continue
        for pan in PANELS:
            crow.append(dict(a=x, b=b, panel=pan, **paired_bootstrap(unit_rmse[x], unit_rmse[b], ut, pan)))
        for lev in ["complete", "partial", "missing"]:
            s = ut.status.to_numpy() == lev
            sub = ut[s].reset_index(drop=True)
            if (sub.is_hcc == 0).sum() > 0:
                crow.append(dict(a=x, b=b, panel=f"source_{lev}", **paired_bootstrap(unit_rmse[x][s], unit_rmse[b][s], sub, "source")))
        if len(preds[x]) == len(preds[b]) > 1:
            for k in range(len(preds[x])):
                ra = tab[(tab.model == x) & (tab.seed_index == k) & (tab.panel == "source")].rmse.iloc[0]
                rb = tab[(tab.model == b) & (tab.seed_index == k) & (tab.panel == "source")].rmse.iloc[0]
                crow.append(dict(a=x, b=b, panel=f"source_seed{k}", diff=ra - rb, lo=np.nan, hi=np.nan))
    pd.DataFrame(crow).to_csv(dst("test_contrasts.tsv"), sep="\t", index=False)
    pd.DataFrame(dict(unit_idx=te, drug_group=ut.drug_group, cell=ut.cell, status=ut.status,
                      **{f"rmse_{k}": np.round(v, 5) for k, v in unit_rmse.items()})).to_csv(dst("test_unit_rmse.tsv.gz"), sep="\t", index=False)
    dump(dict(job=os.environ.get("SLURM_JOB_ID"), commit=os.environ.get("R7_COMMIT"), seconds=round(time.time() - t0), n_units=int(len(te)), tag=tag,
              correction_rms=corr_rms, prediction_dir=out, prediction_sha256=hashes), dst("test_eval_record.json"))
    print(summ.reset_index()[["model", "panel", "rmse_mean", "rmse_std"]].to_string())


if __name__ == "__main__":
    main()
