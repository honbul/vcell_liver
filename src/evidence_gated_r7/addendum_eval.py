"""Addendum A-2 evaluation (input-matched XPert). Runs once after the addendum freeze file is committed.
Reuses the frozen test predictions of job 25976 (sha256-checked) and scores the new XPert-ECFP predictions with the same
metrics and drug-group bootstrap. R7 models and verdicts are not re-run or changed."""
import json, os, subprocess, time
import numpy as np
import pandas as pd

from .common import MAN, runs, pman, dump, sha256
from .data import Store, Knowledge
from .evaluate import unit_metrics, summarise, paired_bootstrap

PANELS = ["source", "HUH7", "JHH5", "JHH7", "hcc_mean"]
ADD = os.path.join(MAN, "addendum_freeze.json")


def main():
    t0 = time.time()
    assert os.environ.get("R7_FINAL_EVAL") == "1"
    assert subprocess.run(["git", "ls-files", "--error-unmatch", ADD], capture_output=True).returncode == 0, "addendum freeze not committed"
    fr = json.load(open(ADD))
    S = Store(); K = Knowledge(len(S.groups))
    te = S.test
    y = S.test_delta(te)
    ut = S.units.iloc[te].reset_index(drop=True)
    ut["status"] = K.status.set_index("group_idx").status.reindex(ut.group_idx).to_numpy()
    cells = ut.cell.to_numpy()
    rec0 = json.load(open(os.path.join(MAN, "test_eval_record.json")))
    pdir = rec0["prediction_dir"]
    preds = {}
    for name in fr["reused_models"]:
        ks = sorted(int(f[len(name) + 1:-4]) for f in rec0["prediction_sha256"] if f.startswith(name + "_") and f[len(name) + 1:-4].isdigit())
        preds[name] = []
        for k in ks:
            f = os.path.join(pdir, f"{name}_{k}.npy")
            assert sha256(f) == rec0["prediction_sha256"][f"{name}_{k}.npy"], f
            preds[name].append(np.load(f))
    preds["xpert_ecfp"] = []
    for x in fr["xpert_ecfp"]:
        r = json.load(open(pman("runs", f"xpert_{x['run_id']}_pred_test.json")))
        assert r["checkpoint_sha256"] == x["checkpoint_sha256"] and r["drug_feat"] == "morgan"
        f = runs("xpert", x["run_id"], "pred_test.npy")
        assert sha256(f) == r["pred_sha256"]
        preds["xpert_ecfp"].append(np.load(f))
    rows, unit_rmse = [], {}
    for name, ps in preds.items():
        per = []
        for k, p in enumerate(ps):
            m = unit_metrics(p, y, cells); per.append(m.rmse.to_numpy())
            for pan in PANELS:
                rows.append(dict(model=name, seed_index=k, panel=pan, **summarise(m, ut, pan)))
        unit_rmse[name] = np.mean(per, 0)
    tab = pd.DataFrame(rows)
    num = [c for c in tab.columns if c not in ("model", "seed_index", "panel")]
    summ = tab.groupby(["model", "panel"])[num].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(pman("addendum_test_summary.tsv"), sep="\t", index=False)
    crow = []
    for a, b in fr["contrasts"]:
        for pan in PANELS:
            crow.append(dict(a=a, b=b, panel=pan, **paired_bootstrap(unit_rmse[a], unit_rmse[b], ut, pan)))
        for lev in ["complete", "partial", "missing"]:
            s = (ut.status == lev).to_numpy()
            crow.append(dict(a=a, b=b, panel=f"source_{lev}", **paired_bootstrap(unit_rmse[a][s], unit_rmse[b][s], ut[s].reset_index(drop=True), "source")))
    pd.DataFrame(crow).to_csv(pman("addendum_test_contrasts.tsv"), sep="\t", index=False)
    dump(dict(job=os.environ.get("SLURM_JOB_ID"), commit=os.environ.get("R7_COMMIT"), seconds=round(time.time() - t0),
              reused_prediction_dir=pdir, reused_models=fr["reused_models"]), pman("addendum_eval_record.json"))
    print(summ.reset_index()[["model", "panel", "rmse_mean", "rmse_std", "centred_pearson_mean", "mrr_mean"]].to_string())
    print(pd.DataFrame(crow).round(5).to_string())


if __name__ == "__main__":
    main()
