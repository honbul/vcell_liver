"""One-time test evaluation. Refuses unless the freeze manifest is committed and the worktree is clean."""
import json, os, subprocess, time
import numpy as np
import pandas as pd
import torch

from .common import MAN, runs, public, dump, sha256, HCC
from .data import Store, TrainStats, FREEZE
from .evaluate import unit_metrics, summarise, paired_bootstrap
from .baselines import fit_all
from .models import W0Native, W1DeltaAnchor, DeltaMLP
from .train import predict

PANELS = ["source", "HUH7", "JHH5", "JHH7", "hcc_mean"]


def main():
    t0 = time.time()
    assert os.environ.get("R4_FINAL_EVAL") == "1"
    assert subprocess.run(["git", "ls-files", "--error-unmatch", FREEZE], capture_output=True).returncode == 0, "freeze manifest not committed"
    assert subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip() == "", "worktree not clean"
    fr = json.load(open(FREEZE))
    out = runs("test_eval", "test_eval_v1")
    os.makedirs(out, exist_ok=False)
    S = Store(); st = TrainStats(S)
    te = S.test
    y = S.test_delta(te)
    ut = S.units.iloc[te].reset_index(drop=True)
    cells = ut.cell.to_numpy()
    dev = torch.device("cuda")
    preds = {}
    bl = fit_all(S, st)
    sel = fr["baseline_selection"]["selected"]
    preds["no_change"] = [bl["no_change"](te)]
    preds["train_cell_mean"] = [bl["train_cell_mean"](te)]
    preds["ecfp_x0_ridge"] = [bl[sel["ecfp_x0_ridge"]["model"]](te)]
    preds["cellmean_ecfp_ridge"] = [bl[sel["cellmean_ecfp_ridge"]["model"]](te)]
    for name, lst in fr["models"].items():
        preds[name] = []
        for r in lst:
            assert sha256(r["checkpoint"]) == r["checkpoint_sha256"], r["run_id"]
            m = {"w0": W0Native, "mlp": DeltaMLP}.get(r["arm"], W1DeltaAnchor)().to(dev)
            m.load_state_dict(torch.load(r["checkpoint"], map_location=dev))
            preds[name].append(predict(m, S, te, dev))
    hashes, rows, unit_rmse = {}, [], {}
    for name, ps in preds.items():
        per_seed = []
        for k, p in enumerate(ps):
            f = os.path.join(out, f"{name}_{k}.npy"); np.save(f, p.astype(np.float32)); hashes[f"{name}_{k}.npy"] = sha256(f)
            met = unit_metrics(p, y, cells)
            per_seed.append(met)
            for pan in PANELS:
                rows.append(dict(model=name, seed_index=k, panel=pan, **summarise(met, ut, pan)))
        unit_rmse[name] = np.mean([m.rmse.to_numpy() for m in per_seed], 0)
    tab = pd.DataFrame(rows)
    tab.to_csv(public(MAN, "test_by_model_seed.tsv"), sep="\t", index=False)
    num = [c for c in tab.columns if c not in ("model", "seed_index", "panel")]
    summ = tab.groupby(["model", "panel"])[num].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(public(MAN, "test_summary.tsv"), sep="\t", index=False)
    # strength tertiles of the true test response (scoring only)
    terc = pd.qcut(np.sqrt((y ** 2).mean(1)), 3, labels=["weak", "mid", "strong"]).astype(str)
    trows = []
    for name, ur in unit_rmse.items():
        for t in ["weak", "mid", "strong"]:
            s = (terc == t) & (ut.is_hcc.to_numpy() == 0)
            from .evaluate import family_mean
            trows.append(dict(model=name, tertile=t, rmse=family_mean(ur[s], ut.family.to_numpy()[s]), n_units=int(s.sum())))
    pd.DataFrame(trows).to_csv(public(MAN, "test_by_strength.tsv"), sep="\t", index=False)
    crow = []
    for a, b in fr["contrasts"]:
        for pan in PANELS:
            c = paired_bootstrap(unit_rmse[a], unit_rmse[b], ut, pan)
            crow.append(dict(a=a, b=b, panel=pan, **c))
    # per-seed paired contrasts for the W2 claim rule
    for a, b in fr["contrasts"]:
        if len(preds[a]) == len(preds[b]) > 1:
            for k in range(len(preds[a])):
                ra = tab[(tab.model == a) & (tab.seed_index == k) & (tab.panel == "source")].rmse.iloc[0]
                rb = tab[(tab.model == b) & (tab.seed_index == k) & (tab.panel == "source")].rmse.iloc[0]
                crow.append(dict(a=a, b=b, panel=f"source_seed{k}", diff=ra - rb, lo=np.nan, hi=np.nan))
    pd.DataFrame(crow).to_csv(public(MAN, "test_contrasts.tsv"), sep="\t", index=False)
    dump(dict(job=os.environ.get("SLURM_JOB_ID"), commit=os.environ.get("R4_COMMIT"), seconds=round(time.time() - t0), n_test_units=int(len(te)),
              prediction_sha256=hashes), public(MAN, "test_eval_record.json"))
    print(summ.reset_index()[["model", "panel", "rmse_mean", "rmse_std"]].to_string())


if __name__ == "__main__":
    main()
