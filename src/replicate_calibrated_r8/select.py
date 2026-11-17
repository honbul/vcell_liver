"""Source-only model selection.

  cv     read the inner family cross-validation curves, average the fold scores per (arm, lr, step) and pick the (lr, steps)
         with the lowest mean K=16 family-equal delta RMSE (ties: fewer steps, smaller lr). Writes cv_curves.tsv, cv_selection.json.
  final  collect the fixed-step final runs into model_selection.json (checkpoint path + sha256 per arm and seed).
No confirmation-family or HCC value takes part in the choice.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

from .common import public, dump, runs, sha256, man_file

PUB = lambda n: public("manifests", "replicate_calibrated_r8", n)


def cv(tag):
    rows = []
    for rj in sorted(glob.glob(runs("cv", "*", "result.json"))):
        r = json.load(open(rj))
        if tag and not r["run_id"].endswith(tag):
            continue
        c = pd.read_csv(os.path.join(os.path.dirname(rj), "curve.tsv"), sep="\t")
        c.insert(0, "run_id", r["run_id"])
        c.insert(1, "arm", r["kind"][3:])
        c.insert(2, "lr", r["lr"])
        c.insert(3, "fold", r["fold"])
        c.insert(4, "seed", r["seed"])
        c["stopped"] = r["stopped"]
        rows.append(c)
    df = pd.concat(rows, ignore_index=True)
    nf = df.groupby(["arm", "lr", "step"]).fold.nunique().rename("n_folds")
    agg = df.groupby(["arm", "lr", "step"])[["val_k8", "val_k16", "val_k32", "train_loss", "lambda", "eta_k8", "eta_k16", "eta_k32"]].mean().join(nf).reset_index()
    full = agg[agg.n_folds == df.fold.nunique()]
    agg.to_csv(PUB("cv_curves.tsv"), sep="\t", index=False, float_format="%.6g")
    df[["run_id", "arm", "lr", "fold", "seed", "step", "train_loss", "val_k8", "val_k16", "val_k32", "lambda", "eta_k16", "stopped"]].to_csv(
        PUB("cv_fold_curves.tsv.gz"), sep="\t", index=False, float_format="%.6g")
    sel = {}
    for arm, g in full[full.step > 0].groupby("arm"):
        b = g.sort_values(["val_k16", "step", "lr"]).iloc[0]
        at0 = full[(full.arm == arm) & (full.step == 0)].val_k16.mean()
        per_lr = {f"{lr:g}": dict(best_step=int(h.sort_values(["val_k16", "step"]).iloc[0].step), best_k16=float(h.val_k16.min()), last_k16=float(h.sort_values("step").iloc[-1].val_k16))
                  for lr, h in g.groupby("lr")}
        folds = df[(df.arm == arm) & (df.lr == b.lr) & (df.step == b.step)].sort_values("fold")
        sel[arm] = dict(lr=float(b.lr), steps=int(b.step), cv_k8=float(b.val_k8), cv_k16=float(b.val_k16), cv_k32=float(b.val_k32), cv_k16_at_step0=float(at0),
                        fold_k16=[float(x) for x in folds.val_k16], max_step_seen=int(g.step.max()), best_is_last_step=bool(b.step == g.step.max()), per_lr=per_lr,
                        n_runs=int(df[df.arm == arm].run_id.nunique()), non_finite_runs=int(df[(df.arm == arm) & (df.stopped != "max_steps")].run_id.nunique()))
    dump(dict(rule="lowest mean-over-folds K=16 family-equal query delta RMSE over lr x step (every 200 steps); ties: fewer steps, then smaller lr",
              tag=tag, selection=sel, commit=os.environ.get("R8_COMMIT", "")), PUB("cv_selection.json"))
    print(json.dumps(sel, indent=1))


def final(tag):
    cvs = json.load(open(man_file("cv_selection.json")))["selection"]
    out, conf = {}, []
    for rj in sorted(glob.glob(runs("final", "*", "result.json"))):
        r = json.load(open(rj))
        if tag and not r["run_id"].endswith(tag):
            continue
        arm = r["arm"]
        assert r["stopped"] == "max_steps" and sha256(r["checkpoint"]) == r["checkpoint_sha256"]
        assert r["steps_run"] == cvs[arm]["steps"] and abs(r["lr"] - cvs[arm]["lr"]) < 1e-12, f"{r['run_id']} does not use the cross-validated setting"
        out.setdefault(arm, {})[str(r["seed"])] = dict(run_id=r["run_id"], checkpoint=r["checkpoint"], checkpoint_sha256=r["checkpoint_sha256"], lr=r["lr"], steps=r["steps_run"],
                                                       lam=r["lam"], eta=r["eta"], n_parameters=r["n_parameters"], commit=r["commit"])
        for k, v in r["confirmation_delta_rmse"].items():
            mode, K = k.split("|")
            conf.append(dict(arm=arm, seed=r["seed"], mode=mode, K=int(K), delta_rmse=v))
    pd.DataFrame(conf).sort_values(["K", "arm", "mode", "seed"]).to_csv(PUB("final_runs_confirmation.tsv"), sep="\t", index=False, float_format="%.6g")
    dump(dict(final=out, cv_selection=cvs, tag=tag, commit=os.environ.get("R8_COMMIT", "")), PUB("model_selection.json"))
    print(json.dumps({a: sorted(s) for a, s in out.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "final"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cv(a.tag) if a.mode == "cv" else final(a.tag)
