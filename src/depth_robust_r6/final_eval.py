"""Predict the evaluation drugs of a setting from finished checkpoints (no fitting, no selection).

In the final setting this is the only place a neural arm meets the 70 held-out drugs, and it
refuses to run unless the freeze manifest is committed and lists the checkpoint's configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .evaluate import cond_rmse, drug_macro
from .train import build_model, predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=["dev", "final"])
    ap.add_argument("--runs", nargs="+", required=True, help="run ids under $R6_ASSETS/runs/<setting>/")
    a = ap.parse_args()
    root = Path(os.environ["R6_ASSETS"]) / "runs" / a.setting
    if a.setting == "final":
        fm = Path("manifests/depth_robust_r6/freeze_manifest.json")
        assert fm.exists(), "freeze before the held-out evaluation"
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(fm)], capture_output=True).returncode == 0
        clean = subprocess.run(["git", "status", "--porcelain", "--", "src/depth_robust_r6", "configs/depth_robust_r6", str(fm)],
                               capture_output=True, text=True).stdout.strip() == ""
        assert tracked and clean, "the freeze manifest and the code must be committed and unmodified"
        frozen = json.loads(fm.read_text())
        allowed = {r["run_id"] for r in frozen["final_runs"]}
        assert set(a.runs) <= allowed, sorted(set(a.runs) - allowed)
    dev = "cuda"
    d = D.load()
    _, eval_drugs = d.split_spec(a.setting)
    for rid in a.runs:
        run = root / rid
        out = run / f"pred_{a.setting}.npz"
        assert not out.exists(), f"{out} exists; predictions are written once"
        ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
        assert ck["setting"] == a.setting and not (set(ck["fit_drugs"]) & set(eval_drugs))
        model = build_model(d, ck["arm"], ck["hp"]).to(dev)
        model.load_state_dict(ck["state_dict"])
        L, Dg, pr = predict(model, d, eval_drugs, dev, ck["y_scale"])
        np.savez_compressed(out, line=L, drug=Dg, **{f"p{p}_s{s}": v.astype(np.float16) for (p, s), v in pr.items()})
        T = d.Y[L, Dg]
        rec = dict(run_id=rid, arm=ck["arm"], setting=a.setting, ckpt_sha256=D.sha256_file(run / "ckpt.pt"),
                   pred_sha256=D.sha256_file(out), n_conditions=int(len(L)),
                   scores={f"p{p}_s{s}": drug_macro(cond_rmse(v, T), Dg) for (p, s), v in pr.items()},
                   commit=os.popen("git rev-parse HEAD").read().strip(), job=os.environ.get("SLURM_JOB_ID"))
        D.dump_json(rec, run / f"eval_{a.setting}.json")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
