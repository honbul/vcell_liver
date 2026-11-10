#!/usr/bin/env python
"""G4 task 1: fetch the official State ST-HVG-Replogle *zeroshot* assets (pinned HF revision).

Per held-out line (hepg2, jurkat, k562, rpe1): config/maps/var_dims/hparams/metrics, checkpoints/best.ckpt,
checkpoints/final.ckpt (step 100k, no val selection;
best.ckpt is selected on val_loss, which for these zeroshot TOMLs falls back to the held-out line's
test cells), and the authors' own zeroshot eval outputs eval_best.ckpt/{adata_real,adata_pred}.h5ad + result CSVs.
The adata files are used only to (i) verify our preprocessing of the raw counts reproduces State's
X_hvg input space and (ii) check our inference reproduces the authors' predictions. Their DE CSVs
(~0.3 GB each) are not fetched. No token; telemetry off. Writes a provenance JSON with sizes + sha256.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "arcinstitute/ST-HVG-Replogle"
REV = "bb6a9562cbbf1fd152df14cc53b4cc7517c77175"
LINES = ["hepg2", "jurkat", "k562", "rpe1"]
SMALL = ["config.yaml", "data_module.torch", "batch_onehot_map.pkl", "cell_type_onehot_map.pkl",
         "pert_onehot_map.pt", "var_dims.pkl", "version_0/hparams.yaml", "version_0/metrics.csv",
         "wandb_path.txt"]


def sha256(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/g4/state/hf/ST-HVG-Replogle")
    ap.add_argument("--prov", default="reports/feedback/G4/provenance_state_fetch.json")
    ap.add_argument("--lines", nargs="*", default=LINES)
    ap.add_argument("--no-adata", action="store_true")
    a = ap.parse_args()
    info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
    meta = {s.rfilename: s for s in info.siblings}
    files = ["MODEL_LICENSE.md", "MODEL_ACCEPTABLE_USE_POLICY.md"]
    for L in a.lines:
        files += [f"zeroshot/{L}/{f}" for f in SMALL] + [f"zeroshot/{L}/checkpoints/best.ckpt", f"zeroshot/{L}/checkpoints/final.ckpt"]
        files += [f"zeroshot/{L}/eval_best.ckpt/{L}_results.csv", f"zeroshot/{L}/eval_best.ckpt/{L}_agg_results.csv"]
        if not a.no_adata:
            files += [f"zeroshot/{L}/eval_best.ckpt/adata_real.h5ad", f"zeroshot/{L}/eval_best.ckpt/adata_pred.h5ad"]
    rec, t0 = [], time.time()
    for f in files:
        p = hf_hub_download(REPO, f, revision=REV, local_dir=a.out)
        s = meta[f]
        d = dict(file=f, bytes=Path(p).stat().st_size, hf_size=s.size)
        if s.lfs is not None:
            d["sha256"] = sha256(p)
            d["lfs_sha256"] = s.lfs.sha256
            d["sha_match"] = d["sha256"] == s.lfs.sha256
        rec.append(d)
        print(json.dumps(d), flush=True)
    out = dict(repo=REPO, revision=REV, last_modified=str(info.last_modified), gated=info.gated,
               seconds=round(time.time() - t0, 1), total_bytes=sum(r["bytes"] for r in rec), files=rec)
    Path(a.prov).parent.mkdir(parents=True, exist_ok=True)
    Path(a.prov).write_text(json.dumps(out, indent=1))
    print("total GB", out["total_bytes"] / 1e9, "seconds", out["seconds"])


if __name__ == "__main__":
    main()
