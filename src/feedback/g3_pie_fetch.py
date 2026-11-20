#!/usr/bin/env python
"""G3 (G1-a prep): fetch the pinned PIE replogle_wdataset checkpoints and the assets they reference.

Runs inside the dedicated PIE env (assets/envs/pie/venv). Everything is pinned:
  - checkpoints: arcinstitute/PIE_replogle_wdataset @ MODEL_REV (sha256 checked against HF LFS oid)
  - data/sources: the hf:// references embedded in each checkpoint config (resolved with PIE's own
    resolver, so the on-disk layout is exactly the one `pie eval` expects)
  - splits: arcinstitute/PIE_splits @ SPLITS_REV (the revision named on the model card)
Writes a provenance JSON (revisions, sha256, sizes, checkpoint config summary).
No credentials are used; all repos are public (gated=False on 2026-11-10).
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

MODEL_REPO = "arcinstitute/PIE_replogle_wdataset"
MODEL_REV = "055c7a2cabdaff5659121c64fa2b609c231fc5be"
SPLITS_REV = "396ab9563175ee887750c9eed7ccaea6f5fdbf50"
FOLDS = ["hepg2", "jurkat", "k562", "rpe1"]
SETTINGS = ["unseen_ctx", "unseen_pert", "unseen_ctx_pert"]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def du(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="local dir for checkpoints")
    ap.add_argument("--out", required=True, help="provenance json path")
    ap.add_argument("--folds", nargs="+", default=FOLDS)
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    from pie.assets import resolve_asset, resolve_split_file
    from pie.predict import load_checkpoint

    t0 = time.time()
    api = HfApi()
    info = api.model_info(MODEL_REPO, revision=MODEL_REV, files_metadata=True)
    lfs = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    prov = dict(model_repo=MODEL_REPO, model_revision=MODEL_REV, splits_revision=SPLITS_REV,
                folds={}, assets={}, splits={})
    models = Path(args.models)
    for fold in args.folds:
        files = [f"{fold}/best_auprc.ckpt", f"{fold}/config.yaml", f"{fold}/data_stats.json"]
        files += [f"{fold}/eval/{s}/{k}_best_auprc.csv" for s in SETTINGS for k in ("metrics", "granular")]
        for f in files:
            hf_hub_download(MODEL_REPO, f, revision=MODEL_REV, local_dir=models)
        ck = models / fold / "best_auprc.ckpt"
        digest = sha256(ck)
        loaded = load_checkpoint(ck)
        d = loaded.config.data
        prov["folds"][fold] = dict(
            ckpt_sha256=digest, hf_lfs_sha256=lfs.get(f"{fold}/best_auprc.ckpt"),
            sha_match=digest == lfs.get(f"{fold}/best_auprc.ckpt"), ckpt_bytes=ck.stat().st_size,
            preprocessed_dirs=list(d.preprocessed_dirs), split_dir=str(d.split_dir),
            source_dirs={k: str(v) for k, v in d.source_dirs.items()}, gene_text_dir=str(d.gene_text_dir),
            aliases_path=str(d.aliases_path) if d.aliases_path else None,
            train_json_sha256=loaded.stats.train_json_sha256, evidence_key=loaded.stats.evidence_key,
            precision=loaded.config.trainer.precision, seed=loaded.config.seed)
        refs = [("preprocessed", p) for p in d.preprocessed_dirs]
        refs += [("source", v) for v in d.source_dirs.values()] + [("source", d.gene_text_dir)]
        for kind, ref in refs:
            ref = str(ref)
            if ref in prov["assets"]:
                continue
            t1 = time.time()
            path = resolve_asset(ref, kind=kind)
            prov["assets"][ref] = dict(kind=kind, local=str(path), bytes=du(path),
                                       seconds=round(time.time() - t1, 1))
            print(f"[fetch] {ref} -> {path} {prov['assets'][ref]['bytes'] / 1e9:.2f} GB", flush=True)
        for s in SETTINGS:
            for name in ("train", "val", "test"):
                ref = (f"hf://datasets/arcinstitute/PIE_splits@{SPLITS_REV}/replogle_wdataset/"
                       f"{s}/{fold}/{name}.json")
                p = resolve_split_file(ref)
                prov["splits"][f"{s}/{fold}/{name}"] = dict(local=str(p), sha256=sha256(p))
        print(f"[fetch] fold {fold} sha_match={prov['folds'][fold]['sha_match']}", flush=True)
    prov["total_asset_bytes"] = sum(v["bytes"] for v in prov["assets"].values())
    prov["seconds"] = round(time.time() - t0, 1)
    prov["env"] = {k: os.environ.get(k) for k in ("PIE_DATA_ROOT", "PIE_RUNS_ROOT", "PIE_CACHE_DIR")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(prov, indent=1))
    print(json.dumps({k: v for k, v in prov.items() if k in ("total_asset_bytes", "seconds")}))


if __name__ == "__main__":
    main()
