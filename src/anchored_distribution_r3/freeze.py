"""Write (or verify) the freeze manifest that must exist before the held-out-drug evaluation.

`--write`   records the code commit, a SHA-256 of every source file of this package, the frozen
            configuration and the evaluation protocol. Refuses to overwrite.
`--verify`  fails unless the package sources still hash to the frozen values.
No treated cell of a validation or test drug is read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

PKG = Path("src/anchored_distribution_r3")
FROZEN_FILES = ["data.py", "metrics.py", "setting.py", "mean_model.py", "baselines.py", "evaluate.py", "aggregate.py"]


def hashes():
    return {f: hashlib.sha256((PKG / f).read_bytes()).hexdigest() for f in FROZEN_FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--config", default="configs/anchored_distribution_r3/final.json")
    ap.add_argument("--out", default="manifests/anchored_distribution_r3/freeze_manifest.json")
    a = ap.parse_args()
    out = Path(a.out)
    if a.write:
        if out.exists():
            raise SystemExit("freeze manifest already exists; it is written once")
        cfg = json.loads(Path(a.config).read_text())
        man = dict(
            study="R3 anchored_distribution_r3",
            frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            code_commit=os.popen("git rev-parse HEAD").read().strip(),
            source_sha256=hashes(),
            config=cfg,
            mean_model=json.loads(Path("configs/anchored_distribution_r3/mean_model.json").read_text()),
            evaluation=dict(
                name="post-freeze evaluation on a reused public benchmark",
                setting="final: fit on all 209 TRAIN drug groups (all eligible cell lines); score TEST drug groups once",
                unused="the prior study's 70 VALIDATION drug groups are used by no step of this study",
                exposure="Tahoe-100M is public and this project reported test scores on this split before (chemCPA study); "
                         "this is not a blind cohort",
                metric="per-well unbiased (U-statistic) energy distance, Euclidean, genes / training within-well SD "
                       "(floor at 10th percentile), no clipping; wells -> condition -> drug, equal weights; "
                       "drug-block bootstrap 2000, seed 20261117",
                n_rule="n = min(observed cells, same-plate control cells, 64); wells with n < 32 dropped",
                hcc_target="SNU-423 (CVCL_0366) reported separately; HepG2/C3A is not called adult HCC"),
            generator_status="absolute DDPM, residual DDPM and CellFlow are NOT trained: gate R3-0 (G1) failed, "
                             "generator claim on hold (reports/anchored_distribution_r3/DECISIONS.md)")
        out.write_text(json.dumps(man, indent=1, ensure_ascii=False))
        print(json.dumps(man, indent=1, ensure_ascii=False))
    if a.verify:
        man = json.loads(out.read_text())
        now = hashes()
        bad = [f for f in FROZEN_FILES if now[f] != man["source_sha256"][f]]
        if bad:
            raise SystemExit(f"sources changed after the freeze: {bad}")
        print(f"freeze verified: {len(FROZEN_FILES)} source files match; frozen commit {man['code_commit'][:7]}")


if __name__ == "__main__":
    main()
