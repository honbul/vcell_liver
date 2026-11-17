"""Second freeze, for the user-approved post-gate generator pilot.

`--write`   requires that the evaluation sources still hash to the FIRST freeze manifest (same
            metric and protocol code), then records the pilot configuration, the code commit and
            hashes of the generator sources. Refuses to overwrite.
`--verify`  fails unless all recorded sources still match.
No treated cell of a validation or test drug is read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from .freeze import FROZEN_FILES, PKG

PILOT_FILES = ["ddpm.py", "cellflow_adapter.py", "pilot_relabel.py"]
FIRST = Path("manifests/anchored_distribution_r3/freeze_manifest.json")
OUT = Path("manifests/anchored_distribution_r3/freeze_manifest_pilot.json")
CFG = Path("configs/anchored_distribution_r3/pilot_final.json")


def sha(f):
    return hashlib.sha256((PKG / f).read_bytes()).hexdigest()


def check_first():
    first = json.loads(FIRST.read_text())
    bad = [f for f in FROZEN_FILES if sha(f) != first["source_sha256"][f]]
    if bad:
        raise SystemExit(f"evaluation sources differ from the first freeze: {bad}")
    return first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    first = check_first()
    if a.write:
        if OUT.exists():
            raise SystemExit("pilot freeze manifest already exists; it is written once")
        man = dict(
            study="R3 anchored_distribution_r3 -- user-approved post-gate generator pilot (exploratory)",
            status="The pre-specified gate G1 failed and the generator claim was put on hold; the baselines' held-out "
                   "scores were reported BEFORE these models were designed. This is a second look at a reused benchmark.",
            frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            code_commit=os.popen("git rev-parse HEAD").read().strip(),
            first_freeze=dict(code_commit=first["code_commit"], frozen_at=first["frozen_at"],
                              evaluation_sources_unchanged=True),
            evaluation_source_sha256={f: sha(f) for f in FROZEN_FILES},
            pilot_source_sha256={f: sha(f) for f in PILOT_FILES},
            config=json.loads(CFG.read_text()))
        OUT.write_text(json.dumps(man, indent=1, ensure_ascii=False))
        print(json.dumps(man, indent=1, ensure_ascii=False))
    if a.verify:
        man = json.loads(OUT.read_text())
        bad = [f for f in PILOT_FILES if sha(f) != man["pilot_source_sha256"][f]]
        if bad or json.loads(CFG.read_text()) != man["config"]:
            raise SystemExit(f"pilot sources or config changed after the pilot freeze: {bad}")
        print(f"pilot freeze verified; frozen commit {man['code_commit'][:7]}")


if __name__ == "__main__":
    main()
