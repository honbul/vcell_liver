"""Assemble a portable offline package and replay its launcher."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from vc_fm_bench.artifacts import digest, file_record, new_directory, verify_record, write_json
from .data import read
from .molecular import CHECKPOINT, checkpoint_verified


def package(args):
    policy = read(args.history / "freeze.json")
    verify_record(args.history / "checkpoint.joblib", policy["checkpoint"])
    checkpoint_verified(args.encoder)
    replay = read(args.replay / "dili_cli_replay_receipt.json")
    if not replay["passed"] or replay["historical_freeze_sha256"] != digest(args.history/"freeze.json"):
        raise ValueError("CLI source replay has not been verified against the selected head")
    with new_directory(args.out) as package:
        source_names = {
            "confirm_dili": ("__init__.py", "data.py", "molecular.py", "models.py", "predict.py"),
            "vc_fm_bench": ("__init__.py", "artifacts.py", "clamp.py"),
        }
        for folder, names in source_names.items():
            destination = package / "app" / folder
            destination.mkdir(parents=True)
            for name in names:
                shutil.copy2(Path(folder) / name, destination / name)
        for folder, source, names in (
            ("model", args.history, ("checkpoint.joblib", "freeze.json")),
            ("encoder", args.encoder, tuple(CHECKPOINT)),
        ):
            destination = package / folder
            destination.mkdir()
            for name in names:
                shutil.copy2(source / name, destination / name)
        checkpoint_verified(package / "encoder")
        verify_record(package / "model/checkpoint.joblib", policy["checkpoint"])
        packages = ("joblib", "lightgbm", "numpy", "pandas", "rdkit", "scikit-learn", "scipy", "torch")
        (package / "requirements-tested.txt").write_text("".join(f"{name}=={version(name)}\n" for name in packages))
        launcher = package / "predict.sh"
        launcher.write_text('''#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$here/app${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${PYTHON:-python3}" -m confirm_dili.predict --model "$here/model" --encoder "$here/encoder" "$@"
''')
        launcher.chmod(0o755)
        from sys import executable
        env = dict(os.environ, PYTHON=executable)
        for field in ("single", "novel_low_support"):
            expected = replay[field]
            result = subprocess.run([launcher, "--smiles", expected["input_smiles"]],
                                    cwd=package, env=env, check=True, capture_output=True, text=True)
            emitted = json.loads(result.stdout)
            for key in ("probability_of_dilirank_concern", "calibration_parameter_interval90",
                        "nearest_training_tanimoto"):
                if not np.allclose(emitted[key], expected[key], atol=1e-7, rtol=0):
                    raise ValueError(f"Portable CLI differs from the source replay: {key}")
            if emitted["abstain"] != expected["abstain"]:
                raise ValueError("Portable CLI changed its out-of-domain decision")
        manifest = {str(p.relative_to(package)): file_record(p) for p in package.rglob("*") if p.is_file()}
        write_json(package / "PACKAGE_RECEIPT.json", dict(
            launcher_replay_passed=True, model=policy["deployment"],
            historical_freeze_sha256=digest(args.history/"freeze.json"),
            asset_manifest=manifest, redistributes_public_author_checkpoint=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "encoder", "replay", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    package(parser.parse_args())
