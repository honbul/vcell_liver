"""Verify the offline CLI in separate processes using only frozen local assets."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from vc_fm_bench.artifacts import digest, new_directory, write_json
from .data import read


def replay(history, outputs):
    with new_directory(outputs) as stage:
        lines = stage / "batch.tsv"
        lines.write_text("smiles\nCCO\nC[Si](C)(C)C\n")
        # Enforce offline behavior in each subprocess by denying network socket creation.
        guard = stage / "guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text(
            "import sys\n"
            "def deny(event, args):\n"
            "    if event in ('socket.connect', 'socket.getaddrinfo'):\n"
            "        raise RuntimeError('Offline replay network access blocked')\n"
            "sys.addaudithook(deny)\n"
        )
        env = dict(os.environ, PYTHONPATH=f"{guard}:{os.environ.get('PYTHONPATH', '')}",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        command = [sys.executable, "-m", "confirm_dili.predict", "--model", str(history)]

        def call(flag, value):
            proc = subprocess.run(command + [flag, value], env=env, text=True, capture_output=True)
            if proc.returncode:
                raise RuntimeError(proc.stderr)
            return json.loads(proc.stdout)

        ethanol = call("--smiles", "CCO")
        novel = call("--smiles", "C[Si](C)(C)C")
        batch = call("--batch", str(lines))
        invalid = subprocess.run(command + ["--smiles", "this is not SMILES"], env=env, text=True, capture_output=True)
        fields = ("probability_of_dilirank_concern", "nearest_training_tanimoto", "calibration_parameter_interval90")
        difference = max(float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key]))))
                         for a, b in zip([ethanol, novel], batch) for key in fields)
        if len(batch) != 2 or difference >= 1e-6 or not novel["abstain"] or not novel["warning"]:
            raise ValueError("Offline CLI consistency or low-support test failed")
        if invalid.returncode != 2 or "Invalid SMILES" not in invalid.stderr or invalid.stdout:
            raise ValueError("Offline CLI accepted a malformed molecular input")
        if not 0 <= ethanol["probability_of_dilirank_concern"] <= 1:
            raise ValueError("Offline CLI emitted a malformed probability")
        if not np.allclose(ethanol["calibration_parameter_interval90"], sorted(ethanol["calibration_parameter_interval90"])):
            raise ValueError("Calibration interval endpoints crossed")
        receipt = dict(
            passed=True, model=read(history / "freeze.json")["deployment"], historical_freeze_sha256=digest(history / "freeze.json"),
            single=ethanol, novel_low_support=novel, batch_single_max_abs_diff=difference,
            invalid_smiles_exit_code=invalid.returncode, offline_python_network_audit_hook=True,
            model_input="SMILES", extractor="frozen ICML 2023 CLAMP molecule-only encoder")
        write_json(stage / "dili_cli_replay_receipt.json", receipt)


if __name__ == "__main__":
    replay(Path(sys.argv[1]), Path(sys.argv[2]))
