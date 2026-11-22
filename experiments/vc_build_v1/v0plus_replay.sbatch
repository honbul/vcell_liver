#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:30:00
set -euo pipefail

repo=$1
root=$2
run=$3
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repo/src:$root/v0plus-model-deps:$runtime"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""

"$python" - "$root" "$run" <<'PY'
from pathlib import Path
import sys

import pandas as pd

root, run = map(Path, sys.argv[1:])
rna = pd.read_parquet(root / "function/cross-v1/test_inputs/rna.parquet").sort_index().head(3)
rna.rename_axis("sample_id").to_csv(run / "replay_rna.tsv", sep="\t")
drugs = pd.read_csv(run / "V0plus/compounds.tsv", sep="\t", usecols=["compound_id"])
drugs.loc[len(drugs), "compound_id"] = "UNREGISTERED_REPLAY_COMPOUND"
drugs.to_csv(run / "replay_compounds.tsv", sep="\t", index=False)
PY
args=(
  --package "$run/V0plus" --rna "$run/replay_rna.tsv" --compounds "$run/replay_compounds.tsv"
  --scale bulk_log2_tpm_plus_1 --out "$run/replay_predictions.tsv"
)
/usr/bin/time -f 'offline wall=%e maxrss_kib=%M' "$python" -m vc_build.v0plus_cli predict "${args[@]}"
"$python" - "$run" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from vc_build.io import publish_json, sha256, verify_forecast

run = Path(sys.argv[1])
frozen, seal = verify_forecast(run / "final.parquet")
card = json.loads((run / "V0plus/release.json").read_text())
method = card["default_model"]
actual = pd.read_csv(run / "replay_predictions.tsv", sep="\t").rename(columns={"sample_id": "cell_id"})
predicted = actual[actual.status.eq("predicted")]
abstained = actual[actual.status.eq("compound_outside_training_panel")]
key = ["cell_id", "compound_id"]
comparison = predicted.merge(frozen[frozen.method.eq(method)][[*key, "predicted_value"]], on=key, validate="1:1")
if len(comparison) != 204 or len(abstained) != 3:
    raise ValueError("Offline replay does not reproduce the registered coverage")
error = np.abs(comparison.sensitivity_score - comparison.predicted_value)
if error.max() > 0.0003:
    raise ValueError(f"Offline replay changed the GPU values: {error.max()}")
publish_json(run / "OFFLINE_REPLAY.json", {
    "samples": int(comparison.cell_id.nunique()), "scored_replay_pairs": len(comparison),
    "explicit_unknown_drug_abstentions": len(abstained), "max_abs_difference": float(error.max()),
    "freeze_sha256": seal["sha256"], "release_receipt_sha256": sha256(run / "V0plus/release.json"),
    "cli_output_sha256": sha256(run / "replay_predictions.tsv"), "cpu_only": True,
})
PY
