#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:35:00
set -euo pipefail
study=$1
workspace=$2
out=$3
cd "$workspace"
export PYTHONPATH="$workspace:$workspace/src:$study/vc_build_v1/v0plus-model-deps:$study/liver_dili_v1/deps"
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
previous="$study/vc_build_v1/v0plus-runs/final-8e5d3f4"
"$python" -m v0plus_v2.predict --package "$out/package" --scale bulk_log2_tpm_plus_1 \
  --rna "$previous/replay_rna.tsv" --compounds "$previous/replay_compounds.tsv" --out "$out/offline.tsv"
"$python" - "$out/offline.tsv" <<'PY'
import json, sys
import pandas as pd
import numpy as np
from vc_build.io import publish_json, sha256
from pathlib import Path
path = Path(sys.argv[1])
result = pd.read_csv(path,sep="\t")
expected = result.status.eq("predicted")
assert expected.any() and np.isfinite(result.loc[expected,"sensitivity_score"]).all()
assert np.allclose(result.loc[expected,"sensitivity_score"]-result.loc[expected,"drug_mean_anchor"],
                   result.loc[expected,"personalization_delta"])
publish_json(path.with_suffix(".json"), {"rows":len(result), "status":result.status.value_counts().to_dict(),
                                        "model":str(result.model.iloc[0]),"offline_fm_cells":int(result.cells_fm.sum()),
                                        "sha256":sha256(path)})
PY
