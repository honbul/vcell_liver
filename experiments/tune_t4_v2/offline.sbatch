#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=12G
#SBATCH --time=00:35:00
set -euo pipefail
study=$1
code=$2
release=$3
pretrain=$4
cd "$code"
r19="$study/vcell_liver_r19_plan"
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$study/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$code:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
replay="$release/offline-02"
mkdir "$replay"
"$python" - "$r19" "$replay" <<'PY'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tune_t4_v2.data import contexts
from src.r19_plan.t4_context_transfer import group_plans, original

source, out = map(Path, sys.argv[1:])
cs = contexts(Path("reports/r19_plan"), source/"t4-run.fS2rpsIm", source/"t4-phh.p1AqqE6o")
for name, ctx in cs.items():
    key, tables = next((key, tables) for key, tables in group_plans(ctx, "f0-A").items()
                       if len(tables["query"]) >= 2)
    rows = tables["query"].iloc[:2]
    normalized = original(ctx, key, tables["query"])["identity6"][:len(rows)]
    df = pd.DataFrame(normalized, columns=ctx.genes).assign(
        sample_id=[f"{name}_{i}" for i in range(len(rows))],
        parent=rows.parent.to_numpy(), dose_uM=rows.actual_dose_uM.astype(float).to_numpy())
    df.to_csv(out/f"{name}_input.tsv", sep="\t", index=False)
PY
for cell in HEPG2 PHH; do
  "$python" -m tune_t4_v2.predict --cell "$cell" --input "$replay/${cell}_input.tsv" \
    --pan "$pretrain" --models "$release/forecasts" --lock "$release/freeze.json" \
    --out "$replay/${cell}_24h.tsv"
done
"$python" - "$replay" <<'PY'
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.r11_chemical_probe.common import sha256
folder = Path(sys.argv[1])
info = {}
for cell in ("HEPG2","PHH"):
    file = folder / f"{cell}_24h.tsv"
    data = pd.read_csv(file,sep="\t")
    assert len(data)==2 and len(data.columns)==982 and data.frozen_heads.gt(0).all()
    assert np.isfinite(data.iloc[:,4:].to_numpy()).all()
    info[cell] = dict(rows=len(data), genes=978, heads=data.frozen_heads.tolist(), forecast_sha256=sha256(file))
(folder/"receipt.json").write_text(json.dumps(info,indent=2)+"\n")
PY
