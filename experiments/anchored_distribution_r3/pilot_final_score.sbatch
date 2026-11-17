#!/usr/bin/env bash
# Pilot, held-out stage: score the frozen-recipe CellFlow output, then aggregate the pilot models
# next to the baseline scores of the first freeze (tag final_v1, not recomputed).
#SBATCH --job-name=r3_pfins
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r3_%x_%j.log
set -euo pipefail
source experiments/anchored_distribution_r3/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
PY="$R3_PY -u -m src.anchored_distribution_r3"
$PY.freeze_pilot --verify
E="$R3_ASSETS/runs/eval/final"
TAGS=pilot_final_ddpm
CF=$(ls assets/gen/cellflow/final/ 2>/dev/null | grep -E "^cf009-final-x2000-it120000-bs1024-s0-[0-9a-f]{8}__ss0.npz$" | sed 's/__ss0.npz$//' || true)
if [ -n "$CF" ]; then
  $PY.evaluate --setting final --tag pilot_final_cf --methods "cells:assets/gen/cellflow/final/${CF}__ss{seed}.npz" --sample-seeds 0,1,2
  TAGS=pilot_final_ddpm,pilot_final_cf
else
  echo "NO CELLFLOW OUTPUT FOUND: its rows stay blank"
fi
TAGS=$TAGS $R3_PY - <<PYEOF
import pandas as pd, os
p = os.environ["R3_ASSETS"] + "/runs/eval/final/"
a = pd.concat([pd.read_csv(p + t + ".tsv.gz", sep="\t") for t in os.environ["TAGS"].split(",")])
a.to_csv(p + "pilot_final_all.tsv.gz", sep="\t", index=False, float_format="%.6g")
PYEOF
$PY.pilot_relabel --setting final --tag pilot_final_all --out-tag pilot_final_named --map configs/anchored_distribution_r3/pilot_final_names.json
T=reports/anchored_distribution_r3/tables
REFS=shifted:anchored2,gauss_lr50:anchored2,control:native,shifted:pre,ddpm_resid:anchored2
SNU=$($R3_PY -c "import json;print(json.load(open('configs/anchored_distribution_r3/pilot_final.json'))['snu423_line_index'])")
$PY.aggregate --setting final --tags final_v1,pilot_final_named --out $T/pilot_final_all_lines --refs $REFS \
  --cond-table predictions/anchored_distribution_r3/pilot_final_condition_metrics.tsv
$PY.aggregate --setting final --tags final_v1,pilot_final_named --out $T/pilot_final_snu423 --refs $REFS --lines "$SNU"
$PY.aggregate --setting final --tags pilot_final_all --out $T/pilot_final_by_train_seed --refs none
$R3_PY - <<PYEOF
import glob, hashlib, json, os
A = os.environ["R3_ASSETS"]
rows = []
for f in sorted(glob.glob(A + "/runs/ddpm/final-*/model.pt") + glob.glob(A + "/runs/ddpm/final-*/record.json")
                + glob.glob(A + "/gen/cellflow/final/cf009-final-*.npz") + glob.glob(A + "/runs/cellflow/cf009-final-*/record.json")
                + glob.glob(A + "/runs/eval/final/pilot_final*")):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    rows.append(dict(path=f, bytes=os.path.getsize(f), sha256=h.hexdigest()))
json.dump(rows, open("predictions/anchored_distribution_r3/pilot_server_files.json", "w"), indent=1)
print(len(rows), "server files hashed")
PYEOF
# keep only pilot rows in the committed per-condition table (baseline rows are already committed)
$R3_PY - <<PYEOF
import pandas as pd
f = "predictions/anchored_distribution_r3/pilot_final_condition_metrics.tsv"
c = pd.read_csv(f, sep="\t")
c[c.method.str.startswith(("ddpm_", "cellflow"))].to_csv(f + ".gz", sep="\t", index=False, float_format="%.5g")
PYEOF
rm -f predictions/anchored_distribution_r3/pilot_final_condition_metrics.tsv
echo "end=$(date -Is)"
