#!/usr/bin/env bash
# R3-6: the one post-freeze held-out-drug evaluation (baselines only; generators on hold by gate).
# Refuses to run unless the freeze manifest exists and the frozen sources are unchanged.
#SBATCH --job-name=r3_final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r3_%x_%j.log
set -euo pipefail
source experiments/anchored_distribution_r3/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
PY="$R3_PY -u -m src.anchored_distribution_r3"
$PY.freeze --verify
CFG=configs/anchored_distribution_r3/final.json
METHODS=$($R3_PY -c "import json;print(','.join(json.load(open('$CFG'))['methods']))")
SAVEV=$($R3_PY -c "import json;print(json.load(open('$CFG'))['restoration_variant'])")
SNU=$($R3_PY -c "import json;print(json.load(open('$CFG'))['snu423_line_index'])")
T=reports/anchored_distribution_r3/tables
REFS=shifted:pre,shifted:anchored,shifted:anchored2,gauss_lr50:pre,gauss_lr50:anchored,gauss_lr50:anchored2
$PY.mean_model --mode fit --setting final
$PY.evaluate --setting final --tag final_v1 --methods "$METHODS" --sample-seeds 0,1,2 --save-cells --save-variant "$SAVEV"
mkdir -p predictions/anchored_distribution_r3
$PY.aggregate --setting final --tags final_v1 --out $T/final_all_lines --refs $REFS \
  --cond-table predictions/anchored_distribution_r3/final_condition_metrics.tsv
$PY.aggregate --setting final --tags final_v1 --out $T/final_snu423 --refs $REFS --lines "$SNU"
$R3_PY - <<PYEOF
import glob, hashlib, json, os
rows = []
for f in sorted(glob.glob(os.environ["R3_ASSETS"] + "/gen/final/final_v1__*.npz") + glob.glob(os.environ["R3_ASSETS"] + "/runs/eval/final/final_v1.*")
                + glob.glob(os.environ["R3_ASSETS"] + "/runs/mean/final/*.npy")):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    rows.append(dict(path=f, bytes=os.path.getsize(f), sha256=h.hexdigest()))
json.dump(rows, open("predictions/anchored_distribution_r3/server_files.json", "w"), indent=1)
print(len(rows), "server files hashed")
PYEOF
gzip -f predictions/anchored_distribution_r3/final_condition_metrics.tsv
echo "end=$(date -Is)"
