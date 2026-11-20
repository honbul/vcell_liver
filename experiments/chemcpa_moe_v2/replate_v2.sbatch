#!/usr/bin/env bash
# Regenerate the v2 condition table into a scratch directory, to obtain `plate_weights.npy`
# without disturbing the frozen truth. The script prints a SHA-256 for every array the study
# already uses; they must all be identical to the ones in $V2_OUT/data, and only the new file is
# copied across. Nothing here reads a response the arms have not already been scored on.
#SBATCH --job-name=v2_replate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v2_replate_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
SCRATCH="$V2_OUT/data_replate"
rm -rf "$SCRATCH" "$V2_OUT/manifests_replate"
"$CPA_PY" -u -m src.chemcpa_moe_v2.prepare_v2 \
  --raw-dir "$V2_OUT/raw" --prep-dir prior_assets/r3/prep \
  --v1-manifest-dir manifests/moe_drug_v1 \
  --out-dir "$SCRATCH" --manifest-dir "$V2_OUT/manifests_replate"

echo "--- byte comparison against the frozen v2 data"
ok=1
for f in Y.npy valid.npy plate.npy basal.npy fp.npy basal_mean.npy basal_sd.npy dmso_plate_mean.npy; do
  a=$(sha256sum "$V2_OUT/data/$f" | cut -d' ' -f1)
  b=$(sha256sum "$SCRATCH/$f" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "same $f $a"; else echo "DIFFERS $f $a $b"; ok=0; fi
done
[ "$ok" -eq 1 ] || { echo "the regenerated truth is not the frozen one; stopping"; exit 1; }
cp "$SCRATCH/plate_weights.npy" "$V2_OUT/data/plate_weights.npy"
echo "copied plate_weights.npy $(sha256sum "$V2_OUT/data/plate_weights.npy" | cut -d' ' -f1)"
echo "end=$(date -Is)"
