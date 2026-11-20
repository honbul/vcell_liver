#!/usr/bin/env bash
# The selected setting again, at three times the epoch budget, for every seed of one arm.
#
# Why this exists. The 10-epoch runs left chemCPA's own reconstruction loss still falling, and
# the control-mode diagnostic showed that about two thirds of the primary score is absolute
# profile error rather than response error. Section 4 of the plan forbids reading an undertrained
# state as a model's intrinsic weakness, so the budget is raised for **all three arms equally**,
# at the learning rate validation already selected. Nothing new is searched: same arms, same
# learning rate, same seeds, same data, only a longer and equal training opportunity.
#
# Usage: sbatch experiments/chemcpa_moe_v2/budget_cpa.sbatch <arm> <lr_scale> [max_epochs]
#SBATCH --job-name=cpa_budget
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=23:00:00
#SBATCH --output=logs/cpa_budget_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
ARM=${1:?arm}; S=${2:?lr_scale}; E=${3:-30}
export PYTHONPATH="$CPA_SRC:${PYTHONPATH:-}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
for SEED in 42 43 44; do
  RUN="${V2_RUNS:-$V2_OUT/runs_e${E}}/${ARM}_lr${S}_s${SEED}"
  "$CPA_PY" -u -m src.chemcpa_moe_v2.train_cpa \
    --chemcpa-src "$CPA_SRC" --adata "$V2_OUT/tahoe_cells.h5ad" \
    --embedding "$V2_OUT/tahoe_rdkit2D_embedding.parquet" \
    --hparams configs/chemcpa_moe_v2/hparams_official.json \
    --v2-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
    --out-dir "$RUN" --arm "$ARM" --lr-scale "$S" --seed "$SEED" --max-epochs "$E"
done
echo "end=$(date -Is)"
