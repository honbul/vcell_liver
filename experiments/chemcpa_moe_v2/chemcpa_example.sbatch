#!/usr/bin/env bash
# Goal section 2: actually run one official chemCPA example, to see the model's inputs and outputs
# and its native training loop before anything is changed.
#
# Configuration is the repo's own `config/dataset/default.yaml` (sci-Plex, all_DEGs) against
# `sciplex_complete_middle_subset.h5ad`, the file the repo's own test_config.yaml names, downloaded
# from the authors' bucket. Two things about that file had to be worked around, neither of them a
# change to the model or the procedure: it is stored in the pre-2021 h5ad layout, which the current
# loader cannot decode, so it is re-saved once through anndata; and its `split_cellcycle_ood` column
# spells the held-out level "odd", which would leave chemCPA's ood split empty, so the run uses
# `split_random`, another of the file's own split columns. The drug embedding is produced by the repo's own
# `embeddings/rdkit/embedding_rdkit.py`. Only the epoch count is shortened, and `load_pretrained`
# is off because the authors' checkpoint hashes point at their private filesystem.
#
# Nothing reports to an external tracker: WANDB_MODE=offline, set in env.sh.
#SBATCH --job-name=cpa_example
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cpa_example_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
EPOCHS=${1:-6}
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "study_commit=$(git rev-parse HEAD) start=$(date -Is)"
cd "$CPA_SRC"
echo "chemcpa_commit=$(git rev-parse HEAD)"
export PYTHONPATH="$CPA_SRC:${PYTHONPATH:-}"
"$CPA_PY" -u chemCPA/train_hydra.py \
  dataset=default \
  dataset.dataset_path=project_folder/datasets/sciplex_complete_middle_subset_v2.h5ad \
  dataset.split_key=split_random \
  model=lincs \
  model.embedding.datapath=project_folder/embeddings/rdkit/sciplex_complete_middle_subset_rdkit2D_embedding.parquet \
  model.load_pretrained=False \
  training.num_epochs="$EPOCHS" \
  training.checkpoint_freq=2 \
  training.save_dir=project_folder/checkpoints/example \
  wandb.project=chemcpa_local wandb.entity=null
echo "end=$(date -Is)"
