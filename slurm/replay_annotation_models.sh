#!/usr/bin/env bash
#SBATCH --job-name=vc_annotation_replay
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_annotation_replay_%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
source envs/activate.sh scgpt
probe_id=${1:-classification_native64_v1}
.venvs/vcell-runtime310/bin/python envs/portable310.py --env research scripts/research/replay_annotation_models.py --model scvi --probe-id "$probe_id"
.venvs/vcell-liverct/bin/python scripts/research/replay_annotation_models.py --model liverct --probe-id "$probe_id"
