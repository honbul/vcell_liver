#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -euo pipefail

repo=$1
root=$2
bundle=$3
run=$4
auxiliary=$5
arms=(v0plus_inductive v0plus_transductive v0plus_fixed no_chronos no_driver no_pathways no_targets no_alignment no_rank)
arm=${arms[SLURM_ARRAY_TASK_ID / 3]}
export SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID % 3))
exec bash "$repo/experiments/vc_build_v1/v0plus_cv.sbatch" "$repo" "$root" "$bundle" "$arm" "$run" --gpu --auxiliary "$auxiliary"
