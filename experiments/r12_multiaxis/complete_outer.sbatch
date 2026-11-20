#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
kind="$1"
outer="$2"
shift 2
launcher=experiments/r12_multiaxis/gpu.sbatch
if [[ "$kind" == primary ]]; then
  bash "$launcher" develop --phase expanded --prior assets/inner-expanded --outer "$outer" \
    --assets assets --out assets/development --device cuda "$@"
  bash "$launcher" infer --outer "$outer" --assets assets --dev assets/development \
    --out assets/final-predictions --device cuda "$@"
else
  bash "$launcher" develop --phase scaffold --outer "$outer" --assets assets --out assets/dev-scaffold \
    --split-dir scaffold --stage-dir staged-scaffold --device cuda "$@"
  bash "$launcher" infer --outer "$outer" --assets assets --dev assets/dev-scaffold \
    --out assets/pred-scaffold --split-dir scaffold --stage-dir staged-scaffold --device cuda "$@"
fi
