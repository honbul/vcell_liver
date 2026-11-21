#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
launcher=experiments/r12_multiaxis/gpu.sbatch
for mode in replay transfer; do
  bash "$launcher" secondary --assets assets --primary-freeze assets/freeze.json \
    --out "assets/$mode" --device cuda --mode "$mode" predict
done
