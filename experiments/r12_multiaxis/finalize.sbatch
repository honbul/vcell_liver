#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
set -euo pipefail
variant="$1"
launcher=experiments/r12_multiaxis/cpu.sbatch
if [[ "$variant" == primary ]]; then
  args=(--assets assets --predictions assets/final-predictions --freeze assets/freeze.json)
  bash "$launcher" score "${args[@]}" freeze
  bash "$launcher" score "${args[@]}" --out assets/scored score
else
  args=(--assets assets --predictions assets/pred-scaffold --freeze assets/scaffold-freeze.json
        --split-dir scaffold --stage-dir staged-scaffold)
  bash "$launcher" score "${args[@]}" freeze
  bash "$launcher" score "${args[@]}" --out assets/scored-scaffold score
fi
