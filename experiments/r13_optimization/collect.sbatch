#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python="$PWD/../vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
"$python" -m src.r13_optimization.collect --run "runs/$1" --out "runs/$1/collected"
