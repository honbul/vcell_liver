#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail

run=$1
plan=$(dirname "$run")
base=$(dirname "$plan")
repo="$run/repo"
reports="$repo/reports/r19_plan"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo" OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /usr/bin/time -v "$python" -m src.r19_plan.hcc_phenotype_support \
  --data "$base/vcell_liver_dev_i/coverage.bk00jUHu/data" \
  --visible "$plan/hcc-scaffold-refit.lWd6KCZt/visible" --reserved "$plan/audit.YBfVmbuo/r18-registered" \
  --protocol "$reports/HCC_PHENOTYPE_SUPPORT_PROTOCOL.md" --reports "$reports" --out "$run/audit"
