#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail

run=$1
repo="$run/repo"
plan_root=$(dirname "$run")
base=$(dirname "$plan_root")
study="$base/vcell_liver_r11_chemical_probe/assets"
reports="$repo/reports"
plan="$reports/r19_plan"
matrix="$plan_root/matrix-index.91pT0e4H/index"
export PYTHONPATH="$repo"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4

"$study/study-env/bin/python" -m src.r19_plan.snu423_donor_support \
  --base "$base" --reports "$reports" --r18-registered "$plan_root/audit.YBfVmbuo/r18-registered" \
  --r18-lock "$plan_root/preflight.kwjS2aLo/r18-input-lock" --r18-access "$plan_root/access-plan.h0O8ex4z/allowlists" \
  --r18-index "$matrix" --matrix-reference "$plan/data/preflight/r18/index/GSE92742/receipt.json" \
  --support-reference "$plan/data/t4_support/receipt.json" --tahoe "$base/vcell_liver_goal_tracks/v1" \
  --tahoe-reference "$plan/data/t2_control_support/receipt.json" --entities "$study/features/entities.tsv" \
  --gene-reference "$reports/r15_lincs/data/external/A/A/receipt.json" --genes "$reports/r15_lincs/data/external/A/A/genes.tsv" \
  --structure-reference "$plan/data/hcc_scaffold/structure/receipt.json" --structure "$plan/data/hcc_scaffold/structure/parent_support.tsv" \
  --protocol "$plan/SNU423_DONOR_SUPPORT_PROTOCOL.md" --out "$run/audit"
