#!/usr/bin/env bash
#SBATCH --job-name=rf_paper_audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/paper_handoff_%j.log
set -euo pipefail
work="${1:?provide a new handoff output directory}"
ledger="${2:?provide the paper submission TSV}"
mkdir -p "$work"
test ! -e "$work/paper_status.json"
exec .venvs/gears/bin/python -u src/paper_comparison.py --strict --submission-log "$ledger" \
    --table "$work/reproduction_comparison.tsv" --output "$work/paper_status.json"
