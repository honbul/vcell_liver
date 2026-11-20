#!/usr/bin/env bash
#SBATCH -J gf_p0_audit
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 02:00:00
#SBATCH -o logs/gf_p0_audit_%j.out
set -euo pipefail
source experiments/genotype_film_v3/env.sh
echo "start=$(date -Is) commit=$(git rev-parse --short HEAD) host=$(hostname)"
"$GF_PY" -m src.genotype_film_v3.audit_metadata --out manifests/genotype_film_v3
echo "end=$(date -Is)"
