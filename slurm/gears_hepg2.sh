#!/usr/bin/env bash
#SBATCH --job-name=vc_gears_hepg2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_hepg2_%j.log
set -euo pipefail
source envs/activate.sh research
python - <<'PY'
from pathlib import Path
import json
p=Path('reports/research_phase2/runs/R01_norman_simulation_seed1_v2/run.json')
assert p.exists(),'Official Norman must complete before this strict extension'
j=json.loads(p.read_text());assert j['status']=='executed' and j['epochs']==20
print('OFFICIAL_NORMAN_GATE_PASSED',flush=True)
PY
python scripts/research/run_gears_hepg2.py --run-id B03_hepg2_strict_unseen_target_seed42_v1
