#!/usr/bin/env bash
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
lora_pid=${1:?seeded LoRA process ID required}
while kill -0 "$lora_pid" 2>/dev/null; do sleep 30; done
source envs/activate.sh geneformer
python - <<'PY'
from pathlib import Path
import json
p=Path('reports/research_phase2/runs/M03_full4096_fold0_seed42_v2/run.json')
assert p.exists(),'Seeded LoRA did not produce an executed run; inspect failure before confirmation'
j=json.loads(p.read_text());assert j['status']=='executed' and j['initialization_seed']==42
s=json.loads(Path('configs/research/confirmation_selection_v2.json').read_text());assert s['status']=='frozen_confirmation_design'
PY
# Manifest row0 reuses the executed M02 screen; rows1..23 are the remaining paired runs.
sbatch --parsable --array=1-23%2 slurm/adapted_confirmation.sh > reports/research_phase2/adapted_confirmation_slurm_job_id.txt
cat reports/research_phase2/adapted_confirmation_slurm_job_id.txt
