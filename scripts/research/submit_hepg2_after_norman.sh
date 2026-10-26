#!/usr/bin/env bash
# Only a lightweight monitor runs outside Slurm; the training itself is a scheduled job.
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
norman_pid=${1:?official Norman process ID required}
while kill -0 "$norman_pid" 2>/dev/null; do sleep 30; done
source envs/activate.sh research
python - <<'PY'
import json
from pathlib import Path
p=Path('reports/research_phase2/runs/R01_norman_simulation_seed1_v2/run.json')
assert p.exists(),'Norman stopped without executed result; inspect failure before submitting extension'
j=json.loads(p.read_text());assert j['status']=='executed' and j['epochs']==20
PY
sbatch --parsable slurm/gears_hepg2.sh > reports/research_phase2/hepg2_slurm_job_id.txt
cat reports/research_phase2/hepg2_slurm_job_id.txt
