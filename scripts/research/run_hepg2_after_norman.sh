#!/usr/bin/env bash
# Wait for this verified official job to finish; start strict HepG2 only after its executed result exists.
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
norman_pid=${1:?official Norman process ID is required}
while kill -0 "$norman_pid" 2>/dev/null; do sleep 30; done
source envs/activate.sh research
python - <<'PY'
import json
from pathlib import Path
p=Path('reports/research_phase2/runs/R01_norman_simulation_seed1_v2/run.json')
assert p.exists(), 'Norman process stopped without an executed result; inspect its failure/log before HepG2'
j=json.loads(p.read_text());assert j['status']=='executed' and j['epochs']==20
print('OFFICIAL_NORMAN_GATE_PASSED',flush=True)
PY
python scripts/research/run_gears_hepg2.py --run-id B03_hepg2_strict_unseen_target_seed42_v1
