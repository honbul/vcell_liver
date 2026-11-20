#!/bin/bash
# GPU wrapper for external models (r9ext env). Usage: sbatch -p gpu --gres=gpu:1 ... run_ext.sh <config> <method>
set -euo pipefail
export R9_PY_OVERRIDE=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_rare_state/assets/envs/r9ext/bin/python
exec /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_rare_state/scripts/r9/slurm/run_py.sh scripts/r9/a2_ext.py "$@"
