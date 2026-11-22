#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=01:30:00
set -euo pipefail

repo=$1
root=$2
bundle=$3
arm=$4
run=$5
shift 5
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repo/src:$root/v0plus-model-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
exec "$python" -m vc_build.v0plus_cv --bundle "$bundle" +  --priors "$root/v0plus-public-priors" +  --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt +  --targets "$root/external-code-v3/drug_targets.json" --arm "$arm" +  --fold "$SLURM_ARRAY_TASK_ID" --out "$run/$arm/fold$SLURM_ARRAY_TASK_ID" "$@"
