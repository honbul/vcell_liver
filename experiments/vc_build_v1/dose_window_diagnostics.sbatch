#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:40:00
set -euo pipefail

repository=$1
root=$2
run=$3
work=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
sources=/mnt/S1/sdata/pubdata/DepMap/DrugScreens
export PYTHONPATH="$repository/src:$root/v0plus-model-deps"
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2
exec "$work/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" +  -m vc_build.dose_window_diagnostics +  --bundle "$root/function/cross-v1" +  --ctrp-post-qc "$sources/CTRP_CTD2/v20.data.per_cpd_post_qc.txt" +  --treatments "$run/prism20q2-treatment.csv" --out "$run/score.json"
