#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=02:15:00
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
worker="$root/confirm_20261123/worker2_dc"
code=$1
run=$2
stage=$3
cd "$code"
py="$root/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$code:$root/top3_tuning_v2/b/code/final-verified:$root/liver_dili_v1/deps:$root/vc_build_v1/v0plus-model-deps:$root/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "$stage" in
    history)
        "$py" -m confirm_dili.fit history --out "$run/historical"
        ;;
    predict)
        scopes=(murcko-{0..4} butina-{0..4} release-0)
        scope=${scopes[${SLURM_ARRAY_TASK_ID:?}]}
        "$py" -m confirm_dili.fit forecast --scope "$scope" --history "$run/historical" --out "$run/forecasts/$scope"
        ;;
    score)
        "$py" -m confirm_dili.score --run "$run" --out "$run/score"
        ;;
    cli)
        "$py" -m confirm_dili.replay "$run/historical" "$run/cli"
        ;;
    *)
        exit 2 ;;
esac
