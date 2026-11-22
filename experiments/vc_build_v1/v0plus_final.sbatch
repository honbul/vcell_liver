#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=25G
#SBATCH --time=03:00:00
set -euo pipefail

repository=$1
root=$2
run=$3
mode=$4
shift 4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
bundle="$root/function/cross-v1"
query="$root/function/cross-v1"
cohort=LIMORE50
case "$mode" in
  lock)
    exec "$python" -m vc_build.v0plus_release lock --out "$run/settings.json" "$@"
    ;;
  predict_cpu | predict_gpu)
    if [[ $mode == predict_cpu ]]; then
      arms=(linear lightgbm cellhit)
      device=()
    else
      arms=(catboost velodrome_inductive velodrome_transductive codeae_inductive codeae_transductive
            v0plus_inductive v0plus_transductive v0plus_fixed no_chronos no_driver no_pathways
            no_targets no_alignment no_rank)
      device=(--gpu)
    fi
    arm=${arms[SLURM_ARRAY_TASK_ID]}
    args=(
      --settings "$run/settings.json" --bundle "$bundle" --query-bundle "$query" --cohort "$cohort"
      --priors "$root/v0plus-public-priors"
      --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
      --targets "$root/external-code-v3/drug_targets.json" --upstream "$root/v0plus-upstream"
      --arm "$arm" --out "$run/predict/$arm" --package
    )
    exec "$python" -m vc_build.v0plus_release predict "${args[@]}" "${device[@]}" "$@"
    ;;
  merge)
    exec "$python" -m vc_build.v0plus_release merge --settings "$run/settings.json" --cohort "$cohort" --source "$run/predict" --out "$run/final.parquet"
    ;;
  score)
    exec "$python" -m vc_build.v0plus_score --settings "$run/settings.json" --cohort-bundle "$query" --direction higher --forecast "$run/final.parquet" --out "$run/LIMORE50_score.json"
    ;;
  package)
    exec "$python" -m vc_build.v0plus_cli package --bundle "$bundle" --forecast "$run/final.parquet" --results "$run/LIMORE50_score.json" --out "$run/V0plus" "$@"
    ;;
  *) exit 2 ;;
esac
