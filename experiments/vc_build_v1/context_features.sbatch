#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail

repository=$1
data=$2
out=$3
shift 3
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
study="$storage/vc_build_v1/function"
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src:$study/prepare-deps"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "$repository"
exec /usr/bin/time -v "$python" -m vc_build.context_features \
  --config configs/vc_build_v1/contract.yaml --prepared "$data" --out "$out" \
  --driver-workbook "$study/sources/LIMORE-CancerCell-2019-TableS3.xlsx" \
  --hallmark-gmt /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt "$@"
