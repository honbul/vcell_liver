#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:20:00
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
worker="$root/confirm_20261123/worker2_dc"
source="$worker/paper-source-01/HCC_SCORE_REPLAY.zip"
work="$worker/paper-replay-01"
expected=9e98f1a0a7d2468381f46eea919e4fb4ac206caa985df6c11af4fec681b79d99
[[ "$(sha256sum "$source" | cut -d ' ' -f 1)" == "$expected" ]]
mkdir "$work"
unzip -q "$source" -d "$work"
cd "$work"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$root/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" -I replay_adult_hcc.py --packet . --out replayed
sha256sum replayed/*.tsv
