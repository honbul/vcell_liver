#!/usr/bin/env bash
#SBATCH -J tb_src_sel
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
# usage: sbatch train_source_selected.sbatch <seed>   (learning rate read from the committed-to-be selection file)
set -euo pipefail
source experiments/transigen_adaptation_b1/env.sh
export TB_COMMIT=$(git rev-parse --short HEAD)
LR=$("$TB_PY" -c "import json;print(json.load(open('manifests/transigen_adaptation_b1/source_lr_selection.json'))['selected_lr'])")
RID=$("$TB_PY" -c "print('source_lr%g_seed%d_v1' % ($LR, $1))")
echo "start=$(date -Is) commit=$TB_COMMIT host=$(hostname) lr=$LR seed=$1 run=$RID"
"$TB_PY" -m src.transigen_adaptation_b1.train_source --lr "$LR" --seed "$1" --run-id "$RID"
echo "end=$(date -Is)"
