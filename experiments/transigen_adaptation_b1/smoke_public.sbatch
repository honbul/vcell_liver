#!/usr/bin/env bash
#SBATCH -J tb_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem 24G
#SBATCH -t 01:00:00
#SBATCH -o logs/tb_smoke_%j.out
# Stage B0: public example I/O, public checkpoint forward/loss/inference, and the upstream training
# entry point run on a patched *copy* (the pinned checkout itself stays unmodified).
set -euo pipefail
source experiments/transigen_adaptation_b1/env.sh
echo "start=$(date -Is) commit=$(git rev-parse --short HEAD) host=$(hostname)"
M=manifests/transigen_adaptation_b1
"$TB_PY" -m pip freeze > $M/env_pip_freeze.txt
"$TB_PY" -m src.transigen_adaptation_b1.smoke_public --out $M/public_smoke.json

COPY="$TB_ASSETS/external/TranSiGen_smoke_copy"
rm -rf "$COPY" "$COPY.orig"; mkdir -p "$COPY" "$COPY.orig"
git -C "$TB_TSG_SRC" archive 8ec2218e2fe4fbb5f3a2c14271a8640a62c1af3a | tar -x -C "$COPY"
git -C "$TB_TSG_SRC" archive 8ec2218e2fe4fbb5f3a2c14271a8640a62c1af3a src | tar -x -C "$COPY.orig"
# version compatibility only: np.str was removed from NumPy
sed -i 's/np\.str)/str)/g' "$COPY/src/utils.py"
( cd "$COPY.orig/.." && diff -u "$(basename "$COPY.orig")/src" "$(basename "$COPY")/src" ) > $M/upstream_smoke_compat.diff || true
cat $M/upstream_smoke_compat.diff
cd "$COPY/src"
PATH="$(dirname "$TB_PY"):$PATH" "$TB_PY" train_TranSiGen_full_data.py \
    --data_path ../data/LINCS2020/data_example/processed_data_id.h5 --molecule_path ../data/LINCS2020/idx2smi.pickle \
    --molecule_feature ECFP4 --initialization_model random --split_data_type smiles_split \
    --n_epochs 3 --n_latent 100 --molecule_feature_embed_dim 400 --batch_size 64 --learning_rate 1e-3 \
    --beta 0.1 --dropout 0.1 --weight_decay 1e-5 --train_flag True --eval_metric True 2>&1 | grep -v "^TranSiGen(\|^  \|^)" | tail -40
ls -la ../results/trained_models_*_cell_smiles_split/364039/feature_ECFP4_init_random/ ../results/trained_models_*_cell_smiles_split/364039/feature_ECFP4_init_random/predict
echo "end=$(date -Is)"
