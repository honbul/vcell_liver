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
run() {  # label, then upstream arguments
  echo "=== upstream entry point: $1"; shift
  PATH="$(dirname "$TB_PY"):$PATH" "$TB_PY" train_TranSiGen_full_data.py \
    --data_path ../data/LINCS2020/data_example/processed_data_id.h5 --molecule_path ../data/LINCS2020/idx2smi.pickle \
    --split_data_type smiles_split --n_latent 100 --molecule_feature_embed_dim 400 --learning_rate 1e-3 \
    --beta 0.1 --dropout 0.1 --weight_decay 1e-5 --train_flag True --eval_metric True "$@" 2>&1 \
    | grep -E "^\[Epoch|train |valid |test |best valid|warning|Error|error|cell count|all data" | awk 'NR<=8 || /Error|best|warning/ {print} {last=$0} END{print "last: " last}' \
    || echo "(run exited with an error)"
  ls ../results/trained_models_43_cell_smiles_split/*/feature_*/predict 2>/dev/null || true
}
run "README configuration (KPGT, shRNA-pretrained VAEs), 3 epochs" --molecule_feature KPGT --initialization_model pretrain_shRNA --n_epochs 3 --batch_size 64
run "ECFP4, random init, batch 64, 3 epochs (1 update per epoch on 65 rows)" --molecule_feature ECFP4 --initialization_model random --n_epochs 3 --batch_size 64 --seed 1
run "ECFP4, random init, batch 16, 60 epochs" --molecule_feature ECFP4 --initialization_model random --n_epochs 60 --batch_size 16 --seed 2
cd "$TB_ROOT"
echo "end=$(date -Is)"
