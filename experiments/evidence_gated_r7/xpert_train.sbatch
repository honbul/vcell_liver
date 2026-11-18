#!/usr/bin/env bash
#SBATCH -J r7_xptrain
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem 160G
#SBATCH -t 72:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch [-J r7_xp_<id>] experiments/evidence_gated_r7/xpert_train.sbatch <run_id> <num_epochs> <seed> [official|ecfp]
# Upstream XPert training script, unchanged, on the R7 h5ad (train / valid / copy-of-valid as 'test'; no R7 test unit).
set -euo pipefail
source experiments/evidence_gated_r7/env.sh
git update-index -q --refresh >/dev/null 2>&1 || true; DIRTY=$(git status --porcelain --untracked-files=no); test -z "$DIRTY" || { echo "worktree not clean:"; echo "$DIRTY"; exit 3; }
RUN=$1; EP=$2; SEED=$3; VAR=${4:-official}
S="$R7_ASSETS/xpert_work"                 # shared prepared inputs (h5ad, drug features)
W="$R7_ASSETS/xpert_runs_work/$RUN"       # per-run working copy: its experiment/ folder belongs to this run only
test ! -e "$W" || { echo "run dir exists: $W"; exit 4; }
mkdir -p "$W/processed_data"
ln -s "$S/processed_data/l1000_r7sdst.h5ad" "$W/processed_data/l1000_r7sdst.h5ad"
ln -s "$S/processed_data/r7_PPI_gene_vector_128d.npy" "$W/processed_data/r7_PPI_gene_vector_128d.npy"
ln -s "$S/r7_drug_unimol.npy" "$W/r7_drug_unimol.npy"; ln -s "$S/r7_drug_hg.npy" "$W/r7_drug_hg.npy"
ln -s "$S/r7_drug_morgan1024.npy" "$W/r7_drug_morgan1024.npy"; ln -s "$S/r7_drug_hg_zero.npy" "$W/r7_drug_hg_zero.npy"
echo "start=$(date -Is) commit=$R7_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} run=$RUN epochs=$EP seed=$SEED"
rsync -a --exclude .git --exclude processed_data --exclude experiment "$R7_ASSETS/external/XPert/" "$W/"
mkdir -p "$W/experiment"
"$XP_PY" - "$W" "$EP" "$VAR" <<'PY'
import sys, yaml
w, ep, var = sys.argv[1], int(sys.argv[2]), sys.argv[3]
c = yaml.safe_load(open(f"{w}/configs/config_l1000.yaml"))
c["model"]["ATTN"]["ppi_gene_vector_path"] = "processed_data/r7_PPI_gene_vector_128d.npy"
c["model"]["HG"]["drug_hg_pretrained_embed_path"] = "r7_drug_hg.npy"
c["dataset"]["drug_unimol_path"] = "r7_drug_unimol.npy"
c["train"]["num_epochs"] = ep
if var == "ecfp":       # addendum A-2: input-matched drug features, HG embedding removed
    c["dataset"]["drug_morgan_path"] = "r7_drug_morgan1024.npy"
    c["model"]["HG"]["drug_hg_pretrained_embed_path"] = "r7_drug_hg_zero.npy"
yaml.safe_dump(c, open(f"{w}/configs/config_r7.yaml", "w"))
print(c["train"])
PY
cd "$W"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
DF=unimol; [ "$VAR" = ecfp ] && DF=morgan
"$XP_PY" train_xpert.py --model XPert --config config_r7 --drug_feat $DF --nfold r7_split --dataset l1000_r7sdst \
    --use_gradscaler True --include_cell_idx True --seed "$SEED" 2>&1 | grep -v "it/s\]" 
N=$(ls -d experiment/l1000/*/ | wc -l); test "$N" = 1 || { echo "expected one experiment folder, found $N"; exit 5; }
D=$(ls -d experiment/l1000/*/)
mkdir -p "$R7_RUNS/xpert/$RUN"
cp -r "$D" "$R7_RUNS/xpert/$RUN/expt"
cp configs/config_r7.yaml "$R7_RUNS/xpert/$RUN/"
echo "expt=$D"
echo "end=$(date -Is)"
