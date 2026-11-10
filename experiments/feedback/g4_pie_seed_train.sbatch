#!/usr/bin/env bash
# G4 task 2: retrain PIE on the replogle_wdataset HepG2-held-out fold with the official recipe and a
# new seed, then evaluate best_auprc (selected on the source-line val split only) on the 3 test sets.
# Usage: sbatch [--gres=gpu:a100:2 | --exclude=g1,g2] experiments/feedback/g4_pie_seed_train.sbatch <seed> [fold]
# Recipe = the official checkpoint's own config.yaml (assets/g3_pie/models/replogle_wdataset/<fold>/config.yaml):
#   2 GPUs DDP (torchrun), batch_size 16/GPU, accumulate_grad_batches 4 (effective 128), max_steps 5000,
#   bf16-mixed, lr 1e-4, val every 100 steps, monitor = val AUPRC; data/sources/split revisions pinned
#   exactly as in that checkpoint (pie_replogle_nadig_essential@20c9fae, pie_sources@cb1aaa4,
#   PIE_splits@396ab95), aliases = packaged curated table (what eval maps the checkpoint's legacy path to).
# Only changes vs the checkpoint config: seed (top-level; evidence seed stays 42 = data recipe),
#   logger.enabled=false (no wandb), run_dir/experiment_name.
#SBATCH --job-name=g4_pie_seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --output=logs/g4_pie_seed_%j.log
set -euo pipefail
source experiments/feedback/g3_pie_env.sh
export HF_HUB_OFFLINE=1   # every pinned asset is already in $PIE_DATA_ROOT/hf (G3 fetch); no downloads
SEED=${1:?seed}; FOLD=${2:-hepg2}
G4_RUNS="$(pwd)/assets/g4/pie_seeds"
RUN_DIR="$G4_RUNS/${FOLD}_seed${SEED}"
mkdir -p "$G4_RUNS"
REV_DATA=hf://datasets/arcinstitute/pie_replogle_nadig_essential@20c9faef76fc96fdc809871340bd93499413dfe7/preprocessed
SRC=hf://datasets/arcinstitute/pie_sources@cb1aaa4e7655605bdc70a9bd77bbd62016b8c7d7
SPLITS=hf://datasets/arcinstitute/PIE_splits@396ab9563175ee887750c9eed7ccaea6f5fdbf50
ALIASES=$("$PIE_PY" -c 'from pie.sources.contract import CURATED_ALIASES_FILE as f; print(f)')
echo "host=$(hostname) job=${SLURM_JOB_ID:-} seed=$SEED fold=$FOLD cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
OVR=(experiment=replogle_wdataset vars.fold=$FOLD seed=$SEED
     experiment_name=g4_seeds/${FOLD}_seed${SEED} run_dir=$RUN_DIR logger.enabled=false
     "data.preprocessed_dirs=[$REV_DATA]"
     data.split_dir=$SPLITS/replogle_wdataset/unseen_ctx/$FOLD
     data.gene_text_dir=$SRC/gene_text data.aliases_path=$ALIASES)
for s in esm2 ncbi_text string_space depmap_gene_effect context_text perturbation_text; do
  OVR+=(data.source_dirs.$s=$SRC/$s)
done
# resume if a previous attempt left last.ckpt (requeue/timeout), else fresh
if [ -f "$RUN_DIR/last.ckpt" ] && [ -f "$RUN_DIR/data_stats.json" ]; then OVR+=(resume=true); else OVR+=(overwrite=true); fi
# NCCL: the first attempt (job 20618, g4 GPUs 4+7) hung in the first collective (both GPUs at 100% util with
# only the CUDA context resident, no progress for 16 min). PCIe peer-to-peer between these GPUs is not usable
# here, so P2P is disabled (NCCL falls back to shared memory). Communication transport only; math unchanged
# (all-reduce order may differ -> last-bit differences). A 2-rank all_reduce preflight fails fast if NCCL still hangs.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} NCCL_DEBUG=${NCCL_DEBUG:-WARN}
cat > "$RUN_DIR.nccl_check.py" <<'PY'
import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl"); r = dist.get_rank(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
t = torch.ones(1 << 20, device="cuda"); t0 = time.time(); dist.all_reduce(t); torch.cuda.synchronize()
print(f"[nccl_check] rank={r} ok sum={t[0].item()} seconds={time.time()-t0:.2f}", flush=True); dist.destroy_process_group()
PY
env -u SLURM_NTASKS -u SLURM_NTASKS_PER_NODE -u SLURM_NPROCS -u SLURM_TASKS_PER_NODE \
  timeout 180 "$(dirname "$PIE_PY")/torchrun" --standalone --nnodes=1 --nproc-per-node=2 "$RUN_DIR.nccl_check.py" \
  || { echo "[g4] NCCL preflight failed/hung"; exit 3; }
# GPU memory sampler (this job's GPUs only)
( while true; do nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed "s/^/$(date +%H:%M:%S) /" >> "$RUN_DIR.gpumem.log" 2>/dev/null || true; sleep 60; done ) &
MON=$!
t0=$(date +%s)
# torchrun sets RANK/LOCAL_RANK; hide the sbatch task count so Lightning uses the torchelastic env, not SLURM's
env -u SLURM_NTASKS -u SLURM_NTASKS_PER_NODE -u SLURM_NPROCS -u SLURM_TASKS_PER_NODE \
  "$(dirname "$PIE_PY")/torchrun" --standalone --nnodes=1 --nproc-per-node=2 --no-python \
  "$PIE_BIN" train "${OVR[@]}"
echo "[g4] train_seconds=$(( $(date +%s) - t0 ))"
for setting in unseen_ctx unseen_pert unseen_ctx_pert; do
  t1=$(date +%s)
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} "$PIE_BIN" eval experiment_name=g4_seeds/${FOLD}_seed${SEED} ckpt=best_auprc \
    run_dir="$RUN_DIR" split_path=$SPLITS/replogle_wdataset/$setting/$FOLD/test.json row_set=$setting \
    save_predictions=true overwrite=true device=cuda
  echo "[g4] eval setting=$setting seconds=$(( $(date +%s) - t1 ))"
done
kill $MON 2>/dev/null || true
echo "end=$(date -Is) total_seconds=$(( $(date +%s) - t0 ))"
