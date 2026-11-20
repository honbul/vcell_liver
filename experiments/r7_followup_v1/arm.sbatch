#!/usr/bin/env bash
#SBATCH -J r7f_arm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem 48G
#SBATCH -t 08:00:00
#SBATCH -o assets/logs/%x_%A_%a.out
# ONE arm of one (context, model seed) cell on one GPU (operational split of cell.sbatch; same train.py call, same
# arguments, same run id convention). The backbone of the cell must already exist (fold contexts) or is the original
# chem_only checkpoint (orig context). usage: sbatch --array=0-N%T experiments/r7_followup_v1/arm.sbatch <task file>
# task file lines: "<ctx> <seed> <arm key>"   e.g. "fold0 42 TD", "orig 44 GS142"
set -euo pipefail
export R7F_THREADS=2      # same thread count per process as cell.sbatch
source experiments/r7_followup_v1/env.sh
git update-index -q --refresh >/dev/null 2>&1 || true; DIRTY=$(git status --porcelain --untracked-files=no); test -z "$DIRTY" || { echo "worktree not clean:"; echo "$DIRTY"; exit 3; }
read -r CTX SEED A <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")"
echo "start=$(date -Is) commit=$R7F_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} ctx=$CTX seed=$SEED arm=$A"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true
P="$TB_PY -m src.r7_followup_v1.train"
if [ "$CTX" = "orig" ]; then
  case $SEED in 42) BBARG="orig:chem/pilot_chem_lr1e-4_s42";; 43) BBARG="orig:chem/sel_chem_lr1e-4_s43";; 44) BBARG="orig:chem/sel_chem_lr1e-4_s44";; esac
else
  BBARG="${CTX}_s${SEED}_chem"
  test -f "$R7F_RUNS/chem/$BBARG/record.json" || { echo "backbone $BBARG missing"; exit 4; }
fi
RID="${CTX}_s${SEED}_${A}"
case $A in
  GS*) KIND=GS; EXTRA="--null_seed ${A#GS}";;
  TDS*) KIND=TDS; EXTRA="--null_seed ${A#TDS}";;
  *) KIND=$A; EXTRA="";;
esac
if [ -f "$R7F_RUNS/$KIND/$RID/record.json" ]; then echo "skip finished $RID"; exit 0; fi
$P --arm "$KIND" --ctx "$CTX" --seed "$SEED" --run_id "$RID" --backbone_from "$BBARG" $EXTRA > "$R7F_ASSETS/logs/run_${RID}_${SLURM_JOB_ID}.log" 2>&1
echo "done $RID: $(tail -1 "$R7F_ASSETS/logs/run_${RID}_${SLURM_JOB_ID}.log")"
echo "end=$(date -Is)"
