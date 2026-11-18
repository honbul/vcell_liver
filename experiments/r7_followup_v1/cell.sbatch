#!/usr/bin/env bash
#SBATCH -J r7f_d3
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 96G
#SBATCH -t 08:00:00
#SBATCH -o assets/logs/%x_%A_%a.out
# One (context, model seed) cell on ONE GPU: the chemistry backbone is refitted first, then every correction arm of the
# cell is trained on that frozen backbone as parallel processes of this job (same GPU).
# usage: sbatch --array=0-8%2 experiments/r7_followup_v1/cell.sbatch "<arms>"      arms e.g. "M G0 GR GS142 GS143 GS144"
set -euo pipefail
export R7F_THREADS=2
source experiments/r7_followup_v1/env.sh
git update-index -q --refresh >/dev/null 2>&1 || true; DIRTY=$(git status --porcelain --untracked-files=no); test -z "$DIRTY" || { echo "worktree not clean:"; echo "$DIRTY"; exit 3; }
CELLS=${R7F_CELLS:-configs/r7_followup_v1/cells.txt}
read -r CTX SEED <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CELLS")"
ARMS=${1:-"M G0 GR GS142 GS143 GS144"}
echo "start=$(date -Is) commit=$R7F_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} ctx=$CTX seed=$SEED arms=$ARMS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true
P="$TB_PY -m src.r7_followup_v1.train"
BB="${CTX}_s${SEED}_chem"
if [ "$CTX" = "orig" ]; then
  case $SEED in 42) BBARG="orig:chem/pilot_chem_lr1e-4_s42";; 43) BBARG="orig:chem/sel_chem_lr1e-4_s43";; 44) BBARG="orig:chem/sel_chem_lr1e-4_s44";; esac
else
  BBARG="$BB"
  if [ ! -f "$R7F_RUNS/chem/$BB/record.json" ]; then $P --arm chem --ctx "$CTX" --seed "$SEED" --run_id "$BB"; fi
fi
pids=(); names=()
for A in $ARMS; do
  RID="${CTX}_s${SEED}_${A}"
  case $A in
    GS*) KIND=GS; EXTRA="--null_seed ${A#GS}";;
    TDS*) KIND=TDS; EXTRA="--null_seed ${A#TDS}";;
    *) KIND=$A; EXTRA="";;
  esac
  if [ -f "$R7F_RUNS/$KIND/$RID/record.json" ]; then echo "skip finished $RID"; continue; fi
  $P --arm "$KIND" --ctx "$CTX" --seed "$SEED" --run_id "$RID" --backbone_from "$BBARG" $EXTRA > "$R7F_ASSETS/logs/run_${RID}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!); names+=("$RID")
done
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "done ${names[$i]}: $(tail -1 "$R7F_ASSETS/logs/run_${names[$i]}_${SLURM_JOB_ID}.log")"; else echo "FAILED ${names[$i]}"; tail -5 "$R7F_ASSETS/logs/run_${names[$i]}_${SLURM_JOB_ID}.log"; rc=1; fi
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 || true
echo "end=$(date -Is) rc=$rc"
exit $rc
