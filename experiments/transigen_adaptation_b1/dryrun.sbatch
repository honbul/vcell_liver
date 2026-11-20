#!/usr/bin/env bash
#SBATCH -J tb_dryrun
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
# Wiring test of every stage up to the freeze, in a scratch area with 1-epoch source models and a
# 200-update adaptation budget. Nothing here is a result, nothing is written to the study's run or
# manifest directories, and target test X1 is not opened: the final scoring
# and table code is exercised on validation rows instead.
set -euo pipefail
source experiments/transigen_adaptation_b1/env.sh
S="$TB_ASSETS/scratch/dryrun_$SLURM_JOB_ID"
mkdir -p "$S/runs"
cp -r manifests/transigen_adaptation_b1 "$S/man"
rm -f "$S"/man/{source_lr_selection,source_selection,adapt_selection,baseline_selection,control_gate,freeze_manifest,adapt_budget,final_predictions}.json "$S"/man/adapt_search_*.tsv "$S"/man/val_scores.tsv
ln -s "$TB_RUNS/baselines_v2" "$S/runs/baselines_v2"
export TB_MAN_DIR="$S/man" TB_RUNS="$S/runs" TB_COMMIT=dryrun
PY="$TB_PY -m src.transigen_adaptation_b1"
echo '{"selected_lr": 0.001}' > "$TB_MAN_DIR/source_lr_selection.json"
for seed in 42 43 44; do $PY.train_source --lr 0.001 --seed $seed --run-id source_lr0.001_seed${seed}_v4 --max-epochs 1 | tail -3; done
$PY.select_runs source_final | tail -3
echo '{"max_updates": 200}' > "$TB_MAN_DIR/adapt_budget.json"
for t in HUH7 JHH5 JHH7; do $PY.adapt_target --target $t --phase search | tail -2; done
$PY.select_runs adapt | tail -12
for t in HUH7 JHH5 JHH7; do $PY.adapt_target --target $t --phase seeds | tail -1; $PY.adapt_target --target $t --phase curve | tail -1; done
$PY.stages val
$PY.stages control
$PY.stages freeze
$PY.select_runs ledger
export TB_REP_DIR="$S/rep"
$PY.stages dryrun_tables | tail -14
$PY.summarize --n-boot 200 | tail -40
echo "dry run complete"
