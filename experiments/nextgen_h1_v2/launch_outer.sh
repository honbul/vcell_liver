#!/usr/bin/env bash
# Run on G1 with this script and its Python modules in an immutable code snapshot.
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_v2/h1_v2
code=$1
cpu_boundary=$2
neural_boundary=$3
run="$code/experiments/nextgen_h1_v2/run.sbatch"
ledger="$root/outer_jobs.tsv"
if [[ -e "$ledger" || -e "$root/outer_score_claim.json" ]]; then
    exit 2
fi
(
    set -o noclobber
    printf 'key\tjob_id\n' > "$ledger"
)
submit() {
    key=$1
    shift
    job=$(sbatch --parsable --chdir="$root" --job-name="ng2-h1-$key" --output="$root/logs/%A_%a-$key.log" "$@")
    printf '%s\t%s\n' "$key" "$job" >> "$ledger"
    printf '%s' "$job"
}
dependency="afterok:$cpu_boundary:$neural_boundary"
cpu="$root/round23-cpu-source"
kernel="$root/round23-kernel"
final="$root/round23-closed"
freeze="$root/frozen-a"
gate="$root/frozen_receipt.json"
primary="$root/primary-a"
p_gate="$root/prediction_receipt.json"
score_dir="$root/score-a"
collect=$(submit collect-inner --dependency="$dependency" --time=00:20:00 "$run" "$code" collect inner     "$root/round1-a" "$cpu" "$kernel" "$final" "$root/inner-aggregates")
cal=$(submit fit-outer --dependency="$dependency" --array=0-15%8 --time=02:00:00 "$run" "$code" freeze fit     "$cpu" "$kernel" "$freeze" --alternative "$final")
solid=$(submit seal-models --dependency="afterok:$cal" --time=00:15:00 "$run" "$code" freeze seal "$freeze" "$gate")
fore=$(submit predict --dependency="afterok:$solid" --array=0-15%8 --time=00:45:00 "$run" "$code" predict run     "$freeze" "$gate" "$primary")
pseal=$(submit seal-predictions --dependency="afterok:$fore" --time=00:10:00 "$run" "$code" predict seal     "$primary" "$gate" "$p_gate")
replay=$(submit replay-primary --dependency="afterok:$pseal" --cpus-per-task=6 --time=00:50:00 "$run" "$code" replay     "$freeze" "$primary" "$root/no-aux-before-score" "$root/replay-primary" "$root/primary_replay_receipt.json")
dry=$(submit prelabel --dependency="afterok:$replay:$collect" --time=00:35:00 "$run" "$code" prelabel     "$freeze" "$gate" "$primary" "$p_gate" "$root/primary_replay_receipt.json" "$root/prelabel_receipt.json")
scored=$(submit score --dependency="afterok:$dry" --cpus-per-task=18 --mem=56G --time=03:30:00 "$run" "$code" score     "$freeze" "$gate" "$primary" "$p_gate" "$score_dir" --prelabel "$root/prelabel_receipt.json")
complete=$(submit replay-final --dependency="afterok:$scored" --cpus-per-task=6 --time=00:55:00 "$run" "$code" replay     "$freeze" "$primary" "$score_dir/aux" "$root/replay-final" "$root/replay_receipt.json")
cat "$ledger"
