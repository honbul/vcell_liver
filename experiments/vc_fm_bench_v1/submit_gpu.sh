#!/usr/bin/env bash
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
[[ $# == 2 || $# == 3 ]] || { echo "Usage: $0 TRACK-v1-JOB /absolute/batch-script [AFTEROK_JOB_IDS]" >&2; exit 2; }
job=$1
script=$2
[[ $job =~ ^(fm|crispr|scrna|ici|combo)-v1-[a-zA-Z0-9_-]+$ && $script == /* && -r $script ]] || exit 2
dependency=()
if [[ $# == 3 ]]; then
    [[ $3 =~ ^[0-9]+(:[0-9]+)*$ ]] || exit 2
    dependency=(--dependency="afterok:$3" --kill-on-invalid-dep=yes)
fi

mkdir -p "$root/logs"
(
    flock -x 9
    queue=$(squeue --array --noheader --user "$USER" --format '%i|%j|%T|%b')
    printf '%s\n' "$queue" > "$root/logs/$job.pre-submit.squeue"
    # Count pending as well as running work, including all array elements.
    count=$(awk -F'|' '$2 ~ /^(crispr|scrna|fm|ici|combo)-v1-/ && $4 ~ /gpu/ {n++} END {print n+0}' <<< "$queue")
    active=$(awk -F'|' -v job="$job" '$2 == job {n++} END {print n+0}' <<< "$queue")
    if (( count >= 4 || active > 0 )); then
        echo "Submission deferred. $count nextgen GPU jobs and $active jobs named $job are in squeue." >&2
        exit 3
    fi
    sbatch --parsable --partition=gpu --gres=gpu:1 --job-name="$job" \
        --output="$root/logs/%x-%j.out" "${dependency[@]}" "$script"
) 9>"$root/.gpu-submit.lock"
