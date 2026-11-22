#!/usr/bin/env bash
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
if (( $# < 2 )); then
    printf 'Usage: %s TRACK-v1-JOB /absolute/batch-script [AFTEROK_JOB_IDS] [-- SCRIPT_ARGS...]\n' "$0" >&2
    exit 2
fi
job=$1
script=$2
shift 2
[[ $job =~ ^(fm|crispr|scrna|ici|combo)-v1-[a-zA-Z0-9_-]+$ && $script == /* && -r $script ]] || exit 2
dependency=()
prerequisites=''
if (( $# > 0 )) && [[ $1 != -- ]]; then
    [[ $1 =~ ^[0-9]+(:[0-9]+)*$ ]] || exit 2
    dependency=(--dependency="afterok:$1" --kill-on-invalid-dep=yes)
    prerequisites=$1
    shift
fi
if (( $# > 0 )); then
    [[ $1 == -- ]] || exit 2
    shift
fi

mkdir -p "$root/logs"
(
    flock -x 9
    queue=$(squeue --array --noheader --user "$USER" --format '%i|%j|%T|%b')
    printf '%s\n' "$queue" > "$root/logs/$job.pre-submit.squeue"
    # Count all jobs that could overlap this one, including pending work.
    # A directly named afterok prerequisite must finish before this job can run.
    count=$(awk -F'|' -v deps=":$prerequisites:" '
        $2 ~ /^(crispr|scrna|fm|ici|combo)-v1-/ && $4 ~ /gpu/ && index(deps, ":" $1 ":") == 0 { n++ }
        END { print n+0 }
    ' <<< "$queue")
    active=$(awk -F'|' -v job="$job" '$2 == job {n++} END {print n+0}' <<< "$queue")
    if (( count >= 4 || active > 0 )); then
        echo "Submission deferred. $count potentially overlapping nextgen GPU jobs and $active jobs named $job are in squeue." >&2
        exit 3
    fi
    sbatch --parsable --partition=gpu --gres=gpu:1 --job-name="$job" \
        --output="$root/logs/%x-%j.out" "${dependency[@]}" "$script" "$@"
) 9>"$root/.gpu-submit.lock"
