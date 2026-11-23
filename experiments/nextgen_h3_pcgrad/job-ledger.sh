#!/usr/bin/env bash
set -euo pipefail
# Run inside the H3 G1 root. It records only the explicitly listed H3 jobs.
ledger=logs/slurm-observations.txt
for id in "$@"; do
    if [[ ! $id =~ ^[0-9]+$ ]]; then
        echo "Expected a numeric Slurm job ID" >&2
        exit 2
    fi
    scontrol -o show job "$id" 2>/dev/null | awk '/JobName=ng-h3-/' >> "$ledger" || true
done
