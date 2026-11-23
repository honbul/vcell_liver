#!/usr/bin/env bash
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_v2/h3_v2
cd "$root"
archive=logs/slurm-observations.txt
while :; do
  scontrol -o show job | awk '/JobName=ng2-h3-/ && /nextgen_v2\/h3_v2/' >> "$archive"
  if [[ -s reports-final/verification.json ]]; then
    exit 0
  fi
  sleep 45
done
