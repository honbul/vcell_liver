#!/usr/bin/env bash
# Laptop side: copy the follow-up code (not yet committed) to the server worktree for a test run.
set -euo pipefail
R=g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r6_followup_v1
for d in src/r6_followup_v1 experiments/r6_followup_v1 configs/r6_followup_v1 tests/r6_followup_v1; do
  rsync -a --exclude='__pycache__' "$d/" "$R/$d/"
done
