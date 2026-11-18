#!/usr/bin/env bash
# Laptop side: copy the small generated files (manifests, tables, figures) from the server worktree.
# Arrays, checkpoints and caches never leave the server.
set -euo pipefail
R=g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r6_followup_v1
for d in manifests/r6_followup_v1 reports/r6_followup_v1 predictions/r6_followup_v1; do
  mkdir -p "$d"
  rsync -a --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' --exclude='*.h5ad' --exclude='*.pkl' "$R/$d/" "$d/" 2>/dev/null || true
done
git status --short | head -60
