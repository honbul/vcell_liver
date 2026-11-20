#!/usr/bin/env bash
# Laptop side: copy the small generated files (manifests, report tables) from the server worktree
# so they can be committed here. Arrays, checkpoints and caches never leave the server.
set -euo pipefail
R=g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_depth_robust_r6
for d in manifests/depth_robust_r6 reports/depth_robust_r6 predictions/depth_robust_r6; do
  mkdir -p "$d"
  rsync -a --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' --exclude='*.h5ad' --exclude='*.pkl' "$R/$d/" "$d/" 2>/dev/null || true
done
git status --short | head -40
