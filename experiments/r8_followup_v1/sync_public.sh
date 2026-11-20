#!/usr/bin/env bash
# Local machine: copy the small publishable follow-up outputs from the server mirror into the repository tree.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -a --exclude '*.npy' --exclude '*.npz' --exclude '*.pt' --exclude '*.tmp' --exclude '*.pkl' --exclude '*.h5ad' \
  g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r8_followup_v1/assets/public/ ./
git status --short | head -40
