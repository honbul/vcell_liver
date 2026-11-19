#!/usr/bin/env bash
# Local machine: copy the small publishable v2 outputs from the server mirror into the repository tree.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -a --exclude '*.npy' --exclude '*.npz' --exclude '*.pt' --exclude '*.tmp' --exclude '*.pkl' --exclude '*.h5ad' --exclude '*.gz' \
  g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r8_transfer_v2/assets/public/ ./
git status --short | head -40
