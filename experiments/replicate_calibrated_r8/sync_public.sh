#!/usr/bin/env bash
# Run on the local machine: copy the small publishable outputs from the server mirror into the repository tree.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -a --exclude '*.npy' --exclude '*.npz' --exclude '*.pt' --exclude '*.tmp' g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_replicate_calibrated_r8/assets/public/ ./
git status --short | head -40
