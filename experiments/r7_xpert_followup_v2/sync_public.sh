#!/usr/bin/env bash
# Run on the local machine: copy the small publishable outputs from the server mirror into the repository tree.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -a --exclude '*.npy' --exclude '*.npz' --exclude '*.pt' --exclude '*.pth' --exclude '*.db' g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r7_xpert_v2/assets/public/ ./
find manifests/r7_xpert_followup_v2 reports/r7_xpert_followup_v2 -type f -size +1M -print | sed 's/^/WARNING >1MB: /' || true
git status --short | head -40
