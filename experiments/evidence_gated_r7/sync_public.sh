#!/usr/bin/env bash
# Run on the local machine: copy the small publishable outputs from the server mirror into the repository tree.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -a --exclude '*.npy' --exclude '*.npz' --exclude '*.pt' --exclude '*.db' --exclude '*.gz' g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_evidence_gated_r7/assets/public/ ./
find manifests/evidence_gated_r7 reports/evidence_gated_r7 tutorials/evidence_gated_r7 -type f -size +3M -print | sed 's/^/WARNING >3MB: /' || true
git status --short | head -40
