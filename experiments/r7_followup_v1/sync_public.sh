#!/usr/bin/env bash
# Run on the local machine: copy the small publishable outputs of the follow-up from the server mirror into the repository tree.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -a --exclude '*.npy' --exclude '*.npz' --exclude '*.pt' --exclude '*.pkl' --exclude '*.h5ad' g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r7_followup_v1/assets/public/ ./
find manifests/r7_followup_v1 reports/r7_followup_v1 predictions/r7_followup_v1 -type f -size +3M -print | sed 's/^/WARNING >3MB: /' || true
git status --short | head -40
