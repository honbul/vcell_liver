#!/usr/bin/env bash
# Server side: bring the follow-up worktree onto the pushed branch tip.
# Files of the follow-up namespace that were copied here for a test run (tracked, modified) are
# restored first; untracked generated files are cleared only when byte-identical to the incoming
# commit (the rule of experiments/chemcpa_moe_v2/sync.sh). Nothing outside this worktree is touched.
set -euo pipefail
git checkout -q -- src/r6_followup_v1 experiments/r6_followup_v1 configs/r6_followup_v1 tests/r6_followup_v1 \
  reports/r6_followup_v1 manifests/r6_followup_v1 predictions/r6_followup_v1 2>/dev/null || true
exec bash experiments/chemcpa_moe_v2/sync.sh research/r6-followup-specificity-observation-v1
