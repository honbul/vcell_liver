#!/usr/bin/env bash
# Fast-forward the server worktree; clears only untracked generated files that are byte-identical
# to the incoming commit (same rule as the prior study's sync.sh).
set -euo pipefail
exec bash experiments/chemcpa_moe_v2/sync.sh research/hcc-anchored-distribution-r3
