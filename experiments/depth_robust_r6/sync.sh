#!/usr/bin/env bash
# Server side: fast-forward the worktree; clears only untracked generated files that are
# byte-identical to the incoming commit (same rule as the earlier studies' sync.sh).
set -euo pipefail
exec bash experiments/chemcpa_moe_v2/sync.sh research/r6-depth-robust-response
