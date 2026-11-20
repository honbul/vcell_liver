#!/usr/bin/env bash
# Server side: bring this worktree to the pushed branch tip (fast-forward only).
set -euo pipefail
git fetch -q origin research/r6-sensitivity-correction-v3
git merge -q --ff-only origin/research/r6-sensitivity-correction-v3
git log --oneline -1
