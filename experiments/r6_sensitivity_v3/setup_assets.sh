#!/usr/bin/env bash
# One-time: read-only symlink from this track's assets/cache to the v1 treated-well cache (no copy).
set -euo pipefail
source experiments/r6_sensitivity_v3/env.sh
[ -e "$R6V3_ASSETS/cache/halves_v1" ] || ln -s "$R6F_V1/assets/cache/halves_v1" "$R6V3_ASSETS/cache/halves_v1"
ls -la "$R6V3_ASSETS/cache"
