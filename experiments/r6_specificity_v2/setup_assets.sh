#!/usr/bin/env bash
# One-time: read-only symlinks from this track's assets/cache to the v1 caches (no copies).
set -euo pipefail
source experiments/r6_specificity_v2/env.sh
for c in halves_v1 views_v1; do
  [ -e "$R6V2_ASSETS/cache/$c" ] || ln -s "$R6F_V1/assets/cache/$c" "$R6V2_ASSETS/cache/$c"
done
ls -la "$R6V2_ASSETS/cache"
