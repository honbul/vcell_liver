#!/usr/bin/env bash
# Rebuild the verified PHH RMA dependency for workers where its pthread setup fails.
set -euo pipefail

archive=$1
library=$2
expected=94da71e954204c6edca9899452488c0a9c5c5771c24faad8de9f29a62fcde7c3
actual=$(sha256sum "$archive" | cut -d ' ' -f 1)
[[ "$actual" == "$expected" ]]
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
tar -xf "$archive" --strip-components=1 -C "$scratch"
sed -i '1i#include <float.h>' "$scratch/src/qnorm.c"
sed -i 's/DOUBLE_EPS/DBL_EPSILON/g' "$scratch/src/qnorm.c"
R CMD INSTALL --configure-args=--disable-threading --library="$library" "$scratch"
