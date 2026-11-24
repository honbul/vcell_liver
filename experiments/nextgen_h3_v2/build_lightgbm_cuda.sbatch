#!/usr/bin/env bash
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_v2/h3_v2
prefix="$root/deps/lightgbm-cuda-4.6.0"
source_digest=cb1c59720eb569389c0ba74d14f52351b573af489f230032a1c9f314f8bab7fe
printf '%s  %s\n' "$source_digest" "$prefix/lightgbm-4.6.0.tar.gz" | sha256sum -c -
test ! -e "$prefix/runtime"
cmake_tool="$prefix/toolchain/cmake/data/bin/cmake"
test -x "$cmake_tool"

cuda_compiler=$(command -v nvcc || true)
for candidate in /usr/local/cuda-12.3/bin/nvcc /usr/local/cuda-12.1/bin/nvcc; do
  if [[ -x $candidate ]]; then
    cuda_compiler=$candidate
    break
  fi
done
test -x "$cuda_compiler"
build=$(mktemp -d "$prefix/build-sm86-XXXXXX")
"$cuda_compiler" --version
"$cmake_tool" --version
"$cmake_tool" -S "$prefix/source" -B "$build" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$cuda_compiler" -DCMAKE_CUDA_ARCHITECTURES=86
"$cmake_tool" --build "$build" --target _lightgbm --parallel "${SLURM_CPUS_PER_TASK:-8}"
library=$(find "$build" "$prefix/source" -name lib_lightgbm.so -type f -print -quit)
test -s "$library"

staged=$(mktemp -d "$prefix/runtime-XXXXXX")
cp -a "$root/deps/lightgbm" "$staged/lightgbm"
install -m 755 "$library" "$staged/lightgbm/lib/lib_lightgbm.so"
sha256sum "$staged/lightgbm/lib/lib_lightgbm.so" | cut -d ' ' -f 1 > "$staged/library.sha256"
printf '%s\t%s\t%s\t%s\n' '4.6.0' 'sm86' "$source_digest" "$cuda_compiler" > "$staged/build.tsv"
mv -T "$staged" "$prefix/runtime"
printf '%s\n' 'Published isolated CUDA LightGBM 4.6.0 runtime'
