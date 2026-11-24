#!/usr/bin/env bash
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_v2/h3_v2
prefix="$root/deps/lightgbm-cuda-4.6.0"
source_digest=cb1c59720eb569389c0ba74d14f52351b573af489f230032a1c9f314f8bab7fe
printf '%s  %s\n' "$source_digest" "$prefix/lightgbm-4.6.0.tar.gz" | sha256sum -c -
test ! -e "$prefix/runtime"
cmake_tool="$prefix/toolchain/cmake/data/bin/cmake"
cuda_toolkit="$prefix/cuda-12.1.105"
cuda_compiler="$cuda_toolkit/bin/nvcc"
build=$(mktemp -d "$prefix/build-sm86-XXXXXX")
source="$build/source"
mkdir "$source"
tar -xzf "$prefix/lightgbm-4.6.0.tar.gz" --strip-components=1 -C "$source"

# LightGBM replaces CMAKE_CUDA_ARCHITECTURES on its targets. Compile only the G3 architecture.
sed -i '/message(STATUS "CUDA_ARCHITECTURES: ${CUDA_ARCHS}")/i\    set(CUDA_ARCHS "86-real")' "$source/CMakeLists.txt"
rg -Fq 'set(CUDA_ARCHS "86-real")' "$source/CMakeLists.txt"
"$cuda_compiler" --version
"$cmake_tool" --version
"$cmake_tool" -S "$source" -B "$build/objects" -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=OFF -DUSE_CUDA=ON \
  -DCUDAToolkit_ROOT="$cuda_toolkit" -DCMAKE_CUDA_COMPILER="$cuda_compiler" -DCMAKE_CUDA_RUNTIME_LIBRARY=Static
"$cmake_tool" --build "$build/objects" --target _lightgbm --parallel "${SLURM_CPUS_PER_TASK:-8}"
library="$source/lib_lightgbm.so"
test -s "$library"

staged=$(mktemp -d "$prefix/runtime-XXXXXX")
cp -a "$root/deps/lightgbm" "$staged/lightgbm"
install -m 755 "$library" "$staged/lightgbm/lib/lib_lightgbm.so"
sha256sum "$staged/lightgbm/lib/lib_lightgbm.so" | cut -d ' ' -f 1 > "$staged/library.sha256"
printf '%s\t%s\t%s\t%s\n' '4.6.0' 'sm86' "$source_digest" "$cuda_compiler" > "$staged/build.tsv"
mv -T "$staged" "$prefix/runtime"
printf '%s\n' 'Published isolated CUDA LightGBM 4.6.0 runtime'
