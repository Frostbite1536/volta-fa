#!/bin/bash
# static.sh: registers / stack (spills) of llama.cpp's D=256 ncols=64 FlashAttention kernels for
# each PR #27997 variant, compiled for sm_70 on a machine WITHOUT a V100.
# Needs the conda nvcc from `WITH_NVCC=1 bin/setup-dev.sh`. Compiles only the four ncols=64
# instance files per variant (a few minutes each); nothing is linked or run.
#   bench/pr27997/static.sh                    all variants (the list in run.sh)
#   VARIANTS="upstream pr_k64_c64" bench/pr27997/static.sh
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 2
ROOT=$PWD
T=$ROOT/tools/cuda-12.8
[ -x "$T/bin/nvcc" ] || { echo "no $T/bin/nvcc: run WITH_NVCC=1 bin/setup-dev.sh" >&2; exit 2; }
export PATH=$ROOT/.venv/bin:$PATH
LLAMA_SRC=${LLAMA_SRC:-$ROOT/third_party/llama.cpp}
W=$ROOT/third_party/pr27997-static
VARIANTS=${VARIANTS:-"upstream pr comb64 pr_comb64 pr_k64_c64"}
OUT=$ROOT/bench/results/pr27997-static-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"

# Same rows as run.sh (keep in sync).
eval "$(sed -n '/^row() {/,/^}/p' "$ROOT/bench/pr27997/run.sh")"

git -C "$LLAMA_SRC" worktree prune
[ -d "$W" ] || git -C "$LLAMA_SRC" worktree add -q --detach "$W" "$(git -C "$LLAMA_SRC" rev-parse HEAD)"
if [ ! -f "$W/build/build.ninja" ]; then
    cmake -S "$W" -B "$W/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_COMPILER="$T/bin/nvcc" -DCUDAToolkit_ROOT="$T" -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF > "$OUT/configure.log" 2>&1
fi
objs=()
for c in 64-ncols2_1 32-ncols2_2 16-ncols2_4 8-ncols2_8; do
    objs+=("ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_$c.cu.o")
done
f=$W/ggml/src/ggml-cuda/fattn-mma-f16.cuh
{ echo "llama.cpp $(git -C "$W" rev-parse --short HEAD), $("$T/bin/nvcc" --version | tail -1)"; } > "$OUT/summary.txt"
for v in $VARIANTS; do
    git -C "$W" checkout -q -- ggml
    r=$(row "$v")
    [ -n "$r" ] && R="$r" perl -0pi -e 's|(\n    // TODO tune specifically for Volta)|\n$ENV{R}\n$1|' "$f"
    echo "== $v" | tee -a "$OUT/summary.txt"
    if ! ninja -C "$W/build" "${objs[@]}" > "$OUT/$v.build.log" 2>&1; then
        echo "  build failed (see $OUT/$v.build.log)" | tee -a "$OUT/summary.txt"; continue
    fi
    for o in "${objs[@]}"; do "$T/bin/cuobjdump" -res-usage "$W/build/$o"; done \
        | grep -A1 'ILi256ELi256E' > "$OUT/$v.res-usage.txt" || true
    grep -oE 'REG:[0-9]+ STACK:[0-9]+' "$OUT/$v.res-usage.txt" | sort | uniq -c | tee -a "$OUT/summary.txt"
done
git -C "$W" checkout -q -- ggml
echo "results: $OUT/summary.txt"
