#!/bin/bash
# run.sh: A/B the volta-fa llama.cpp patch series (third_party/patches/) on a real V100.
#
#   bench/patches/run.sh                                      correctness + kernel perf + dequant microbench
#   MODELS="/m/qwen.gguf /m/llama.gguf" bench/patches/run.sh  also llama-bench (small batches, quantised-KV decode)
#   VARIANTS="upstream qkv" ROUNDS=5 bench/patches/run.sh
#   STEPS="build lb" MODELS="..." bench/patches/run.sh        later run: reuse builds, llama-bench only
#
# Variants are cumulative, on the same llama.cpp commit (bench/LLAMA_COMMIT):
#   upstream  no patch
#   fix       0001       one Volta heads-per-tile rule for selection + dispatch (crash fix), tests     P-6
#   p6        0001-0002  padded heads per tile for non-power-of-two GQA at 3-16 tokens, sink guard    P-6
#   p6cfg     0001-0003  Volta D=256 configs for ncols 32/64 (no spills, 2 blocks/SM)                 P-6
#   qkv       0001-0004  Volta single-token decode with quantised/BF16 K/V uses the VEC kernel        P-8
#   conv      0001-0005  K/V-to-F16 conversion: several short rows per block                          P-8
#   conv8     0001-0006  vectorised q4_0/q8_0 conversion (8 values per thread)                        P-8
#   vecpack   0001-0007  VEC packs 2/4 Q heads per block for single-token decode with quantised K     P-8
#   qmm       0001-0008  Volta tensor-core matmul for Q4_K/Q5_K/Q6_K/IQ4_XS/Q8_0 at 5-16 tokens      P-9
# Designs: docs/design/P6-small-batch-gqa.md, docs/design/P8-quantised-kv-decode.md, docs/design/P8-change4-packed-vec.md, docs/design/P9-small-batch-qmatmul.md.
# Results: bench/results/patches-<stamp>/ (summary.md: kernel table, llama-bench tables, microbench).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 2
ROOT=$PWD
LLAMA_SRC=${LLAMA_SRC:-$ROOT/third_party/llama.cpp}
WORK=${WORK:-$ROOT/third_party/patches-ab}
VARIANTS=${VARIANTS:-"upstream fix p6 p6cfg qkv conv conv8 vecpack qmm"}
LB_P6_VARIANTS=${LB_P6_VARIANTS:-"upstream p6 p6cfg conv8"}   # llama-bench small batches at depth (f16 KV)
LB_QKV_VARIANTS=${LB_QKV_VARIANTS:-"upstream p6cfg qkv conv8 vecpack"} # llama-bench quantised-KV decode + pp8
MULMAT_VARIANTS=${MULMAT_VARIANTS:-"upstream qmm"}           # MUL_MAT correctness (0005/0006 touch its conversions)
STEPS=${STEPS:-"micro build test perf lb"}                     # build is incremental after the first run
ROUNDS=${ROUNDS:-3}
LB_ROUNDS=${LB_ROUNDS:-2}
REPS=${REPS:-3}
PATCHES=$ROOT/third_party/patches
OUT=$ROOT/bench/results/patches-$(date +%Y%m%d-%H%M%S)-$(hostname -s 2>/dev/null || hostname)
mkdir -p "$OUT" "$WORK"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
gpu_state() { nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active \
    --format=csv,noheader 2>/dev/null | head -1 || echo "n/a"; }
in_list() { local x=$1; shift; for y in "$@"; do [ "$x" = "$y" ] && return 0; done; return 1; }
export CCACHE_BASEDIR=$WORK CCACHE_NOHASHDIR=1 CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-20G}

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[[ "$gpu" == *V100* || "${FORCE:-0}" == 1 ]] || { echo "GPU is '$gpu', not a V100 (FORCE=1 to override)" >&2; exit 2; }
ver=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
[ "$ver" = 12.8 ] || [ "${FORCE:-0}" = 1 ] || { echo "nvcc is '$ver'; CUDA 12.8 required (FORCE=1 to override)" >&2; exit 2; }

npatch() {
    case $1 in
        upstream) echo 0 ;; fix) echo 1 ;; p6) echo 2 ;; p6cfg) echo 3 ;; qkv) echo 4 ;; conv) echo 5 ;; conv8) echo 6 ;; vecpack) echo 7 ;; qmm) echo 8 ;;
        *) echo "unknown variant $1" >&2; return 2 ;;
    esac
}
patches_for() {
    local n k
    n=$(npatch "$1") || exit 2
    for ((k = 1; k <= n; k++)); do ls "$PATCHES"/"$(printf '%04d' "$k")"-*.patch; done
}
for v in $VARIANTS; do patches_for "$v" > /dev/null; done   # fail early on a typo or a missing patch

base=$(git -C "$LLAMA_SRC" rev-parse HEAD)
{ echo "llama.cpp base: $base (bench/LLAMA_COMMIT: $(cat "$ROOT/bench/LLAMA_COMMIT" 2>/dev/null))"; echo "gpu: $gpu"
  nvidia-smi --query-gpu=driver_version,clocks.max.sm,power.limit --format=csv,noheader
  nvcc --version | tail -1; echo "variants: $VARIANTS"; echo "rounds: $ROUNDS (llama-bench: $LB_ROUNDS x $REPS reps)"
  echo "patches:"; sha256sum "$PATCHES"/*.patch; } > "$OUT/env.txt" 2>&1 || true

# shellcheck disable=SC2086
step() { in_list "$1" $STEPS; }

# ---- 0. conversion microbenchmark (P-8 Changes 2-3; seconds, no llama.cpp build needed)
if step micro; then
log "dequant microbench"
if nvcc -O3 -std=c++17 -arch=sm_70 bench/micro/dequant_nc.cu -o "$WORK/dequant_nc" > "$OUT/dequant_nc.build.txt" 2>&1; then
    "$WORK/dequant_nc" 50 > "$OUT/dequant_nc.txt" 2>&1 || log "  microbench failed (see dequant_nc.txt)"
else
    log "  microbench build failed (see dequant_nc.build.txt)"
fi
fi

# Local-only perf cases: small batches at depth (P-6, F16) and quantised K/V at 1-8 tokens (P-8).
PERF_P6='    for (int hs : {128, 256}) { for (int gqa : {3, 4, 5, 6, 7, 8}) { for (int kv : {4096, 32768}) { for (int nb : {1, 2, 4, 8, 16}) { test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 4, {gqa, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16)); } } } } // patches harness (P-6)'
PERF_P8='    for (int hs : {128, 256}) { for (int gqa : {4, 6, 8}) { for (int kv : {4096, 32768}) { for (int nb : {1, 2, 8}) { for (ggml_type t : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) { test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 4, {gqa, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, t, t)); } } } } } // patches harness (P-8)'
PERF_FILTER='nh=4,nr23=\[[345678],1\],kv=(4096|32768),nb=(1|2|4|8|16),'

# ---- 1. prepare + build + correctness for every variant
built=()
git -C "$LLAMA_SRC" worktree prune
for v in $VARIANTS; do
    src=$WORK/$v
    [ -d "$src" ] || git -C "$LLAMA_SRC" worktree add -q --detach "$src" "$base"
    git -C "$src" checkout -q -- .
    for p in $(patches_for "$v"); do git -C "$src" apply "$p" || { log "patch $p does not apply for $v"; exit 1; }; done
    t=$src/tests/test-backend-ops.cpp
    anchor='test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776'
    [ "$(grep -cF "$anchor" "$t")" = 1 ] || { log "perf anchor not unique in $t"; exit 1; }
    L="$PERF_P6
$PERF_P8" perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "$ENV{L}\n" }' "$t"
    git -C "$src" diff > "$OUT/$v.patch"

    log "build $v"
    if ! { cmake -S "$src" -B "$src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=70 -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF > "$OUT/$v.build.log" 2>&1 &&
           cmake --build "$src/build" --target test-backend-ops llama-bench >> "$OUT/$v.build.log" 2>&1; }; then
        log "  BUILD FAILED for $v"; tail -15 "$OUT/$v.build.log" | tee -a "$OUT/run.log"; continue
    fi
    built+=("$v")
    step test || continue

    # Full FA correctness suite. Upstream is expected to abort on the Volta 16-column crash (§4 of the findings).
    log "correctness $v (full FLASH_ATTN_EXT)"
    "$src/build/bin/test-backend-ops" test -b CUDA0 -o FLASH_ATTN_EXT > "$OUT/$v.test.txt" 2>&1 && rc=0 || rc=$?
    log "  rc=$rc  OK=$(grep -c 'OK' "$OUT/$v.test.txt" || true)  FAIL=$(grep -c 'FAIL' "$OUT/$v.test.txt" || true)"
    # shellcheck disable=SC2086  # variant lists are space-separated words
    if in_list "$v" $MULMAT_VARIANTS; then
        log "correctness $v (MUL_MAT: non-contiguous quantised conversions)"
        "$src/build/bin/test-backend-ops" test -b CUDA0 -o MUL_MAT > "$OUT/$v.test-mulmat.txt" 2>&1 && rc=0 || rc=$?
        log "  rc=$rc  OK=$(grep -c 'OK' "$OUT/$v.test-mulmat.txt" || true)  FAIL=$(grep -c 'FAIL' "$OUT/$v.test-mulmat.txt" || true)"
    fi
done
[ ${#built[@]} -gt 0 ] || { log "no variant built"; exit 1; }

# ---- 2. kernel timing, interleaved rounds (small-batch F16 and quantised cases)
step perf || ROUNDS=0
for ((i = 1; i <= ROUNDS; i++)); do
    for v in "${built[@]}"; do
        log "kernel perf $v round $i  gpu[sm,mem,temp,power,throttle]: $(gpu_state)"
        stdbuf -oL "$WORK/$v/build/bin/test-backend-ops" perf -b CUDA0 -o FLASH_ATTN_EXT -p "$PERF_FILTER" \
            --output sql > "$OUT/$v.perf.r$i.sql" 2> "$OUT/$v.perf.r$i.err" || log "  perf run failed (see .err)"
    done
done

# ---- 3. end to end (optional), interleaved rounds
lb_run() {  # lb_run <tag> <variants> <llama-bench args...>
    local tag=$1 vs=$2 m n i v; shift 2
    for m in ${MODELS:-}; do
        n=$(basename "$m" .gguf)
        for ((i = 1; i <= LB_ROUNDS; i++)); do
            for v in "${built[@]}"; do
                # shellcheck disable=SC2086
                in_list "$v" $vs || continue
                log "llama-bench $tag $n $v round $i  gpu: $(gpu_state)"
                "$WORK/$v/build/bin/llama-bench" -m "$m" -dev CUDA0 -ngl 99 -fa on -r "$REPS" -o jsonl "$@" \
                    > "$OUT/$v.$n.$tag.r$i.jsonl" 2> "$OUT/$v.$n.$tag.r$i.err" || log "  llama-bench failed"
            done
        done
    done
}
step lb || MODELS=""
lb_run small "$LB_P6_VARIANTS" -p 1,2,4,8,16 -n 0 -d 4096,32768
for kv in q8_0 q4_0; do
    lb_run "kv$kv" "$LB_QKV_VARIANTS" -ctk "$kv" -ctv "$kv" -p 8 -n 64 -d 16384
done

# ---- 4. summary
{
    echo "# Patch series A/B ($(date +%F), $gpu)"; echo
    if step test; then
        echo "## Correctness"; echo; grep -E '^\[.*\] (correctness|  rc=)' "$OUT/run.log" | sed 's/^/    /' || true; echo
    fi
    if [ -s "$OUT/dequant_nc.txt" ]; then
        echo "## Conversion microbenchmark (bench/micro/dequant_nc.cu)"; echo; sed 's/^/    /' "$OUT/dequant_nc.txt"; echo
    fi
    [ "$ROUNDS" -gt 0 ] && echo "## Kernel timing (median of $ROUNDS interleaved rounds)" && echo
    groups=()
    for v in "${built[@]}"; do
        g=""; for ((i = 1; i <= ROUNDS; i++)); do g+="${g:+,}$OUT/$v.perf.r$i.sql"; done
        groups+=("$g")
    done
    [ "$ROUNDS" -gt 0 ] && { python3 "$ROOT/bench/analyze.py" tbo "${groups[@]}" --labels "$(IFS=,; echo "${built[*]}")" 2>&1 || echo "analyze tbo failed"; }
    for m in ${MODELS:-}; do
        n=$(basename "$m" .gguf)
        for tag in small kvq8_0 kvq4_0; do
            vs=(); gs=()
            for v in "${built[@]}"; do
                g=""; for ((i = 1; i <= LB_ROUNDS; i++)); do
                    f=$OUT/$v.$n.$tag.r$i.jsonl; [ -s "$f" ] && g+="${g:+,}$f"
                done
                [ -n "$g" ] && { vs+=("$v"); gs+=("$g"); }
            done
            [ ${#gs[@]} -gt 0 ] || continue
            echo; echo "## llama-bench $n $tag (median of rounds)"; echo
            python3 "$ROOT/bench/analyze.py" lb "${gs[@]}" --labels "$(IFS=,; echo "${vs[*]}")" 2>&1 || echo "analyze lb failed"
        done
    done
} > "$OUT/summary.md"
log "done: $OUT (summary.md)"
