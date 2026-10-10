#!/bin/bash
# run.sh: A/B the P-6 small-batch GQA patches (docs/design/P6-small-batch-gqa.md) on a real V100.
#
#   bench/p6/run.sh                                   kernel level: correctness + interleaved perf rounds
#   MODELS="/m/qwen.gguf /m/llama.gguf" bench/p6/run.sh   also llama-bench small batches at depth
#   VARIANTS="upstream fix" ROUNDS=5 bench/p6/run.sh
#
# Variants (same llama.cpp commit, patches from third_party/patches/):
#   upstream  no patch
#   fix       0001 (one Volta heads-per-tile rule for selection + dispatch, 16-column guard, crash-class tests)
#   p6        0001 + 0002 (padded heads per tile for non-power-of-two GQA at small batch, sink guard, tests)
#   p6cfg     0001 + 0002 + 0003 (Volta D=256 configs for ncols 32/64: no spills, 2 blocks/SM)
# Results: bench/results/p6-<stamp>/ (summary.md: median kernel table).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 2
ROOT=$PWD
LLAMA_SRC=${LLAMA_SRC:-$ROOT/third_party/llama.cpp}
WORK=${WORK:-$ROOT/third_party/p6-ab}
VARIANTS=${VARIANTS:-"upstream fix p6 p6cfg"}
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-5}
PATCHES=$ROOT/third_party/patches
OUT=$ROOT/bench/results/p6-$(date +%Y%m%d-%H%M%S)-$(hostname -s 2>/dev/null || hostname)
mkdir -p "$OUT" "$WORK"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
gpu_state() { nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active \
    --format=csv,noheader 2>/dev/null | head -1 || echo "n/a"; }
export CCACHE_BASEDIR=$WORK CCACHE_NOHASHDIR=1 CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-20G}

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[[ "$gpu" == *V100* || "${FORCE:-0}" == 1 ]] || { echo "GPU is '$gpu', not a V100 (FORCE=1 to override)" >&2; exit 2; }
ver=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
[ "$ver" = 12.8 ] || [ "${FORCE:-0}" = 1 ] || { echo "nvcc is '$ver'; CUDA 12.8 required (FORCE=1 to override)" >&2; exit 2; }

patches_for() {
    case $1 in
        upstream) ;;
        fix)   ls "$PATCHES"/0001-*.patch ;;
        p6)    ls "$PATCHES"/0001-*.patch "$PATCHES"/0002-*.patch ;;
        p6cfg) ls "$PATCHES"/0001-*.patch "$PATCHES"/0002-*.patch "$PATCHES"/0003-*.patch ;;
        *) echo "unknown variant $1" >&2; exit 2 ;;
    esac
}

base=$(git -C "$LLAMA_SRC" rev-parse HEAD)
{ echo "llama.cpp base: $base"; echo "gpu: $gpu"; nvidia-smi --query-gpu=driver_version,clocks.max.sm,power.limit --format=csv,noheader
  nvcc --version | tail -1; echo "variants: $VARIANTS"; echo "rounds: $ROUNDS"; } > "$OUT/env.txt" 2>&1 || true

# Local-only perf cases: small batches at depth for power-of-two and non-power-of-two GQA.
PERF_LINE='    for (int hs : {128, 256}) { for (int gqa : {3, 4, 5, 6, 7, 8}) { for (int kv : {4096, 32768}) { for (int nb : {1, 2, 4, 8, 16}) { test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 4, {gqa, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16)); } } } } // p6 harness'

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
    L="$PERF_LINE" perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "$ENV{L}\n" }' "$t"
    git -C "$src" diff > "$OUT/$v.patch"

    log "build $v"
    if ! { cmake -S "$src" -B "$src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=70 -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF > "$OUT/$v.build.log" 2>&1 &&
           cmake --build "$src/build" --target test-backend-ops llama-bench >> "$OUT/$v.build.log" 2>&1; }; then
        log "  BUILD FAILED for $v"; tail -15 "$OUT/$v.build.log" | tee -a "$OUT/run.log"; continue
    fi
    built+=("$v")

    # Full FA correctness suite. Upstream is expected to abort on the Volta 16-column crash (§4 of the findings).
    log "correctness $v (full FLASH_ATTN_EXT)"
    "$src/build/bin/test-backend-ops" test -b CUDA0 -o FLASH_ATTN_EXT > "$OUT/$v.test.txt" 2>&1 && rc=0 || rc=$?
    log "  rc=$rc  OK=$(grep -c 'OK' "$OUT/$v.test.txt")  FAIL=$(grep -c 'FAIL' "$OUT/$v.test.txt")"
done
[ ${#built[@]} -gt 0 ] || { log "no variant built"; exit 1; }

# ---- 2. kernel timing, interleaved rounds (small-batch cases only)
for ((i = 1; i <= ROUNDS; i++)); do
    for v in "${built[@]}"; do
        log "kernel perf $v round $i  gpu[sm,mem,temp,power,throttle]: $(gpu_state)"
        stdbuf -oL "$WORK/$v/build/bin/test-backend-ops" perf -b CUDA0 -o FLASH_ATTN_EXT -p 'nh=4,nr23=\[[345678],1\],kv=(4096|32768),nb=(1|2|4|8|16),' \
            --output sql > "$OUT/$v.perf.r$i.sql" 2> "$OUT/$v.perf.r$i.err" || log "  perf run failed (see .err)"
    done
done

# ---- 3. end to end (optional): small batches at depth, interleaved
for m in ${MODELS:-}; do
    n=$(basename "$m" .gguf)
    for ((i = 1; i <= ROUNDS; i++)); do
        for v in "${built[@]}"; do
            log "llama-bench $n $v round $i  gpu: $(gpu_state)"
            "$WORK/$v/build/bin/llama-bench" -m "$m" -dev CUDA0 -ngl 99 -fa on -p 1,2,4,8,16 -n 0 -d 4096,32768 \
                -r "$REPS" -o jsonl > "$OUT/$v.$n.lb.r$i.jsonl" 2> "$OUT/$v.$n.lb.r$i.err" || log "  llama-bench failed"
        done
    done
done

# ---- 4. summary
groups=()
for v in "${built[@]}"; do
    g=""; for ((i = 1; i <= ROUNDS; i++)); do g+="${g:+,}$OUT/$v.perf.r$i.sql"; done
    groups+=("$g")
done
labels=$(IFS=,; echo "${built[*]}")
python3 "$ROOT/bench/analyze.py" tbo "${groups[@]}" --labels "$labels" > "$OUT/summary.md" 2>&1 || log "  analyze failed (see summary.md)"
log "done: $OUT (summary.md)"
