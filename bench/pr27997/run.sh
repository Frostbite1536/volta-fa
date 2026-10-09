#!/bin/bash
# run.sh: A/B the Volta D=256 FlashAttention config from llama.cpp PR #27997 against
# upstream and two untested alternatives, on a real V100. See bench/pr27997/README.md.
#
#   bench/pr27997/run.sh                         all variants, kernel-level only
#   MODEL=/path/qwen-d256.gguf bench/pr27997/run.sh   also llama-bench end to end
#   VARIANTS="upstream pr" ROUNDS=5 bench/pr27997/run.sh
#
# Each variant is a separate static build of the same llama.cpp commit, differing only in
# one row of ggml_cuda_fattn_mma_get_config_volta. All variants are built first; timing then
# runs in ROUNDS interleaved rounds (upstream, pr, ... repeated) so drift and throttling hit
# every variant alike. Results: bench/results/pr27997-<stamp>/ (summary.md has the table).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 2
ROOT=$PWD
LLAMA_SRC=${LLAMA_SRC:-$ROOT/third_party/llama.cpp}
WORK=${WORK:-$ROOT/third_party/pr27997-work}
VARIANTS=${VARIANTS:-"upstream pr comb64 pr_comb64"}
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-5}
OUT=$ROOT/bench/results/pr27997-$(date +%Y%m%d-%H%M%S)-$(hostname -s 2>/dev/null || hostname)
mkdir -p "$OUT" "$WORK"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
gpu_state() { nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active \
    --format=csv,noheader 2>/dev/null | head -1 || echo "n/a"; }

# Variants live in different worktree paths; without a base dir ccache never hits across them.
export CCACHE_BASEDIR=$WORK CCACHE_NOHASHDIR=1 CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-20G}

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[[ "$gpu" == *V100* || "${FORCE:-0}" == 1 ]] || { echo "GPU is '$gpu', not a V100 (FORCE=1 to override)" >&2; exit 2; }
ver=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
[ "$ver" = 12.8 ] || [ "${FORCE:-0}" = 1 ] || { echo "nvcc is '$ver'; CUDA 12.8 required (FORCE=1 to override)" >&2; exit 2; }

# The Volta config row each variant inserts (fields: DKQ, DV, ncols, nthreads, occupancy,
# nbatch_fa, nbatch_K2, nbatch_V2, nbatch_combine, nstages_target, Q_in_reg).
row() {
    case $1 in
        upstream)  echo "" ;;  # falls through to the Ampere row: (…, 128, 2, 32, 128, 128, 128, 2, true)
        pr)        echo "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128,  64, 128, 2, false);" ;;
        comb64)    echo "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128,  64, 2, true);" ;;
        pr_comb64) echo "    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128,  64,  64, 2, false);" ;;
        *) echo "unknown variant $1" >&2; exit 2 ;;
    esac
}

base=$(git -C "$LLAMA_SRC" rev-parse HEAD)
{ echo "llama.cpp base: $base"; echo "gpu: $gpu"; nvidia-smi --query-gpu=driver_version,clocks.max.sm,power.limit --format=csv,noheader
  nvcc --version | tail -1; echo "variants: $VARIANTS"; echo "rounds: $ROUNDS"; } > "$OUT/env.txt" 2>&1 || true

# ---- 1. prepare + build every variant; a failed variant is logged and dropped, the rest continue
built=()
git -C "$LLAMA_SRC" worktree prune   # a deleted $WORK leaves stale registrations that block "worktree add"
for v in $VARIANTS; do
    src=$WORK/$v
    [ -d "$src" ] || git -C "$LLAMA_SRC" worktree add -q --detach "$src" "$base"
    git -C "$src" checkout -q -- .
    f=$src/ggml/src/ggml-cuda/fattn-mma-f16.cuh
    r=$(row "$v")
    if [ -n "$r" ]; then
        # Insert just before the Volta fallback; grep proves the anchor exists exactly once.
        [ "$(grep -c '// TODO tune specifically for Volta' "$f")" = 1 ] || { log "anchor not unique in $f"; exit 1; }
        R="$r" perl -0pi -e 's|(\n    // TODO tune specifically for Volta)|\n$ENV{R}\n$1|' "$f"
    fi
    # Local-only perf cases: D=256 prefill (nb=512 -> ncols=64 path) at several KV depths.
    t=$src/tests/test-backend-ops.cpp
    anchor='test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776'
    [ "$(grep -cF "$anchor" "$t")" = 1 ] || { log "test anchor not unique in $t"; exit 1; }
    perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "    for (int kv : {4096, 16384, 65536}) { for (int gqa : {2, 4}) { test_cases.emplace_back(new test_flash_attn_ext(256, 256, 4, {gqa, 1}, kv, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16)); } } // pr27997 harness\n" }' "$t"
    git -C "$src" diff -- ggml > "$OUT/$v.patch"

    log "build $v"
    if ! { cmake -S "$src" -B "$src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=70 -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF > "$OUT/$v.build.log" 2>&1 &&
           cmake --build "$src/build" --target test-backend-ops llama-bench >> "$OUT/$v.build.log" 2>&1; }; then
        log "  BUILD FAILED for $v (last lines below; full log $OUT/$v.build.log)"
        tail -15 "$OUT/$v.build.log" | tee -a "$OUT/run.log"
        continue
    fi
    built+=("$v")

    # Registers / spills of the D=256 kernels actually compiled (mangled name contains ILi256ELi256E).
    { cuobjdump -res-usage "$src/build/bin/test-backend-ops" 2>/dev/null \
        | grep -A1 'flash_attn_ext_f16ILi256ELi256E' | grep -E 'Function|REG' || true; } > "$OUT/$v.res-usage.txt"

    log "correctness $v"
    "$src/build/bin/test-backend-ops" test -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=256' > "$OUT/$v.test.txt" 2>&1 || true
    { grep -E "tests passed|FAIL" "$OUT/$v.test.txt" || echo "  (no pass/fail summary found)"; } | tail -3 | tee -a "$OUT/run.log"
done
[ ${#built[@]} -gt 0 ] || { log "no variant built"; exit 1; }

# ---- 2. kernel timing, interleaved rounds
for ((i = 1; i <= ROUNDS; i++)); do
    for v in "${built[@]}"; do
        log "kernel perf $v round $i  gpu[sm,mem,temp,power,throttle]: $(gpu_state)"
        stdbuf -oL "$WORK/$v/build/bin/test-backend-ops" perf -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=4,.*nb=512' \
            --output sql > "$OUT/$v.perf.r$i.sql" 2> "$OUT/$v.perf.r$i.err" || log "  perf run failed (see .err)"
    done
done

# ---- 3. end to end (optional), interleaved rounds as well
if [ -n "${MODEL:-}" ]; then
    for ((i = 1; i <= ROUNDS; i++)); do
        for v in "${built[@]}"; do
            log "llama-bench $v round $i  gpu: $(gpu_state)"
            "$WORK/$v/build/bin/llama-bench" -m "$MODEL" -dev CUDA0 -ngl 99 -fa on -p 1024 -n 0 \
                -d 4096,16384,65536 -r "$REPS" -o jsonl > "$OUT/$v.lb.r$i.jsonl" 2> "$OUT/$v.lb.r$i.err" || log "  llama-bench failed"
        done
    done
fi

# ---- 4. summary table (median over rounds)
groups=()
for v in "${built[@]}"; do
    g=""; for ((i = 1; i <= ROUNDS; i++)); do g+="${g:+,}$OUT/$v.perf.r$i.sql"; done
    groups+=("$g")
done
labels=$(IFS=,; echo "${built[*]}")
python3 "$ROOT/bench/analyze.py" tbo "${groups[@]}" --labels "$labels" > "$OUT/summary.md" 2>&1 || log "  analyze failed (see summary.md)"
log "done: $OUT (summary.md)"
