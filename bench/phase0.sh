#!/bin/bash
# phase0.sh: Phase 0 baseline on the V100 box (docs/ROADMAP.md). Run from the repo root.
#   bench/phase0.sh                 full run: env, build, FA support/correctness/perf, llama-bench grid
#   STEPS="env perf" bench/phase0.sh   run only some steps (env build support test perf bench multi)
# Models: one GGUF path per line in bench/models.txt (see bench/models.example.txt).
# Output: bench/results/phase0-<date>-<host>/ (commit the small text/csv/jsonl files, never models).
set -euo pipefail
cd "$(dirname "$0")/.." || exit 2
LLAMA=${LLAMA:-third_party/llama.cpp}
BUILD=$LLAMA/build-sm70
STEPS=${STEPS:-"env build support test perf bench multi"}
REPS=${REPS:-3}
OUT=bench/results/phase0-$(date +%Y%m%d-%H%M)-$(hostname -s)
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }

# Refuse to produce "baseline" numbers on the wrong hardware or toolchain (INV-PERF-001, INV-ARCH-003).
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[[ "$gpu" == *V100* || "${FORCE:-0}" == 1 ]] || { echo "GPU is '$gpu', not a V100 (FORCE=1 to override)" >&2; exit 2; }
ver=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
[ "$ver" = 12 ] || { echo "nvcc major version is '$ver'; CUDA 12.x required" >&2; exit 2; }
ngpu=$(nvidia-smi -L | grep -c V100 || true)

if has env; then
  log "env"
  { echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "volta-fa: $(git rev-parse --short HEAD)"
    echo "llama.cpp: $(git -C "$LLAMA" rev-parse HEAD)"; nvcc --version | tail -2; uname -r
    nvidia-smi --query-gpu=index,name,driver_version,pcie.link.gen.max,pcie.link.width.max,clocks.max.sm,clocks.max.mem,power.limit,memory.total --format=csv
    echo; nvidia-smi topo -m; echo; nvidia-smi topo -p2p r 2>/dev/null || true
    command -v ncu >/dev/null && ncu --version | tail -1; command -v nsys >/dev/null && nsys --version
  } > "$OUT/env.txt" 2>&1
fi

if has build; then
  log "build llama.cpp for sm_70"
  cmake -S "$LLAMA" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 -DLLAMA_CURL=OFF > "$OUT/build.log" 2>&1
  cmake --build "$BUILD" --target llama-bench test-backend-ops llama-cli >> "$OUT/build.log" 2>&1
fi
TBO=$BUILD/bin/test-backend-ops; LB=$BUILD/bin/llama-bench

# Which FLASH_ATTN_EXT cases the CUDA backend even supports on cc 7.0 (model-compatibility gaps).
has support && { log "FA support matrix"; "$TBO" support -o FLASH_ATTN_EXT -b CUDA0 > "$OUT/fa-support.txt" 2>&1 || true; }
# Correctness vs CPU (a Volta-path bug is itself a finding).
has test && { log "FA correctness"; "$TBO" test -o FLASH_ATTN_EXT -b CUDA0 > "$OUT/fa-test.txt" 2>&1 || true; grep -E "FAIL|passed" "$OUT/fa-test.txt" | tail -3 | tee -a "$OUT/run.log"; }
# Kernel-level timing of every FA test case.
has perf && { log "FA kernel perf"; "$TBO" perf -o FLASH_ATTN_EXT -b CUDA0 --output csv > "$OUT/fa-perf.csv" 2> "$OUT/fa-perf.err" || true; }

if has bench; then
  [ -f bench/models.txt ] || { log "no bench/models.txt; skipping llama-bench"; exit 0; }
  while read -r m; do
    [[ -z "$m" || "$m" == \#* ]] && continue
    n=$(basename "$m" .gguf); log "llama-bench $n"
    # Prefill and decode vs. context depth, FA on/off (FA off = the non-fused baseline).
    "$LB" -m "$m" -dev CUDA0 -ngl 99 -fa on,off -p 512,4096 -n 128 -d 0,4096,16384,32768 \
      -r "$REPS" -o jsonl > "$OUT/lb-$n-depth.jsonl" 2> "$OUT/lb-$n-depth.err" || log "  depth sweep failed (see .err)"
    # Quantised KV cache only works with FA on.
    "$LB" -m "$m" -dev CUDA0 -ngl 99 -fa on -ctk f16,q8_0,q4_0 -ctv f16,q8_0,q4_0 -p 4096 -n 128 -d 0,16384 \
      -r "$REPS" -o jsonl > "$OUT/lb-$n-kvq.jsonl" 2> "$OUT/lb-$n-kvq.err" || log "  kv-quant sweep failed (see .err)"
    # Ubatch size changes the Q-tile count per FA call during prefill.
    "$LB" -m "$m" -dev CUDA0 -ngl 99 -fa on -ub 256,512,1024,2048 -p 4096 -n 0 -d 0,16384 \
      -r "$REPS" -o jsonl > "$OUT/lb-$n-ubatch.jsonl" 2> "$OUT/lb-$n-ubatch.err" || log "  ubatch sweep failed (see .err)"
  done < bench/models.txt
fi

if has multi && [ "$ngpu" -ge 2 ] && [ -f bench/models.txt ]; then
  m=$(grep -v '^#' bench/models.txt | head -1); n=$(basename "$m" .gguf)
  log "dual-GPU split modes on $n"
  "$LB" -m "$m" -dev CUDA0/CUDA1 -ngl 99 -fa on -sm layer,row,tensor -p 512,4096 -n 128 -d 0,16384 \
    -r "$REPS" -o jsonl > "$OUT/lb-$n-multi.jsonl" 2> "$OUT/lb-$n-multi.err" || log "  multi sweep failed (see .err)"
fi
log "done: $OUT"
