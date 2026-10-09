#!/bin/bash
# phase0.sh: Phase 0 baseline on the V100 box (docs/ROADMAP.md). Run from the repo root.
#   bench/phase0.sh                 full run: env, build, FA support/correctness/perf, llama-bench grid
#   STEPS="env perf" bench/phase0.sh   run only some steps (env build support test perf bench multi)
#   LLAMA=third_party/llama-superfast-v100 EXTRA_CMAKE="-DGGML_CUDA_CUB_3DOT2=ON" bench/phase0.sh
#                                      same kit against another llama.cpp tree (results tagged by tree name)
# Models: one GGUF path per line in bench/models.txt (see bench/models.example.txt).
# Output: bench/results/phase0-<date>-<host>/ (commit the small text/csv/jsonl files, never models).
set -euo pipefail
cd "$(dirname "$0")/.." || exit 2
LLAMA=${LLAMA:-third_party/llama.cpp}
BUILD=$LLAMA/build-sm70
STEPS=${STEPS:-"env build support test perf bench multi"}
REPS=${REPS:-3}
TAG=${TAG:-$(basename "$LLAMA")}
EXTRA_CMAKE=${EXTRA_CMAKE:-}
read -ra EXTRA <<< "$EXTRA_CMAKE"
OUT=bench/results/phase0-$TAG-$(date +%Y%m%d-%H%M%S)-$(hostname -s 2>/dev/null || hostname)
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }

# Refuse to produce "baseline" numbers on the wrong hardware or toolchain (INV-PERF-001, INV-ARCH-003).
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[[ "$gpu" == *V100* || "${FORCE:-0}" == 1 ]] || { echo "GPU is '$gpu', not a V100 (FORCE=1 to override)" >&2; exit 2; }
ver=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
[ "$ver" = 12.8 ] || [ "${FORCE:-0}" = 1 ] || { echo "nvcc is '$ver'; CUDA 12.8 required (13 dropped sm_70; 12.9: llama.cpp #28416). FORCE=1 to override" >&2; exit 2; }
ngpu=$(nvidia-smi -L | grep -c V100 || true)

if has env; then
  log "env"
  # Subshell with set +e: a missing tool in a container (topo, ncu) must not abort the whole run.
  ( set +e; echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "volta-fa: $(git rev-parse --short HEAD)"
    echo "llama.cpp tree: $LLAMA ($TAG) $(git -C "$LLAMA" rev-parse HEAD)"; echo "extra cmake: $EXTRA_CMAKE"; nvcc --version | tail -2; uname -r
    nvidia-smi --query-gpu=index,name,driver_version,pcie.link.gen.max,pcie.link.width.max,clocks.max.sm,clocks.max.mem,power.limit,memory.total --format=csv
    echo; nvidia-smi topo -m; echo; nvidia-smi topo -p2p r 2>/dev/null || true
    command -v ncu >/dev/null && ncu --version | tail -1; command -v nsys >/dev/null && nsys --version
  ) > "$OUT/env.txt" 2>&1 || true   # the subshell's status is its last command's (e.g. no nsys)
fi

if has build; then
  log "build llama.cpp for sm_70"
  if ! { cmake -S "$LLAMA" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 -DLLAMA_CURL=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${EXTRA[@]}" > "$OUT/build.log" 2>&1 &&
         cmake --build "$BUILD" --target llama-bench test-backend-ops llama-cli >> "$OUT/build.log" 2>&1; }; then
    log "BUILD FAILED (last lines below; full log $OUT/build.log)"; tail -20 "$OUT/build.log" | tee -a "$OUT/run.log"; exit 1
  fi
  # GGML_CUDA_NCCL defaults ON but silently builds without NCCL if it is not installed.
  if grep -q "GGML_USE_NCCL" "$BUILD/compile_commands.json" 2>/dev/null; then log "  NCCL: linked"
  else log "  NCCL: NOT linked (install libnccl2 + libnccl-dev for CUDA 12; -sm tensor nccl runs fall back)"; fi
fi
TBO=$BUILD/bin/test-backend-ops; LB=$BUILD/bin/llama-bench

# Which FLASH_ATTN_EXT cases the CUDA backend even supports on cc 7.0 (model-compatibility gaps).
has support && { log "FA support matrix"; "$TBO" support -o FLASH_ATTN_EXT -b CUDA0 > "$OUT/fa-support.txt" 2>&1 || true; }
# Correctness vs CPU (a Volta-path bug is itself a finding).
has test && { log "FA correctness"; "$TBO" test -o FLASH_ATTN_EXT -b CUDA0 > "$OUT/fa-test.txt" 2>&1 || true; { grep -E "FAIL|passed" "$OUT/fa-test.txt" || echo "  (no pass/fail summary found)"; } | tail -3 | tee -a "$OUT/run.log"; }
# Kernel-level timing of every FA test case.
has perf && { log "FA kernel perf"; stdbuf -oL "$TBO" perf -o FLASH_ATTN_EXT -b CUDA0 --output sql > "$OUT/fa-perf.sql" 2> "$OUT/fa-perf.err" || true; }

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
    # Tiny batches at depth: Q*GQA <= 16 routes to the non-tensor-core TILE kernel on Volta
    # (fattn.cu:673-681). This is the speculative-decoding verify shape (research gap 1).
    "$LB" -m "$m" -dev CUDA0 -ngl 99 -fa on -p 1,2,4,8,16 -n 0 -d 4096,32768 \
      -r "$REPS" -o jsonl > "$OUT/lb-$n-smallbatch.jsonl" 2> "$OUT/lb-$n-smallbatch.err" || log "  small-batch sweep failed (see .err)"
    # Ubatch size changes the Q-tile count per FA call during prefill.
    "$LB" -m "$m" -dev CUDA0 -ngl 99 -fa on -ub 256,512,1024,2048 -p 4096 -n 0 -d 0,16384 \
      -r "$REPS" -o jsonl > "$OUT/lb-$n-ubatch.jsonl" 2> "$OUT/lb-$n-ubatch.err" || log "  ubatch sweep failed (see .err)"
  done < bench/models.txt
fi

if has multi && [ "$ngpu" -ge 2 ] && [ -f bench/models.txt ]; then
  mm=$(grep -v -e '^#' -e '^[[:space:]]*$' bench/models.txt | head -1 || true)
fi
if has multi && [ -n "${mm:-}" ]; then
  n=$(basename "$mm" .gguf)
  log "dual-GPU split modes on $n"
  # -sm row no longer exists for CUDA (llama.cpp #24216). Tensor split all-reduce is NCCL or an
  # internal host-staged fallback; GGML_CUDA_P2P only enables peer access for plain copies.
  env -u GGML_CUDA_P2P -u GGML_CUDA_ALLREDUCE "$LB" -m "$mm" -dev CUDA0/CUDA1 -ngl 99 -fa on -sm layer -p 512,4096 -n 128 -d 0,16384 \
    -r "$REPS" -o jsonl > "$OUT/lb-$n-multi-layer.jsonl" 2> "$OUT/lb-$n-multi-layer.err" || log "  layer split failed"
  # GGML_CUDA_P2P is presence-checked (getenv != nullptr): "=0" would ENABLE it, so unset for off.
  for ar in nccl internal; do for p2p in 0 1; do
    p2penv=(-u GGML_CUDA_P2P); [ "$p2p" = 1 ] && p2penv=(GGML_CUDA_P2P=1)
    env "${p2penv[@]}" GGML_CUDA_ALLREDUCE=$ar "$LB" -m "$mm" -dev CUDA0/CUDA1 -ngl 99 -fa on -sm tensor \
      -p 512,4096 -n 128 -d 0,16384 -r "$REPS" -o jsonl > "$OUT/lb-$n-multi-tensor-$ar-p2p$p2p.jsonl" \
      2> "$OUT/lb-$n-multi-tensor-$ar-p2p$p2p.err" || log "  tensor split ($ar, p2p=$p2p) failed"
  done; done
fi
log "done: $OUT"
