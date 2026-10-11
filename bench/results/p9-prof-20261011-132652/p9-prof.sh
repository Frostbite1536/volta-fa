#!/usr/bin/env bash
# On the V100: where the time of a Qwen 8/16-token evaluation goes with patch 0008 (rev build = 0001-0008), P-9 vs the
# upstream path in the same binary; nsys kernel sums + GPU trace.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
R=~/volta-fa; W=$R/third_party/patches-ab/rev
OUT=$R/bench/results/p9-prof-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader)" | tee -a $OUT/run.log; }
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | tail -1)
Q=/root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf; B=$W/build/bin/llama-bench
git -C $W diff --stat | tail -1 > $OUT/tree.txt
for v in p9 up; do
  for p in 16 8; do
    log "nsys $v pp$p"
    if [ $v = up ]; then export GGML_CUDA_NO_MMQ_VOLTA=1; else unset GGML_CUDA_NO_MMQ_VOLTA; fi
    $NSYS profile -t cuda --cuda-graph-trace=node -o $OUT/nsys-$v-pp$p --force-overwrite true \
      $B -m $Q -dev CUDA0 -ngl 99 -fa on -r 5 -p $p -n 0 -d 0 -o jsonl > $OUT/nsys-$v-pp$p.jsonl 2> $OUT/nsys-$v-pp$p.err
    $NSYS stats --report cuda_gpu_kern_sum --format csv -o $OUT/kern-$v-pp$p $OUT/nsys-$v-pp$p.nsys-rep > /dev/null 2>&1
    $NSYS stats --report cuda_gpu_trace --format csv -o $OUT/trace-$v-pp$p $OUT/nsys-$v-pp$p.nsys-rep > /dev/null 2>&1
    $NSYS stats --report cuda_gpu_mem_time_sum --format csv -o $OUT/mem-$v-pp$p $OUT/nsys-$v-pp$p.nsys-rep > /dev/null 2>&1
  done
done
unset GGML_CUDA_NO_MMQ_VOLTA
log "done RESULTS $OUT"
