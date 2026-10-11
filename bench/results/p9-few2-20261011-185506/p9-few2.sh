#!/usr/bin/env bash
# On the V100: few-row matrices in the real model (weights not in L2). Qwen pp16 under nsys with the K split capped at
# nsb/2 (default) or nsb (1 superblock per block); llama-bench both, interleaved. p9-dev11.diff = final4 + dev switches.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
R=~/volta-fa; W=$R/third_party/patches-ab/rev
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
for p in ~/patches-rev4/*.patch ~/p9-dev11.diff; do git -C $W apply $p || { echo APPLY_FAILED $p; exit 1; }; done
OUT=$R/bench/results/p9-few2-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cp ~/p9-dev11.diff $0 $OUT/
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
cmake -S $W -B $W/build > /dev/null 2>&1
log build
cmake --build $W/build --target test-backend-ops llama-bench > $OUT/build.log 2>&1 || { echo BUILD_FAILED | tee -a $OUT/run.log; exit 1; }
B=$W/build/bin/llama-bench; Q=/root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
log correctness
GGML_CUDA_MMQ_VOLTA_PDIV=1 $W/build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT > $OUT/mulmat-pd1.txt 2>&1; echo "MUL_MAT pdiv1 rc=$? $(grep 'tests passed' $OUT/mulmat-pd1.txt)" | tee $OUT/correct.txt
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | tail -1)
for pd in 2 1; do
  log "nsys pd$pd"
  GGML_CUDA_MMQ_VOLTA_PDIV=$pd $NSYS profile -t cuda --cuda-graph-trace=node -o $OUT/nsys-pd$pd --force-overwrite true $B -m $Q -dev CUDA0 -ngl 99 -fa on -r 5 -p 8,16 -n 0 -d 0 -o jsonl > /dev/null 2>&1
  $NSYS stats --report cuda_gpu_kern_sum --format csv -o $OUT/kern-pd$pd $OUT/nsys-pd$pd.nsys-rep > /dev/null 2>&1
  python3 /root/shapes.py $OUT/nsys-pd$pd.sqlite > $OUT/shapes-pd$pd.txt 2>&1
  rm -f $OUT/nsys-pd$pd.nsys-rep $OUT/nsys-pd$pd.sqlite
done
for i in 1 2; do
  for M in /root/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf $Q; do
    m=$(basename $M .gguf)
    for pd in 2 1; do
      log "lb $m pd$pd r$i"
      GGML_CUDA_MMQ_VOLTA_PDIV=$pd $B -m $M -dev CUDA0 -ngl 99 -fa on -r 5 -o jsonl -p 8,12,16 -n 0 -d 0 > $OUT/lb-pd$pd.$m.r$i.jsonl 2> $OUT/lb-pd$pd.$m.r$i.err
    done
  done
done
log "done RESULTS $OUT"
