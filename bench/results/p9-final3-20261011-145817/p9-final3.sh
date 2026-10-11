#!/usr/bin/env bash
# On the V100: the exact candidate 0008 (Q6_K, tail rows, wave-based K split; no dev switches). Resources, full MUL_MAT,
# llama-bench both models (upstream path in the same binary vs P-9), nsys of Qwen pp16.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
R=~/volta-fa; W=$R/third_party/patches-ab/rev
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
for p in ~/patches-rev4/*.patch ~/p9-final3.diff; do git -C $W apply $p || { echo APPLY_FAILED $p; exit 1; }; done
OUT=$R/bench/results/p9-final3-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cp ~/p9-final3.diff $OUT/
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
cmake -S $W -B $W/build > /dev/null 2>&1
log build
cmake --build $W/build --target test-backend-ops llama-bench > $OUT/build.log 2>&1 || { echo BUILD_FAILED | tee -a $OUT/run.log; exit 1; }
cuobjdump -res-usage $W/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq-volta.cu.o 2>&1 | grep -A1 -E "Function .*(mul_mat_q_volta|mmq_volta_prep)" | paste - - | sed -E 's/\s+/ /g' > $OUT/resources.txt
log correctness
$W/build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT > $OUT/mulmat.txt 2>&1; echo "MUL_MAT rc=$? $(grep 'tests passed' $OUT/mulmat.txt)" | tee $OUT/correct.txt
B=$W/build/bin/llama-bench
for i in 1 2; do
  for M in /root/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf /root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf; do
    m=$(basename $M .gguf)
    log "lb $m r$i"
    GGML_CUDA_NO_MMQ_VOLTA=1 $B -m $M -dev CUDA0 -ngl 99 -fa on -r 5 -o jsonl -p 1,2,4,8,12,16 -n 0 -d 0 > $OUT/lb-up.$m.r$i.jsonl 2> $OUT/lb-up.$m.r$i.err
    $B -m $M -dev CUDA0 -ngl 99 -fa on -r 5 -o jsonl -p 1,2,4,8,12,16 -n 0 -d 0 > $OUT/lb-p9.$m.r$i.jsonl 2> $OUT/lb-p9.$m.r$i.err
  done
done
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | tail -1)
log nsys
$NSYS profile -t cuda --cuda-graph-trace=node -o $OUT/nsys-p9-pp16 --force-overwrite true $B -m /root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf -dev CUDA0 -ngl 99 -fa on -r 5 -p 16 -n 0 -d 0 -o jsonl > /dev/null 2>&1
$NSYS stats --report cuda_gpu_kern_sum --format csv -o $OUT/kern-p9-pp16 $OUT/nsys-p9-pp16.nsys-rep > /dev/null 2>&1
python3 /root/shapes.py $OUT/nsys-p9-pp16.sqlite > $OUT/shapes-p9-pp16.txt 2>&1
rm -f $OUT/nsys-p9-pp16.nsys-rep $OUT/nsys-p9-pp16.sqlite
log "done RESULTS $OUT"
