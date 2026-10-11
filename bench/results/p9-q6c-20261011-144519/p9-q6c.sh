#!/usr/bin/env bash
# On the V100 after p9-acc: rev tree = 0001-0007 + 0008 with Q6_K, tail rows, wave-based K split as default, dev switches
# (p9-dev5.diff). Correctness; per-shape timing (Q6_K ffn shapes, small N, Llama Q8_0) vs upstream; nsys of the N = 48
# Q8_0 case; llama-bench on both models, upstream path vs P-9.
# Usage: p9-q6c.sh TAG WAITLOG
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
TAG=$1; WAIT=$2
until grep -q "RESULTS\|FAILED" $WAIT 2>/dev/null; do sleep 15; done
R=~/volta-fa; W=$R/third_party/patches-ab/rev
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
for p in ~/patches-rev4/*.patch ~/p9-dev5.diff; do git -C $W apply $p || { echo APPLY_FAILED $p; exit 1; }; done
t=$W/tests/test-backend-ops.cpp; anchor='test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776'
L='    for (ggml_type ta : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) { for (auto mk : std::vector<std::pair<int,int>>{{17408, 5120}, {5120, 17408}, {1024, 5120}, {256, 5120}, {48, 5120}, {14336, 4096}, {4096, 14336}, {4096, 4096}}) { for (int n : {4, 5, 6, 7, 8, 9, 12, 16}) { test_cases.emplace_back(new test_mul_mat(ta, GGML_TYPE_F32, mk.first, n, mk.second, {1, 1}, {1, 1})); } } } // p9 q6c perf' \
  perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "$ENV{L}\n" }' $t
OUT=$R/bench/results/$TAG-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cp ~/p9-dev5.diff $OUT/
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
cmake -S $W -B $W/build > /dev/null 2>&1
log build
cmake --build $W/build --target test-backend-ops llama-bench llama-perplexity > $OUT/build.log 2>&1 || { echo BUILD_FAILED | tee -a $OUT/run.log; tail -30 $OUT/build.log; exit 1; }
cuobjdump -res-usage $W/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq-volta.cu.o 2>&1 | grep -A1 -E "Function .*(mul_mat_q_volta|mmq_volta_prep)" | paste - - | sed -E 's/\s+/ /g' > $OUT/resources.txt
log correctness
B=$W/build/bin/test-backend-ops
$B test -b CUDA0 -o MUL_MAT > $OUT/mulmat.txt 2>&1; echo "MUL_MAT rc=$? $(grep 'tests passed' $OUT/mulmat.txt)" | tee $OUT/correct.txt
GGML_CUDA_MMQ_VOLTA_MIN=2 GGML_CUDA_MMQ_VOLTA_MINN=1 $B test -b CUDA0 -o MUL_MAT -p 'type_a=(q[456]_K|iq4_xs|q8_0)' > $OUT/mulmat-min2.txt 2>&1; echo "MUL_MAT min2 minn1 rc=$? $(grep 'tests passed' $OUT/mulmat-min2.txt)" | tee -a $OUT/correct.txt
F1='type_a=q6_K,type_b=f32,m=(17408|5120),n=(4|5|6|7|8|9|12|16),k=(5120|17408),bs=\[1,1\],nr=\[1,1\]'
F2='type_a=(q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(1024|256|48),n=(9|12|16),k=5120,bs=\[1,1\],nr=\[1,1\]'
F3='type_a=q8_0,type_b=f32,m=(14336|4096),n=(5|6|7|8|9|12|16),k=(4096|14336),bs=\[1,1\],nr=\[1,1\]'
declare -A V=( [up]="GGML_CUDA_NO_MMQ_VOLTA=1" [p9]="GGML_CUDA_MMQ_VOLTA_MIN=4 GGML_CUDA_MMQ_VOLTA_MINN=1" )
for i in 1 2; do
  for v in up p9; do
    log "perf $v r$i"
    for f in "$F1" "$F2" "$F3"; do env ${V[$v]} $B perf -b CUDA0 -o MUL_MAT -p "$f" --output sql 2>/dev/null; done > $OUT/$v.r$i.sql
  done
done
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | tail -1)
log nsys-n48
GGML_CUDA_MMQ_VOLTA_MINN=1 $NSYS profile -t cuda -o $OUT/nsys-n48 --force-overwrite true $B perf -b CUDA0 -o MUL_MAT -p 'type_a=q8_0,type_b=f32,m=48,n=(9|16),k=5120,bs=\[1,1\],nr=\[1,1\]' > $OUT/nsys-n48.txt 2>&1
$NSYS stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv -o $OUT/n48 $OUT/nsys-n48.nsys-rep > /dev/null 2>&1
rm -f $OUT/nsys-n48.nsys-rep
LB=$W/build/bin/llama-bench
for i in 1 2; do
  for M in /root/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf /root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf; do
    m=$(basename $M .gguf)
    log "lb $m r$i"
    GGML_CUDA_NO_MMQ_VOLTA=1 $LB -m $M -dev CUDA0 -ngl 99 -fa on -r 5 -o jsonl -p 1,4,8,12,16 -n 0 -d 0 > $OUT/lb-up.$m.r$i.jsonl 2> $OUT/lb-up.$m.r$i.err
    $LB -m $M -dev CUDA0 -ngl 99 -fa on -r 5 -o jsonl -p 1,4,8,12,16 -n 0 -d 0 > $OUT/lb-p9.$m.r$i.jsonl 2> $OUT/lb-p9.$m.r$i.err
  done
done
log "done RESULTS $OUT"
