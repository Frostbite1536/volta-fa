#!/usr/bin/env bash
# On the V100: rev tree = 0001-0007 + 0008 with Q6_K, tail rows and dev switches (p9-dev4.diff). Resources, correctness,
# then per-shape timing vs the upstream path: K split (old target 4 blocks/SM vs least waves*(steps+cost)), 2 vs 4 warps,
# Q6_K thresholds, and the N = 48 Q8_0 shape.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
R=~/volta-fa; W=$R/third_party/patches-ab/rev; TAG=${1:-p9-q6b}
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
for p in ~/patches-rev4/*.patch ~/p9-dev4.diff; do git -C $W apply $p || { echo APPLY_FAILED $p; exit 1; }; done
t=$W/tests/test-backend-ops.cpp; anchor='test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776'
L='    for (ggml_type ta : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K}) { for (auto mk : std::vector<std::pair<int,int>>{{17408, 5120}, {5120, 17408}, {5120, 6144}, {6144, 5120}, {10240, 5120}, {12288, 5120}}) { for (int n : {4, 5, 6, 7, 8, 9, 12, 16}) { test_cases.emplace_back(new test_mul_mat(ta, GGML_TYPE_F32, mk.first, n, mk.second, {1, 1}, {1, 1})); } } } for (int n : {9, 12, 16}) { test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 48, n, 5120, {1, 1}, {1, 1})); } // p9 q6 perf' \
  perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "$ENV{L}\n" }' $t
OUT=$R/bench/results/$TAG-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cp ~/p9-dev4.diff $OUT/
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
cmake -S $W -B $W/build > /dev/null 2>&1
log build
cmake --build $W/build --target test-backend-ops llama-bench llama-perplexity > $OUT/build.log 2>&1 || { echo BUILD_FAILED; tail -30 $OUT/build.log; exit 1; }
cuobjdump -res-usage $W/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq-volta.cu.o 2>&1 | grep -A1 -E "Function .*(mul_mat_q_volta|mmq_volta_prep)" | paste - - | sed -E 's/\s+/ /g' > $OUT/resources.txt
log correctness
B=$W/build/bin/test-backend-ops
$B test -b CUDA0 -o MUL_MAT > $OUT/mulmat.txt 2>&1; echo "MUL_MAT rc=$? $(grep 'tests passed' $OUT/mulmat.txt)" | tee $OUT/correct.txt
GGML_CUDA_MMQ_VOLTA_MIN=2 GGML_CUDA_MMQ_VOLTA_MINN=1 $B test -b CUDA0 -o MUL_MAT -p 'type_a=(q[456]_K|iq4_xs|q8_0)' > $OUT/mulmat-min2.txt 2>&1; echo "MUL_MAT min2 minn1 rc=$? $(grep 'tests passed' $OUT/mulmat-min2.txt)" | tee -a $OUT/correct.txt
GGML_CUDA_MMQ_VOLTA_MIN=2 GGML_CUDA_MMQ_VOLTA_MINN=1 GGML_CUDA_MMQ_VOLTA_WAVE=2 $B test -b CUDA0 -o MUL_MAT -p 'type_a=(q[456]_K|iq4_xs|q8_0)' > $OUT/mulmat-wave.txt 2>&1; echo "MUL_MAT min2 minn1 wave2 rc=$? $(grep 'tests passed' $OUT/mulmat-wave.txt)" | tee -a $OUT/correct.txt
GGML_CUDA_MMQ_VOLTA_VERBOSE=1 GGML_CUDA_MMQ_VOLTA_WAVE=2 GGML_CUDA_MMQ_VOLTA_MIN=4 $B perf -b CUDA0 -o MUL_MAT -p 'type_a=(q5_K|q6_K),type_b=f32,m=6144,n=(8|16),k=5120' 2>&1 | grep blocks/SM > $OUT/occupancy.txt
FK='type_a=(q4_K|q5_K|iq4_xs),type_b=f32,m=(17408|5120|6144|10240|12288),n=(8|12|16),k=(5120|17408|6144),bs=\[1,1\],nr=\[1,1\]'
F6='type_a=q6_K,type_b=f32,m=(5120|6144|10240),n=(4|5|6|7|8|9|12|16),k=(5120|6144),bs=\[1,1\],nr=\[1,1\]'
F8='type_a=q8_0,type_b=f32,m=48,n=(9|12|16),k=5120,bs=\[1,1\],nr=\[1,1\]'
M="GGML_CUDA_MMQ_VOLTA_MIN=4 GGML_CUDA_MMQ_VOLTA_MINN=1"
declare -A V=( [up]="GGML_CUDA_NO_MMQ_VOLTA=1" [p9]="$M" [w0]="$M GGML_CUDA_MMQ_VOLTA_WAVE=0" [w2]="$M GGML_CUDA_MMQ_VOLTA_WAVE=2"
               [w4]="$M GGML_CUDA_MMQ_VOLTA_WAVE=4" [nr2w2]="$M GGML_CUDA_MMQ_VOLTA_NR=2 GGML_CUDA_MMQ_VOLTA_WAVE=2" )
for i in 1 2; do
  for v in up p9 w0 w2 w4 nr2w2; do
    log "$v r$i"
    for f in "$FK" "$F6" "$F8"; do
      env ${V[$v]} $B perf -b CUDA0 -o MUL_MAT -p "$f" --output sql 2>/dev/null
    done > $OUT/$v.r$i.sql
  done
done
log "done RESULTS $OUT"
