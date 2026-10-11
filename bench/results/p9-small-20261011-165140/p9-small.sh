#!/usr/bin/env bash
# On the V100: matrices with few rows (N = 16..256) vs upstream at 5-16 tokens, with the split-K reduction fix
# (p9-dev7.diff), small N forced on.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
R=~/volta-fa; W=$R/third_party/patches-ab/rev
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
for p in ~/patches-rev4/*.patch ~/p9-dev7.diff; do git -C $W apply $p || { echo APPLY_FAILED $p; exit 1; }; done
t=$W/tests/test-backend-ops.cpp; anchor='test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776'
L='    for (ggml_type ta : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) { for (int m : {16, 32, 48, 64, 128, 256, 512}) { for (int k : {4096, 5120}) { for (int n : {5, 6, 7, 8, 9, 12, 16}) { test_cases.emplace_back(new test_mul_mat(ta, GGML_TYPE_F32, m, n, k, {1, 1}, {1, 1})); } } } } // p9 small perf' \
  perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "$ENV{L}\n" }' $t
OUT=$R/bench/results/p9-small-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cp ~/p9-dev7.diff $OUT/
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
cmake -S $W -B $W/build > /dev/null 2>&1
log build
cmake --build $W/build --target test-backend-ops > $OUT/build.log 2>&1 || { echo BUILD_FAILED | tee -a $OUT/run.log; exit 1; }
B=$W/build/bin/test-backend-ops
log correctness
GGML_CUDA_MMQ_VOLTA_MIN=2 GGML_CUDA_MMQ_VOLTA_MINN=1 $B test -b CUDA0 -o MUL_MAT -p 'type_a=(q[456]_K|iq4_xs|q8_0),type_b=f32,m=(16|32|48|64|128|256|512),n=(5|9|16),k=(4096|5120),bs=\[1,1\]' > $OUT/mulmat-small.txt 2>&1; echo "MUL_MAT small rc=$? $(grep 'tests passed' $OUT/mulmat-small.txt)" | tee $OUT/correct.txt
F='type_a=(q4_K|q5_K|iq4_xs|q6_K|q8_0),type_b=f32,m=(16|32|48|64|128|256|512),n=(5|6|7|8|9|12|16),k=(4096|5120),bs=\[1,1\],nr=\[1,1\]'
declare -A V=( [up]="GGML_CUDA_NO_MMQ_VOLTA=1" [p9]="GGML_CUDA_MMQ_VOLTA_MIN=5 GGML_CUDA_MMQ_VOLTA_MINN=1" )
for i in 1 2 3; do
  for v in up p9; do
    log "perf $v r$i"
    env ${V[$v]} $B perf -b CUDA0 -o MUL_MAT -p "$F" --output sql 2>/dev/null > $OUT/$v.r$i.sql
  done
done
log "done RESULTS $OUT"
