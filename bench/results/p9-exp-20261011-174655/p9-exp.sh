#!/usr/bin/env bash
# On the V100: where the kernel's time goes. Same binary (p9-dev8.diff): NOLOAD (weights from L2: all row tiles read
# rows 0..31), NC (loads and staging only, same blocks per SM; both NC and NOLOAD give wrong results) and MB (3 blocks of
# 4 warps / 6 of 2 warps per SM via __launch_bounds__), vs upstream and the default.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
R=~/volta-fa; W=$R/third_party/patches-ab/rev
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
for p in ~/patches-rev4/*.patch ~/p9-dev8.diff; do git -C $W apply $p || { echo APPLY_FAILED $p; exit 1; }; done
t=$W/tests/test-backend-ops.cpp; anchor='test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776'
L='    for (ggml_type ta : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) { for (auto mk : std::vector<std::pair<int,int>>{{17408, 5120}, {5120, 17408}, {5120, 6144}, {6144, 5120}, {10240, 5120}, {12288, 5120}, {14336, 4096}, {4096, 14336}}) { for (int n : {8, 16}) { test_cases.emplace_back(new test_mul_mat(ta, GGML_TYPE_F32, mk.first, n, mk.second, {1, 1}, {1, 1})); } } } // p9 exp perf' \
  perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "$ENV{L}\n" }' $t
OUT=$R/bench/results/p9-exp-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cp ~/p9-dev8.diff $OUT/
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
cmake -S $W -B $W/build > /dev/null 2>&1
log build
cmake --build $W/build --target test-backend-ops > $OUT/build.log 2>&1 || { echo BUILD_FAILED | tee -a $OUT/run.log; exit 1; }
cuobjdump -res-usage $W/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq-volta.cu.o 2>&1 | grep -A1 -E "Function .*(mul_mat_q_volta|mmq_volta_prep)" | paste - - | sed -E 's/\s+/ /g' > $OUT/resources.txt
B=$W/build/bin/test-backend-ops
log correctness
GGML_CUDA_MMQ_VOLTA_MB=3 GGML_CUDA_MMQ_VOLTA_MIN=2 GGML_CUDA_MMQ_VOLTA_MINN=1 $B test -b CUDA0 -o MUL_MAT -p 'type_a=(q[456]_K|iq4_xs|q8_0)' > $OUT/mulmat-mb.txt 2>&1; echo "MUL_MAT mb min2 minn1 rc=$? $(grep 'tests passed' $OUT/mulmat-mb.txt)" | tee $OUT/correct.txt
F='type_a=(q4_K|q5_K|iq4_xs|q6_K),type_b=f32,m=(17408|5120|6144|10240|12288),n=(8|16),k=(5120|17408|6144),bs=\[1,1\],nr=\[1,1\]'
F8='type_a=q8_0,type_b=f32,m=(14336|4096),n=(8|16),k=(4096|14336),bs=\[1,1\],nr=\[1,1\]'
M="GGML_CUDA_MMQ_VOLTA_MIN=5"
declare -A V=( [up]="GGML_CUDA_NO_MMQ_VOLTA=1" [base]="$M" [mb]="$M GGML_CUDA_MMQ_VOLTA_MB=3" [nl]="$M GGML_CUDA_MMQ_VOLTA_NOLOAD=1" [nlmb]="$M GGML_CUDA_MMQ_VOLTA_NOLOAD=1 GGML_CUDA_MMQ_VOLTA_MB=3" [nc]="$M GGML_CUDA_MMQ_VOLTA_NOCOMPUTE=1" )
for i in 1 2; do
  for v in up base mb nl nlmb nc; do
    log "perf $v r$i"
    for f in "$F" "$F8"; do env ${V[$v]} $B perf -b CUDA0 -o MUL_MAT -p "$f" --output sql 2>/dev/null; done > $OUT/$v.r$i.sql
  done
done
log "done RESULTS $OUT"
