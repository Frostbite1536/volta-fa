#!/usr/bin/env bash
# On the V100: split-K reduction with independent loads (p9-dev7.diff). Correctness; kernel times for few-row shapes
# (nsys); per-shape timing vs upstream with the split cost at 0/1/2; small N forced.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
R=~/volta-fa; W=$R/third_party/patches-ab/rev
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
for p in ~/patches-rev4/*.patch ~/p9-dev7.diff; do git -C $W apply $p || { echo APPLY_FAILED $p; exit 1; }; done
t=$W/tests/test-backend-ops.cpp; anchor='test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776'
L='    for (ggml_type ta : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) { for (auto mk : std::vector<std::pair<int,int>>{{17408, 5120}, {5120, 17408}, {5120, 6144}, {6144, 5120}, {10240, 5120}, {12288, 5120}, {1024, 5120}, {256, 5120}, {48, 5120}, {14336, 4096}, {4096, 14336}, {4096, 4096}}) { for (int n : {4, 5, 6, 7, 8, 9, 12, 16}) { test_cases.emplace_back(new test_mul_mat(ta, GGML_TYPE_F32, mk.first, n, mk.second, {1, 1}, {1, 1})); } } } // p9 red perf' \
  perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "$ENV{L}\n" }' $t
OUT=$R/bench/results/p9-red-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cp ~/p9-dev7.diff $OUT/
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
cmake -S $W -B $W/build > /dev/null 2>&1
log build
cmake --build $W/build --target test-backend-ops llama-bench > $OUT/build.log 2>&1 || { echo BUILD_FAILED | tee -a $OUT/run.log; exit 1; }
cuobjdump -res-usage $W/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq-volta.cu.o 2>&1 | grep -A1 -E "Function .*(mul_mat_q_volta|mmq_volta_prep)" | paste - - | sed -E 's/\s+/ /g' > $OUT/resources.txt
log correctness
B=$W/build/bin/test-backend-ops
$B test -b CUDA0 -o MUL_MAT > $OUT/mulmat.txt 2>&1; echo "MUL_MAT rc=$? $(grep 'tests passed' $OUT/mulmat.txt)" | tee $OUT/correct.txt
GGML_CUDA_MMQ_VOLTA_MIN=2 GGML_CUDA_MMQ_VOLTA_MINN=1 $B test -b CUDA0 -o MUL_MAT -p 'type_a=(q[456]_K|iq4_xs|q8_0)' > $OUT/mulmat-min2.txt 2>&1; echo "MUL_MAT min2 minn1 rc=$? $(grep 'tests passed' $OUT/mulmat-min2.txt)" | tee -a $OUT/correct.txt
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | tail -1)
for ta in q8_0 q4_K; do for m in 48 4096; do for n in 8 16; do for pd in 2 1; do
  tag=$ta.m$m.n$n.pd$pd; F="type_a=$ta,type_b=f32,m=$m,n=$n,k=5120,bs=\[1,1\],nr=\[1,1\]"
  GGML_CUDA_MMQ_VOLTA_MIN=2 GGML_CUDA_MMQ_VOLTA_MINN=1 GGML_CUDA_MMQ_VOLTA_PDIV=$pd $NSYS profile -t cuda -o $OUT/$tag --force-overwrite true $B perf -b CUDA0 -o MUL_MAT -p "$F" > /dev/null 2>&1
  $NSYS stats --report cuda_gpu_kern_sum --format csv -o $OUT/$tag $OUT/$tag.nsys-rep > /dev/null 2>&1; rm -f $OUT/$tag.nsys-rep $OUT/$tag.sqlite
  echo "$tag $(grep -h mul_mat_q_volta $OUT/${tag}_cuda_gpu_kern_sum.csv | awk -F, '{printf "kern avg %.1f us", $4/1000}')" | tee -a $OUT/summary.txt
done; done; done; done
FK='type_a=(q4_K|q5_K|iq4_xs|q6_K),type_b=f32,m=(17408|5120|6144|10240|12288),n=(5|6|7|8|9|12|16),k=(5120|17408|6144),bs=\[1,1\],nr=\[1,1\]'
FS='type_a=(q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(1024|256|48),n=(9|12|16),k=5120,bs=\[1,1\],nr=\[1,1\]'
F8='type_a=q8_0,type_b=f32,m=(14336|4096),n=(5|6|7|8|9|12|16),k=(4096|14336),bs=\[1,1\],nr=\[1,1\]'
M="GGML_CUDA_MMQ_VOLTA_MIN=5 GGML_CUDA_MMQ_VOLTA_MINN=1"
declare -A V=( [up]="GGML_CUDA_NO_MMQ_VOLTA=1" [w2]="$M" [w1]="$M GGML_CUDA_MMQ_VOLTA_WAVE=1" [w0]="$M GGML_CUDA_MMQ_VOLTA_WAVE=0" )
for i in 1 2; do
  for v in up w2 w1 w0; do
    log "perf $v r$i"
    for f in "$FK" "$FS" "$F8"; do env ${V[$v]} $B perf -b CUDA0 -o MUL_MAT -p "$f" --output sql 2>/dev/null; done > $OUT/$v.r$i.sql
  done
done
log "done RESULTS $OUT"
