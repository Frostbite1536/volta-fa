#!/usr/bin/env bash
# On the V100: Q8_0 weight prefetch into registers, 0 / 8 / 17 of the 17 units (MMQ_VOLTA_Q8_0_PF), one build per
# value (bin directory copied; LD_LIBRARY_PATH picks its libraries), rounds interleaved, each vs upstream in the same binary.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
R=~/volta-fa; W=$R/third_party/patches-ab/rev
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
for p in ~/patches-rev4/*.patch ~/p9-dev10.diff; do git -C $W apply $p || { echo APPLY_FAILED $p; exit 1; }; done
t=$W/tests/test-backend-ops.cpp; anchor='test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776'
L='    for (auto mk : std::vector<std::pair<int,int>>{{4096, 4096}, {14336, 4096}, {4096, 14336}, {1024, 4096}, {17408, 5120}, {5120, 17408}, {48, 5120}}) { for (int n : {9, 12, 16}) { test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, mk.first, n, mk.second, {1, 1}, {1, 1})); } } // p9 pf perf' \
  perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "$ENV{L}\n" }' $t
OUT=$R/bench/results/p9-pf-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cp ~/p9-dev10.diff $0 $OUT/
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
cmake -S $W -B $W/build > /dev/null 2>&1
F=$W/ggml/src/ggml-cuda/mmq-volta.cu
for pf in 0 8 17; do
  sed -i -E "s/^#define MMQ_VOLTA_Q8_0_PF [0-9]+/#define MMQ_VOLTA_Q8_0_PF $pf/" $F; grep -q "^#define MMQ_VOLTA_Q8_0_PF $pf\$" $F || { echo SED_FAILED | tee -a $OUT/run.log; exit 1; }
  log "build pf=$pf"
  cmake --build $W/build --target test-backend-ops > $OUT/build-pf$pf.log 2>&1 || { echo BUILD_FAILED | tee -a $OUT/run.log; exit 1; }
  rm -rf ~/bin-pf$pf; cp -r $W/build/bin ~/bin-pf$pf
  cuobjdump -res-usage $W/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq-volta.cu.o 2>&1 | grep -A1 -E "Function .*mul_mat_q_volta.*type8" | paste - - | sed -E 's/\s+/ /g' > $OUT/resources-pf$pf.txt
done
LD_LIBRARY_PATH=~/bin-pf8 ldd ~/bin-pf8/test-backend-ops | grep ggml-cuda | tee $OUT/ldd.txt
log correctness
for pf in 0 8 17; do
  LD_LIBRARY_PATH=~/bin-pf$pf ~/bin-pf$pf/test-backend-ops test -b CUDA0 -o MUL_MAT > $OUT/mulmat-pf$pf.txt 2>&1; echo "pf=$pf MUL_MAT rc=$? $(grep 'tests passed' $OUT/mulmat-pf$pf.txt)" | tee -a $OUT/correct.txt
  GGML_CUDA_MMQ_VOLTA_MIN=2 GGML_CUDA_MMQ_VOLTA_MINN=1 LD_LIBRARY_PATH=~/bin-pf$pf ~/bin-pf$pf/test-backend-ops test -b CUDA0 -o MUL_MAT -p 'type_a=q8_0' > $OUT/mulmat-min2-pf$pf.txt 2>&1; echo "pf=$pf q8_0 min2 minn1 rc=$? $(grep 'tests passed' $OUT/mulmat-min2-pf$pf.txt)" | tee -a $OUT/correct.txt
done
FQ='type_a=q8_0,type_b=f32,m=(4096|14336|1024|17408|5120|48),n=(9|12|16),k=(4096|14336|5120|17408),bs=\[1,1\],nr=\[1,1\]'
for i in 1 2 3; do
  for pf in 0 8 17; do
    log "perf pf=$pf r$i"
    LD_LIBRARY_PATH=~/bin-pf$pf ~/bin-pf$pf/test-backend-ops perf -b CUDA0 -o MUL_MAT -p "$FQ" --output sql 2>/dev/null > $OUT/pf$pf.r$i.sql
    GGML_CUDA_NO_MMQ_VOLTA=1 LD_LIBRARY_PATH=~/bin-pf$pf ~/bin-pf$pf/test-backend-ops perf -b CUDA0 -o MUL_MAT -p "$FQ" --output sql 2>/dev/null > $OUT/up$pf.r$i.sql
  done
done
log "done RESULTS $OUT"
