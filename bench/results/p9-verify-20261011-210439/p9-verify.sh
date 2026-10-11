#!/usr/bin/env bash
# On the V100: final check of the committed series. Applies 0001-0007, the regenerated 0008 (per-slice activation
# scaling) and the local-only range tests (0009) to a clean tree, checks that mmq-volta.cu is the benchmarked one,
# builds test-backend-ops, records resources and runs MUL_MAT, MUL_MAT_ID and the range tests (also on the current path).
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
R=~/volta-fa; W=$R/third_party/patches-ab/rev
OUT=$R/bench/results/p9-verify-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cp ~/p9v/*.patch $0 $OUT/
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
log apply
ls ~/patches-rev4/ | tee -a $OUT/run.log
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
for p in ~/patches-rev4/000[1-7]*.patch ~/p9v/0008-*.patch ~/p9v/0009-*.patch; do git -C $W apply $p || { echo APPLY_FAILED $p | tee -a $OUT/run.log; exit 1; }; done
echo "mmq-volta.cu md5 $(md5sum < $W/ggml/src/ggml-cuda/mmq-volta.cu) expected $(cat ~/p9v/slice.md5)" | tee -a $OUT/run.log
log build
cmake -S $W -B $W/build > /dev/null 2>&1; cmake --build $W/build --target test-backend-ops > $OUT/build.log 2>&1 || { echo BUILD_FAILED | tee -a $OUT/run.log; exit 1; }
cuobjdump -res-usage $W/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq-volta.cu.o 2>&1 | grep -A1 -E "Function .*(mul_mat_q_volta|mmq_volta_prep)" | paste - - | sed -E 's/\s+/ /g' > $OUT/resources.txt
B=$W/build/bin/test-backend-ops
log tests
$B test -b CUDA0 -o MUL_MAT > $OUT/mulmat.txt 2>&1; echo "MUL_MAT rc=$? $(grep 'tests passed' $OUT/mulmat.txt)" | tee -a $OUT/correct.txt
$B test -b CUDA0 -o MUL_MAT_ID > $OUT/mulmatid.txt 2>&1; echo "MUL_MAT_ID rc=$? $(grep 'tests passed' $OUT/mulmatid.txt)" | tee -a $OUT/correct.txt
$B test -b CUDA0 -o MUL_MAT -p src1_scale > $OUT/range.txt 2>&1; echo "range rc=$? $(grep 'tests passed' $OUT/range.txt)" | tee -a $OUT/correct.txt
GGML_CUDA_NO_MMQ_VOLTA=1 $B test -b CUDA0 -o MUL_MAT -p src1_scale > $OUT/range-up.txt 2>&1; echo "range current path rc=$? $(grep 'tests passed' $OUT/range-up.txt)" | tee -a $OUT/correct.txt
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
log "done RESULTS $OUT"
