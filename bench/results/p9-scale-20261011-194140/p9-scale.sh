#!/usr/bin/env bash
# On the V100: activation scaling (each 256-value slice of a token scaled by a power of two before the FP16 conversion,
# undone in FP32; src1-precision guard). Two builds with the same new tests: "new" = p9-scale.diff (exact candidate),
# "old" = p9-final4.diff kernel (unscaled). ggml is linked statically, so the copied binaries are independent.
# Correctness (old must fail the range tests), per-matmul and end-to-end timing interleaved, logits old vs new (KLD).
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
R=~/volta-fa; W=$R/third_party/patches-ab/rev
OUT=$R/bench/results/p9-scale-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cp ~/p9-scale.diff $0 $OUT/
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
prep() { git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/ && for p in ~/patches-rev4/*.patch $1; do git -C $W apply $p || { echo APPLY_FAILED $p | tee -a $OUT/run.log; exit 1; }; done; }
build() { cmake -S $W -B $W/build > /dev/null 2>&1; cmake --build $W/build --target test-backend-ops llama-bench llama-perplexity > $OUT/build-$1.log 2>&1 || { echo BUILD_FAILED $1 | tee -a $OUT/run.log; exit 1; }; rm -rf ~/bin-$1; mkdir ~/bin-$1; cp $W/build/bin/{test-backend-ops,llama-bench,llama-perplexity} ~/bin-$1/; }
log "build new"; prep ~/p9-scale.diff; cp $W/tests/test-backend-ops.cpp ~/tbo-new.cpp; build new
cuobjdump -res-usage $W/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq-volta.cu.o 2>&1 | grep -A1 -E "Function .*(mul_mat_q_volta|mmq_volta_prep)" | paste - - | sed -E 's/\s+/ /g' > $OUT/resources.txt
log "build old"; prep ~/p9-final4.diff; cp ~/tbo-new.cpp $W/tests/test-backend-ops.cpp; build old
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
log correctness
~/bin-new/test-backend-ops test -b CUDA0 -o MUL_MAT > $OUT/mulmat-new.txt 2>&1; echo "new MUL_MAT rc=$? $(grep 'tests passed' $OUT/mulmat-new.txt)" | tee $OUT/correct.txt
~/bin-new/test-backend-ops test -b CUDA0 -o MUL_MAT -p 'src1_scale' > $OUT/range-new.txt 2>&1; echo "new range rc=$? $(grep 'tests passed' $OUT/range-new.txt)" | tee -a $OUT/correct.txt
~/bin-old/test-backend-ops test -b CUDA0 -o MUL_MAT -p 'src1_scale' > $OUT/range-old.txt 2>&1; echo "old range rc=$? $(grep 'tests passed' $OUT/range-old.txt)" | tee -a $OUT/correct.txt
FK='type_a=(q4_K|q5_K|iq4_xs|q6_K),type_b=f32,m=(17408|5120|6144|10240),n=(8|16),k=(5120|17408|6144),bs=\[1,1\],nr=\[1,1\]'
F8='type_a=q8_0,type_b=f32,m=(14336|4096),n=(9|16),k=(4096|14336),bs=\[1,1\],nr=\[1,1\]'
for i in 1 2; do
  log "perf r$i"
  for v in old new; do for f in "$FK" "$F8"; do ~/bin-$v/test-backend-ops perf -b CUDA0 -o MUL_MAT -p "$f" --output sql 2>/dev/null; done > $OUT/$v.r$i.sql; done
  for f in "$FK" "$F8"; do GGML_CUDA_NO_MMQ_VOLTA=1 ~/bin-new/test-backend-ops perf -b CUDA0 -o MUL_MAT -p "$f" --output sql 2>/dev/null; done > $OUT/up.r$i.sql
done
for i in 1 2; do
  for M in /root/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf /root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf; do
    m=$(basename $M .gguf)
    log "lb $m r$i"
    GGML_CUDA_NO_MMQ_VOLTA=1 ~/bin-new/llama-bench -m $M -dev CUDA0 -ngl 99 -fa on -r 5 -o jsonl -p 1,2,4,8,12,16 -n 0 -d 0 > $OUT/lb-up.$m.r$i.jsonl 2> $OUT/lb-up.$m.r$i.err
    ~/bin-old/llama-bench -m $M -dev CUDA0 -ngl 99 -fa on -r 5 -o jsonl -p 1,2,4,8,12,16 -n 0 -d 0 > $OUT/lb-old.$m.r$i.jsonl 2> $OUT/lb-old.$m.r$i.err
    ~/bin-new/llama-bench -m $M -dev CUDA0 -ngl 99 -fa on -r 5 -o jsonl -p 1,2,4,8,12,16 -n 0 -d 0 > $OUT/lb-p9.$m.r$i.jsonl 2> $OUT/lb-p9.$m.r$i.err
  done
done
TXT=$(ls -d $R/bench/results/p9-acc-*/text.txt | head -1)
A="-c 512 -b 16 -ub 16 -ngl 99 -fa on --chunks 8"
for M in /root/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf /root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf; do
  m=$(basename $M .gguf)
  log "kld $m"
  ~/bin-old/llama-perplexity -m $M -f $TXT $A --kl-divergence-base $OUT/$m.old.kld > $OUT/$m.old.txt 2>&1
  ~/bin-new/llama-perplexity -m $M $A --kl-divergence-base $OUT/$m.old.kld --kl-divergence > $OUT/$m.new-vs-old.txt 2>&1
  rm -f $OUT/$m.old.kld
done
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | tail -1)
log nsys
$NSYS profile -t cuda --cuda-graph-trace=node -o $OUT/nsys-p9-pp16 --force-overwrite true ~/bin-new/llama-bench -m /root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf -dev CUDA0 -ngl 99 -fa on -r 5 -p 16 -n 0 -d 0 -o jsonl > /dev/null 2>&1
$NSYS stats --report cuda_gpu_kern_sum --format csv -o $OUT/kern-p9-pp16 $OUT/nsys-p9-pp16.nsys-rep > /dev/null 2>&1
python3 /root/shapes.py $OUT/nsys-p9-pp16.sqlite > $OUT/shapes-p9-pp16.txt 2>&1
rm -f $OUT/nsys-p9-pp16.nsys-rep $OUT/nsys-p9-pp16.sqlite
log "done RESULTS $OUT"
