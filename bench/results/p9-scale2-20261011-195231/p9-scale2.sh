#!/usr/bin/env bash
# On the V100: activation scaling, per 256-value slice (p9-slice.diff: scales undone per superblock) vs per token
# (p9-tok.diff: undone once at the store), against the unscaled kernel (p9-final4.diff, with the same new tests) and
# the current path. ggml is linked statically, so the copied binaries are independent. Accuracy: llama-perplexity KL
# divergence of each against an FP32 reference (p9-ref.diff = per-token + GGML_CUDA_F32_REF dev switch).
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
R=~/volta-fa; W=$R/third_party/patches-ab/rev
OUT=$R/bench/results/p9-scale2-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; cp ~/p9-slice.diff ~/p9-tok.diff ~/p9-ref.diff $0 $OUT/
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
prep() { git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/ && for p in ~/patches-rev4/*.patch $1; do git -C $W apply $p || { echo APPLY_FAILED $p | tee -a $OUT/run.log; exit 1; }; done; }
perfline() { t=$W/tests/test-backend-ops.cpp; anchor='test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776'
  L='    for (ggml_type ta : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) { for (auto mk : std::vector<std::pair<int,int>>{{17408, 5120}, {5120, 17408}, {5120, 6144}, {6144, 5120}, {10240, 5120}, {14336, 4096}, {4096, 14336}, {4096, 4096}, {48, 5120}}) { for (int n : {8, 16}) { test_cases.emplace_back(new test_mul_mat(ta, GGML_TYPE_F32, mk.first, n, mk.second, {1, 1}, {1, 1})); } } } // p9 scale perf' \
  perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "$ENV{L}\n" }' $t; }
build() { cmake -S $W -B $W/build > /dev/null 2>&1; cmake --build $W/build --target test-backend-ops llama-bench llama-perplexity > $OUT/build-$1.log 2>&1 || { echo BUILD_FAILED $1 | tee -a $OUT/run.log; exit 1; }; rm -rf ~/bin-$1; mkdir ~/bin-$1; cp $W/build/bin/{test-backend-ops,llama-bench,llama-perplexity} ~/bin-$1/; }
log "build slice"; prep ~/p9-slice.diff; cp $W/tests/test-backend-ops.cpp ~/tbo-new.cpp; perfline; build slice
cuobjdump -res-usage $W/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq-volta.cu.o 2>&1 | grep -A1 -E "Function .*(mul_mat_q_volta|mmq_volta_prep)" | paste - - | sed -E 's/\s+/ /g' > $OUT/resources-slice.txt
log "build tok"; prep ~/p9-tok.diff; perfline; build tok
cuobjdump -res-usage $W/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq-volta.cu.o 2>&1 | grep -A1 -E "Function .*(mul_mat_q_volta|mmq_volta_prep)" | paste - - | sed -E 's/\s+/ /g' > $OUT/resources-tok.txt
log "build old"; prep ~/p9-final4.diff; cp ~/tbo-new.cpp $W/tests/test-backend-ops.cpp; perfline; build old
log "build ref"; prep ~/p9-ref.diff; build ref
git -C $W checkout -q -- . && git -C $W clean -fdq ggml/src/ggml-cuda/
log correctness
for v in slice tok; do
  ~/bin-$v/test-backend-ops test -b CUDA0 -o MUL_MAT > $OUT/mulmat-$v.txt 2>&1; echo "$v MUL_MAT rc=$? $(grep 'tests passed' $OUT/mulmat-$v.txt)" | tee -a $OUT/correct.txt
done
for v in old slice tok; do
  ~/bin-$v/test-backend-ops test -b CUDA0 -o MUL_MAT -p src1_scale > $OUT/range-$v.txt 2>&1; echo "$v range rc=$? $(grep 'tests passed' $OUT/range-$v.txt)" | tee -a $OUT/correct.txt
done
GGML_CUDA_NO_MMQ_VOLTA=1 ~/bin-tok/test-backend-ops test -b CUDA0 -o MUL_MAT -p src1_scale > $OUT/range-up.txt 2>&1; echo "up range rc=$? $(grep 'tests passed' $OUT/range-up.txt)" | tee -a $OUT/correct.txt
F='type_a=(q4_K|q5_K|iq4_xs|q6_K|q8_0),type_b=f32,m=(17408|5120|6144|10240|14336|4096|48),n=(8|16),k=(5120|17408|6144|4096|14336),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1,src_overlap=0,m_v=0,pad=0$'
for i in 1 2 3; do
  log "perf r$i"
  for v in old slice tok; do ~/bin-$v/test-backend-ops perf -b CUDA0 -o MUL_MAT -p "$F" --output sql 2>/dev/null > $OUT/$v.r$i.sql; done
  GGML_CUDA_NO_MMQ_VOLTA=1 ~/bin-tok/test-backend-ops perf -b CUDA0 -o MUL_MAT -p "$F" --output sql 2>/dev/null > $OUT/up.r$i.sql
done
for i in 1 2; do
  for M in /root/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf /root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf; do
    m=$(basename $M .gguf)
    log "lb $m r$i"
    GGML_CUDA_NO_MMQ_VOLTA=1 ~/bin-tok/llama-bench -m $M -dev CUDA0 -ngl 99 -fa on -r 5 -o jsonl -p 1,4,8,12,16 -n 0 -d 0 > $OUT/lb-up.$m.r$i.jsonl 2> $OUT/lb-up.$m.r$i.err
    for v in old slice tok; do
      ~/bin-$v/llama-bench -m $M -dev CUDA0 -ngl 99 -fa on -r 5 -o jsonl -p 1,4,8,12,16 -n 0 -d 0 > $OUT/lb-$v.$m.r$i.jsonl 2> $OUT/lb-$v.$m.r$i.err
    done
  done
done
TXT=$(ls -d $R/bench/results/p9-acc-*/text.txt | head -1)
A="-c 512 -b 16 -ub 16 -ngl 99 -fa on --chunks 8"
for M in /root/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf /root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf; do
  m=$(basename $M .gguf)
  log "kld $m ref"
  GGML_CUDA_F32_REF=1 ~/bin-ref/llama-perplexity -m $M -f $TXT $A --kl-divergence-base $OUT/$m.ref.kld > $OUT/$m.ref.txt 2>&1
  GGML_CUDA_NO_MMQ_VOLTA=1 ~/bin-tok/llama-perplexity -m $M $A --kl-divergence-base $OUT/$m.ref.kld --kl-divergence > $OUT/$m.up.txt 2>&1
  for v in old slice tok; do
    log "kld $m $v"
    ~/bin-$v/llama-perplexity -m $M $A --kl-divergence-base $OUT/$m.ref.kld --kl-divergence > $OUT/$m.$v.txt 2>&1
  done
  rm -f $OUT/$m.ref.kld
done
log "done RESULTS $OUT"
