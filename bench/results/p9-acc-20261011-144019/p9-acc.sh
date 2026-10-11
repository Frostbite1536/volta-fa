#!/usr/bin/env bash
# On the V100 after p9-q6b: accuracy against an FP32 reference on the same quantized weights. The reference
# (GGML_CUDA_F32_REF, dev switch) dequantizes every quantized weight to FP32 and multiplies with cuBLAS SGEMM (Volta has
# no TF32); upstream's path and P-9 are each compared with it by llama-perplexity KL divergence. Batches of 16 tokens
# everywhere, so the non-matmul kernels (attention etc.) are the same in all runs.
# Usage: p9-acc.sh TAG WAITLOG
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
TAG=$1; WAIT=$2
until grep -q "RESULTS\|FAILED" $WAIT 2>/dev/null; do sleep 15; done
R=~/volta-fa; W=$R/third_party/patches-ab/rev
OUT=$R/bench/results/$TAG-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT
log() { echo "[$(date +%T)] $* $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" | tee -a $OUT/run.log; }
cat $W/docs/*.md $W/docs/*/*.md 2>/dev/null | head -c 120000 > $OUT/text.txt; wc -c $OUT/text.txt > $OUT/text.size
PP=$W/build/bin/llama-perplexity
A="-c 512 -b 16 -ub 16 -ngl 99 -fa on --chunks 8"
for M in /root/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf /root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf; do
  m=$(basename $M .gguf)
  log "$m ref"
  GGML_CUDA_F32_REF=1 $PP -m $M -f $OUT/text.txt $A --kl-divergence-base $OUT/$m.ref.kld > $OUT/$m.ref.txt 2>&1
  for v in up p9 p9all; do
    case $v in
      up)    E="GGML_CUDA_NO_MMQ_VOLTA=1" ;;
      p9)    E="GGML_CUDA_MMQ_VOLTA_X=0" ;;
      p9all) E="GGML_CUDA_MMQ_VOLTA_MIN=2 GGML_CUDA_MMQ_VOLTA_MINN=1" ;;
    esac
    log "$m $v"
    env $E $PP -m $M $A --kl-divergence-base $OUT/$m.ref.kld --kl-divergence > $OUT/$m.$v.txt 2>&1
  done
  rm -f $OUT/$m.ref.kld
done
log "done RESULTS $OUT"
