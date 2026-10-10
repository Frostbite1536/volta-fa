Laptop GPU (Quadro M2200, sm_52), NOT a V100. llama.cpp 79e2e74 + third_party/patches/0001..0006,
built with nvcc 12.8.93 -DCMAKE_CUDA_ARCHITECTURES=52 (third_party/p6-work/build-sm52).
Maxwell routes quantised/BF16 FA at >2 tokens through TILE + the K/V conversion (patches 0005/0006);
the Volta-only selection changes (0001/0002/0004) are not exercised here.

fa.txt.gz          test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT          4200/4200 passed
mulmat-sub.txt.gz  ... -o MUL_MAT -p 'type_a=(q4_0|q8_0|q4_1|q5_0|q5_1|bf16|f16|f32),'   1297/1297 passed
mulmat.txt.gz      ... -o MUL_MAT (unfiltered)   aborted at a q4_K non-contiguous case (no q4_K nc converter;
                   see docs/design/P8-quantised-kv-decode.md "Local GPU correctness")
