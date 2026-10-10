Laptop GPU (Quadro M2200, sm_52, 2 MB L2), NOT a V100. Relative numbers only.
unpacked = llama.cpp 79e2e74 + patches 0001-0006 (third_party/p6-work/build-sm52)
packed   = same + third_party/patches/next/0007 (GQA-packed VEC; third_party/p8c4-work/build-sm52)
Perf: test-backend-ops perf with a local-only case line (D 128/256, nh=4, GQA 1/2/4/6/8, KV 4K/16K, 1 token,
q8_0/q4_0), 3 interleaved rounds per build; summary.md = bench/analyze.py tbo medians.
fa-test.txt.gz: packed build, full FLASH_ATTN_EXT: 4260/4260 passed (after the KQ buffer sizing fix; the first
build failed 14 D=64 GQA>4 cases, see docs/design/P8-change4-packed-vec.md).
