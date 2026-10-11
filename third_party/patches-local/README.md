# Local-only patches

Applied after `third_party/patches/0001-0008`, never meant for upstream.

- `0009-tests-MUL_MAT-with-activations-beyond-FP16-s-range-l.patch`: `test_mul_mat_range`, 35 MUL_MAT cases with
  src1 scaled by 1e5 or 1e-5 (and /64, /4096 per 256-value slice), for the Volta kernel's activation scaling
  (patch 0008). Kept local because the current q8_1 paths fail about half of these cases (block_q8_1 keeps its scale
  and sum in FP16), so they would fail on unmodified upstream. Run with
  `test-backend-ops test -b CUDA0 -o MUL_MAT -p src1_scale`. Results: `bench/results/p9-scale2-*`, `p9-verify-*`;
  details in `docs/design/P9-small-batch-qmatmul.md` ("Phase 2, fourth part").
