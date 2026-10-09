#!/bin/bash
# Fails if source uses device features newer than sm_70 (INV-ARCH-001).
# Allowed on Volta: nvcuda::wmma (16x16x16 half) and PTX mma.sync.aligned.m8n8k4.
# NVRTC/nvcc at compute_70 rejects most of these anyway; this catches inline PTX
# and code paths hidden behind #if __CUDA_ARCH__ guards.
cd "$(dirname "$0")/.." || exit 2
PATTERN='__nv_bfloat16|nv_bfloat162?|cuda_bf16\.h|cuda_fp8\.h|__nv_fp8|\.e4m3|\.e5m2|cp\.async|__pipeline_memcpy_async|memcpy_async|mma\.sync\.aligned\.m16n8|m16n8k(8|16|32)|ldmatrix|stmatrix|movmatrix|wgmma|setmaxnreg|cluster_group|cuda/barrier|cp\.async\.bulk|cp\.reduce\.async'
dirs=()
for d in src include tests bench; do [ -d "$d" ] && dirs+=("$d"); done
[ ${#dirs[@]} -eq 0 ] && { echo "ok (nothing to scan)"; exit 0; }
if grep -rnE "$PATTERN" "${dirs[@]}" --include='*.cu' --include='*.cuh' --include='*.h' --include='*.hpp' --include='*.cpp'; then
  echo "forbidden sm_75+ feature found (INV-ARCH-001)" >&2; exit 1
fi
echo "ok"
