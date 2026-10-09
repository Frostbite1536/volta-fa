# Toolchain

## Version pins (INV-ARCH-003)

| Piece | Pin | Why |
| --- | --- | --- |
| CUDA toolkit | 12.8.x | CUDA 13.0 dropped sm_70 offline compilation; NVRTC 12.x already prints "Architectures prior to sm_75 are deprecated". **Not 12.9**: llama.cpp #28416 (open) reports sm_70 builds with 12.9.86 failing weight allocation on a 32 GB V100 while 12.8.93 works. Re-check before upgrading. |
| Driver | R580 branch | Last branch supporting Volta. The dev laptop already runs 580.178. |
| Nsight Compute | **unverified**: find the newest release that still profiles sm_70 before relying on `bin/profile-ncu.sh` | Recent Nsight Compute releases may have dropped Volta; check the release notes on the V100 box and record the version here. |
| Nsight Systems | ships with 12.8 | timelines only |
| llama.cpp | pinned in `bench/BASELINE.md` | INV-PERF-002 |

## Dev machine (no V100)

No sudo, no system CUDA. Everything lives in the gitignored `.venv/` and `tools/`:

```bash
# one-time, from the repo root
curl -fsSLO https://github.com/astral-sh/uv/releases/download/0.12.17/uv-x86_64-unknown-linux-gnu.tar.gz   # verify .sha256
UV_CACHE_DIR=$PWD/tools/.uv-cache tools/uv venv --python 3.12 .venv
UV_CACHE_DIR=$PWD/tools/.uv-cache tools/uv pip install --python .venv/bin/python \
  cmake ninja shellcheck-py numpy \
  "nvidia-cuda-nvcc-cu12==12.8.*" "nvidia-cuda-nvrtc-cu12==12.8.*" \
  "nvidia-cuda-runtime-cu12==12.8.*" "nvidia-cuda-cccl-cu12==12.8.*"
```

What works here:

- First, once per shell: `export PATH=$PWD/.venv/bin:$PATH` (puts `cmake`, `ninja`, `ctest`, `python`, `shellcheck` from the venv first).
- **CPU build + tests** (reference attention, tiled numerics model): `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build`. CMake finds no CUDA compiler and builds the CPU targets only.
- **Static sm_70 kernel reports**: `python bin/sm70-report.py <file.cu>` gives exact registers, spills, stack and static smem per kernel via NVRTC + ptxas. The pip `nvidia-cuda-nvcc-cu12` wheel ships `ptxas` but **not** the `nvcc` driver, which is why NVRTC is used.
- **Host-only CUDA code (P2P)** builds against the pip `libcudart` (auto-detected in `.venv`) and runs on the laptop GPU: `ctest` runs the staged-path tests for real; the peer test skips with one GPU.
- Lint: `shellcheck -x bin/*.sh`, `bin/check-forbidden.sh`.

What does not work here: running anything on sm_70 (the local GPU is a Maxwell Quadro M2200), timing, ncu/nsys metrics, the peer (2-GPU) P2P path.

## V100 box (when built)

1. Driver R580, CUDA 12.8 toolkit (full, with `nvcc`, Nsight Systems, a Volta-capable Nsight Compute).
2. `cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=70` builds the CUDA targets too.
3. Record `nvidia-smi -q`, `nvidia-smi topo -m`, `nvcc --version`, `ncu --version` in `bench/BASELINE.md`.
4. Allow profiling without root if needed: `NVreg_RestrictProfilingToAdminUsers=0`.
5. For P2P: check BIOS (Above 4G decoding, ACS/IOMMU settings), consider the `iommu=pt` kernel parameter (the llama-superfast-v100 P2P all-reduce requires it), and run `p2pBandwidthLatencyTest` from cuda-samples.
