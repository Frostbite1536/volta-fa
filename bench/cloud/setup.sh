#!/bin/bash
# setup.sh: prepare a rented Ubuntu 22.04/24.04 V100 instance for bench/phase0.sh and
# bench/pr27997/run.sh. Run from the repo root. Does NOT install a driver (the cloud image
# provides one). See docs/CLOUD_V100.md.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 2
SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
log() { echo "[setup] $*"; }

nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit --format=csv
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100 || { echo "not a V100" >&2; exit 2; }

log "base packages"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq git cmake ninja-build ccache build-essential python3-venv wget perl > /dev/null

if ! /usr/local/cuda-12.8/bin/nvcc --version 2>/dev/null | grep -q "release 12.8"; then
    log "CUDA 12.8 toolkit (13 dropped sm_70; 12.9 has llama.cpp #28416)"
    . /etc/os-release
    distro="ubuntu${VERSION_ID//./}"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
    $SUDO dpkg -i /tmp/cuda-keyring.deb > /dev/null
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq cuda-toolkit-12-8 > /dev/null
fi
export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc --version | tail -1

log "NCCL (for -sm tensor runs; harmless on 1 GPU)"
$SUDO apt-get install -y -qq libnccl2 libnccl-dev > /dev/null 2>&1 || log "  NCCL install failed; tensor-split nccl runs will fall back"

log "llama.cpp at the pinned commit"
commit=$(cat bench/LLAMA_COMMIT)
if [ ! -d third_party/llama.cpp/.git ]; then
    git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
fi
git -C third_party/llama.cpp fetch -q origin "$commit" 2>/dev/null || true
git -C third_party/llama.cpp checkout -q --detach "$commit"
git -C third_party/llama.cpp log -1 --format='llama.cpp %h %cs %s'

log "llama-superfast-v100 fork at the pinned commit (A/B against upstream in Phase 0)"
fcommit=$(cat bench/SUPERFAST_COMMIT)
if [ ! -d third_party/llama-superfast-v100/.git ]; then
    git clone -q --filter=blob:none https://github.com/mylordmonkeyman/llama-superfast-v100.git third_party/llama-superfast-v100
fi
git -C third_party/llama-superfast-v100 checkout -q --detach "$fcommit"
# The fork's README builds with gcc-14 (Ubuntu 24.04 has it; 22.04 does not without a PPA).
if $SUDO apt-get install -y -qq gcc-14 g++-14 > /dev/null 2>&1; then log "  gcc-14 installed (add -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 to EXTRA_CMAKE)"
else log "  gcc-14 unavailable: try the fork with the default compiler"; fi

log "profiler capabilities"
if command -v ncu > /dev/null; then
    ncu --version | tail -1
    if ncu --list-chips 2>/dev/null | tr ',' '\n' | grep -qi gv100; then log "  ncu supports GV100"
    else log "  this ncu does NOT list GV100 (Volta was removed in Nsight Compute 2025.3): use the 2025.1.x from CUDA 12.8; skip bin/profile-ncu.sh until then"; fi
fi
if [ -r /proc/driver/nvidia/params ] && grep -q "RmProfilingAdminOnly: 1" /proc/driver/nvidia/params; then
    log "  profiling restricted to admin (RmProfilingAdminOnly=1): run ncu as root or skip it"
fi

log "volta-fa CPU tests (sanity)"
python3 -m venv .venv > /dev/null && .venv/bin/pip install -q cmake ninja > /dev/null
# venv cmake: Ubuntu 22.04 apt ships 3.22, volta-fa needs >= 3.24 (llama.cpp is fine with apt cmake).
.venv/bin/cmake -S . -B build-cpu -G Ninja > /dev/null && .venv/bin/cmake --build build-cpu > /dev/null && .venv/bin/ctest --test-dir build-cpu | tail -1

echo
echo "next: export PATH=/usr/local/cuda-12.8/bin:\$PATH"
echo "      STEPS=\"env build support test perf\" bench/phase0.sh     # no models needed"
echo "      bench/pr27997/run.sh"
echo "      LLAMA=third_party/llama-superfast-v100 EXTRA_CMAKE=\"-DGGML_CUDA_CUB_3DOT2=ON -DGGML_NATIVE=OFF\" STEPS=\"env build support test perf\" bench/phase0.sh"
