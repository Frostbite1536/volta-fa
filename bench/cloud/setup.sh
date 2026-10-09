#!/bin/bash
# setup.sh: prepare a rented Ubuntu 22.04/24.04 V100 instance for bench/phase0.sh and
# bench/pr27997/run.sh. Run from the repo root. Does NOT install a driver (the cloud image
# provides one). See docs/CLOUD_V100.md.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 2
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo -E"
mkdir -p bench/results
LOG=bench/results/setup-$(date +%Y%m%d-%H%M%S).log
log() { echo "[setup] $*" | tee -a "$LOG"; }
# Never let apt/dpkg stop at an invisible prompt (tzdata, conffiles); keep its output in $LOG.
export DEBIAN_FRONTEND=noninteractive
APT=(apt-get -y -q -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)
apt_install() { $SUDO "${APT[@]}" install "$@" >> "$LOG" 2>&1; }

nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit --format=csv
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100 || { echo "not a V100" >&2; exit 2; }

log "base packages"
$SUDO "${APT[@]}" update >> "$LOG" 2>&1
apt_install ca-certificates git cmake ninja-build ccache build-essential python3-venv wget perl tmux \
    || { log "base package install failed; see $LOG"; tail -20 "$LOG"; exit 1; }

if ! /usr/local/cuda-12.8/bin/nvcc --version 2>/dev/null | grep -q "release 12.8"; then
    log "CUDA 12.8 toolkit (13 dropped sm_70; 12.9 has llama.cpp #28416)"
    . /etc/os-release
    distro="ubuntu${VERSION_ID//./}"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb \
        || { log "could not download the CUDA keyring for $distro"; exit 1; }
    $SUDO dpkg -i /tmp/cuda-keyring.deb >> "$LOG" 2>&1
    $SUDO "${APT[@]}" update >> "$LOG" 2>&1
    apt_install cuda-toolkit-12-8 || { log "cuda-toolkit-12-8 install failed; see $LOG"; tail -20 "$LOG"; exit 1; }
fi
export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc --version | tail -1

log "NCCL (for -sm tensor runs; harmless on 1 GPU)"
# The newest libnccl2 in NVIDIA's repo is a CUDA 13 build (no sm_70): pin the +cuda12.8 one.
ncclv=$(apt-cache madison libnccl2 2>/dev/null | awk -F'|' '/cuda12\.8/ {gsub(/ /, "", $2); print $2; exit}')
if [ -n "$ncclv" ] && apt_install "libnccl2=$ncclv" "libnccl-dev=$ncclv"; then log "  NCCL $ncclv"
else log "  NCCL for CUDA 12.8 not installed; tensor-split nccl runs will fall back"; fi

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
# Non-fatal: the upstream runs must not depend on the fork being reachable.
if { [ -d third_party/llama-superfast-v100/.git ] || git clone -q --filter=blob:none https://github.com/mylordmonkeyman/llama-superfast-v100.git third_party/llama-superfast-v100; } &&
   { git -C third_party/llama-superfast-v100 fetch -q origin "$fcommit" 2>/dev/null || true; } &&
   git -C third_party/llama-superfast-v100 checkout -q --detach "$fcommit"; then
    git -C third_party/llama-superfast-v100 log -1 --format='  fork %h %cs %s' | tee -a "$LOG"
else
    log "  fork clone/checkout failed; skip runbook row 2"
fi
# The fork's README builds with gcc-14 (Ubuntu 24.04 has it; 22.04 does not without a PPA).
if apt_install gcc-14 g++-14; then log "  gcc-14 installed (add -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 to EXTRA_CMAKE)"
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
if ! { [ -x .venv/bin/cmake ] || { python3 -m venv .venv >> "$LOG" 2>&1 && .venv/bin/pip install -q cmake ninja >> "$LOG" 2>&1; }; }; then
    log "  venv for the sanity build failed (does not affect the llama.cpp runs)"
fi
# venv cmake: Ubuntu 22.04 apt ships 3.22, volta-fa needs >= 3.24 (llama.cpp is fine with apt cmake).
if PATH=$PWD/.venv/bin:$PATH cmake -S . -B build-cpu -G Ninja >> "$LOG" 2>&1 && PATH=$PWD/.venv/bin:$PATH cmake --build build-cpu >> "$LOG" 2>&1; then
    .venv/bin/ctest --test-dir build-cpu 2>&1 | tail -1 || true
else log "  volta-fa sanity build failed (does not affect the llama.cpp runs); see $LOG"; fi

# shellcheck disable=SC2016  # $PATH must expand at login, not now
grep -q cuda-12.8/bin ~/.bashrc 2>/dev/null || echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo
echo "setup log: $LOG"
echo "next (CUDA 12.8 is now on PATH in new shells / tmux panes): export PATH=/usr/local/cuda-12.8/bin:\$PATH"
echo "      STEPS=\"env build support test perf\" bench/phase0.sh     # no models needed"
echo "      bench/pr27997/run.sh"
echo "      LLAMA=third_party/llama-superfast-v100 EXTRA_CMAKE=\"-DGGML_CUDA_CUB_3DOT2=ON -DGGML_NATIVE=OFF\" STEPS=\"env build support test perf\" bench/phase0.sh"
