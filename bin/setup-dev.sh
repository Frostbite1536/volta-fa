#!/bin/bash
# setup-dev.sh: no-sudo dev toolchain for a machine without a V100 or system CUDA (docs/TOOLCHAIN.md).
#   .venv/            cmake, ninja, shellcheck, numpy, NVRTC/ptxas/cudart wheels (CUDA 12.8)
#   tools/uv          pinned uv binary (python3-venv/pip may be missing)
#   tools/cuda-12.8/  full nvcc 12.8 + cudart + cuBLAS headers from conda-forge (optional: WITH_NVCC=1)
# Everything is gitignored and checksum-verified. Re-running is safe.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 2
UV_VERSION=0.12.17
MAMBA_VERSION=2.8.1-1
mkdir -p tools
fetch() {  # url dest: download and verify against the release's .sha256 file
    curl -fsSL -o "$2" "$1"
    curl -fsSL -o "$2.sha256" "$1.sha256"
    echo "$(awk '{print $1}' "$2.sha256")  $2" | sha256sum -c --quiet
    rm -f "$2.sha256"
}

if [ ! -x tools/uv ]; then
    echo "[setup-dev] uv $UV_VERSION"
    tmp=$(mktemp -d)
    fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" "$tmp/uv.tgz"
    tar xzf "$tmp/uv.tgz" -C "$tmp" && cp "$tmp/uv-x86_64-unknown-linux-gnu/uv" tools/uv && rm -rf "$tmp"
fi

echo "[setup-dev] .venv"
export UV_CACHE_DIR=$PWD/tools/.uv-cache
[ -x .venv/bin/python ] || tools/uv venv -q --python 3.12 .venv
tools/uv pip install -q --python .venv/bin/python cmake ninja shellcheck-py numpy \
    "nvidia-cuda-nvcc-cu12==12.8.*" "nvidia-cuda-nvrtc-cu12==12.8.*" \
    "nvidia-cuda-runtime-cu12==12.8.*" "nvidia-cuda-cccl-cu12==12.8.*"

if [ "${WITH_NVCC:-0}" = 1 ] && [ ! -x tools/cuda-12.8/bin/nvcc ]; then
    echo "[setup-dev] nvcc 12.8 via conda-forge (~2.4 GB)"
    [ -x tools/micromamba ] || { fetch "https://github.com/mamba-org/micromamba-releases/releases/download/$MAMBA_VERSION/micromamba-linux-64" tools/micromamba; chmod +x tools/micromamba; }
    export MAMBA_ROOT_PREFIX=$PWD/tools/mamba
    tools/micromamba create -y -q -p tools/cuda-12.8 -c conda-forge \
        "cuda-version=12.8" "cuda-nvcc=12.8" "cuda-cudart-dev=12.8" "cuda-cccl=12.8" "cuda-cuobjdump=12.8" "libcublas-dev=12.8"
    tools/micromamba clean -a -y -q > /dev/null
fi

echo "[setup-dev] check"
.venv/bin/python bin/sm70-report.py --help > /dev/null && echo "  sm70-report: ok"
[ -x tools/cuda-12.8/bin/nvcc ] && echo "  nvcc: $(tools/cuda-12.8/bin/nvcc --version | tail -1)"
echo "next: export PATH=\$PWD/.venv/bin:\$PATH; cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build"
