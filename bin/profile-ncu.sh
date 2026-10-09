#!/bin/bash
# profile-ncu.sh <binary> [args...]
# Runs Nsight Compute and writes a TEXT summary to profiles/. Raw .ncu-rep goes to
# profiles/raw/ (gitignored). Paste only the summary into model prompts.
# Volta needs an Nsight Compute release that still supports sm_70; see docs/TOOLCHAIN.md.
# shellcheck source=bin/env.sh
. "$(dirname "$0")/env.sh"
command -v ncu >/dev/null || { echo "ncu not found (needs CUDA 12.x toolkit on the V100 box)" >&2; exit 2; }
[ $# -ge 1 ] || { echo "usage: $0 <binary> [args...]" >&2; exit 2; }
mkdir -p "$VFA_PROFILES/raw"
TAG=$(basename "$1")-$(date +%Y%m%d-%H%M%S)
OUT="$VFA_PROFILES/$TAG.txt"
{
  echo "# $TAG"
  echo "commit: $(git -C "$VFA_HOME" rev-parse --short HEAD 2>/dev/null)$(git -C "$VFA_HOME" diff --quiet || echo ' (dirty)')"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem --format=csv,noheader | head -1)"
  echo "cuda: $(nvcc --version 2>/dev/null | tail -1)"
  echo "ncu: $(ncu --version | tail -1)"
  echo "cmd: $*"
  echo
} > "$OUT"
ncu --set full --import-source yes -o "$VFA_PROFILES/raw/$TAG" -f "$@" > /dev/null || { echo "ncu failed" >&2; exit 1; }
ncu --import "$VFA_PROFILES/raw/$TAG.ncu-rep" --page details >> "$OUT"
echo "$OUT"
