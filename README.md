# volta-fa

Fused FP16 attention and PCIe peer-to-peer tensor-parallel kernels for NVIDIA Volta (sm_70, V100), written in CUDA C++ and meant to plug into llama.cpp.

**Status: scaffold.** No kernels yet. Nothing here is benchmarked. Every performance claim in this repo must link to a file in `profiles/` or `bench/results/`; until it does, treat it as a hypothesis.

## Why

Used V100 32GB cards are cheap and common in homelabs and small AI shops, but modern attention work targets Ampere and newer. The goal is better long-context prefill and dual-GPU (no NVLink) decode on Volta:

1. **Prefill speed.** Less time to first token on large prompts.
2. **Context length.** Fused attention never materialises the N×N score matrix, so VRAM stops spiking with context.
3. **Model compatibility.** Head dims, GQA ratios and masking modes that the stock Volta path handles slowly or not at all.
4. **Dual-GPU smoothness.** Overlap tensor-parallel transfers with compute over PCIe P2P.
5. **Power and heat.** Fewer cycles at full draw, if (and only if) the benchmarks show it.

## Read this before trusting the plan

llama.cpp already ships a Volta-specific FlashAttention path: its `mma` kernel runs on sm_70 through the native `mma.m8n8k4` instruction, with Volta-specific dispatch. Any custom kernel has to beat that on specific shapes. The first job is therefore **Phase 0: measure the existing kernels and find the gap**. If a shape has no gap, we don't write a kernel for it. Until the V100s arrive we work on the pre-hardware track: reference and tests, static sm_70 reports, research, and the measurement kit. See [docs/ROADMAP.md](docs/ROADMAP.md) and [docs/research/](docs/research/).

## Ground rules

- Target: `sm_70` only, CUDA 12.x (CUDA 13 dropped Volta). Tensor cores via WMMA 16×16×16 or PTX `mma.m8n8k4`. No BF16, FP8, `cp.async`, `ldmatrix` or Turing/Ampere `mma` shapes. Details in [docs/INVARIANTS.md](docs/INVARIANTS.md).
- Correctness first: every kernel is checked against an FP32 reference before it is timed.
- Profile-driven: changes are justified by Nsight output stored in `profiles/`.

## Layout

```
src/attention/   fused attention kernels (tensor cores)
src/p2p/         peer-to-peer copy / overlap code
include/         public headers
tests/           correctness tests vs FP32 reference
bench/           microbenchmarks + results/
profiles/        committed nsys/ncu text summaries (raw reports are gitignored)
third_party/     llama.cpp clone (gitignored) + patches/ (committed)
tickets/         agent work tickets
bin/             multi-model orchestration + profiling helpers
prompts/         reusable prompts
docs/            ARCHITECTURE, INVARIANTS, ROADMAP, WORKFLOW, PROFILING, TOOLCHAIN, research/
```

## Build and test

Without a V100 or CUDA toolkit (see [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md) for the no-sudo setup), the CPU reference and tests build and run, and kernels get static sm_70 register/spill reports:

```bash
export PATH=$PWD/.venv/bin:$PATH   # once per shell
cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure
python bin/sm70-report.py src/attention/<file>.cu
```

On the V100 box (CUDA 12.9):

```bash
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=70 && cmake --build build && ctest --test-dir build --output-on-failure
```

## How we build it

An Opus 5.5 manager plus GPT-6.1 Sol workers (via Codex), adapted from the `opus-sol` playbook. See [docs/WORKFLOW.md](docs/WORKFLOW.md). Agent rules are in [CLAUDE.md](CLAUDE.md); review rules in [REVIEW.md](REVIEW.md).

## License

TBD (suggest Apache-2.0 or MIT to stay compatible with llama.cpp's MIT).
