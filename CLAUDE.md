# volta-fa agent instructions

Fused FP16 attention + PCIe P2P kernels for NVIDIA Volta (sm_70). CUDA C++17, CMake. Baseline and integration target: llama.cpp (`third_party/llama.cpp`, gitignored clone; our changes live as patches in `third_party/patches/`).

## Hard rules (every agent, every ticket)

- **Volta sm_70 ONLY.** Tensor cores via `nvcuda::wmma` (16×16×16 half) or PTX `mma.sync.aligned.m8n8k4`. Do NOT use BF16, FP8, `cp.async`, `ldmatrix`, `mma.sync` m16n8k*, TMA, clusters, or code copied from FlashAttention-2/3. If you think you need one, STOP and say so.
- **CUDA 12.x only** (12.9). CUDA 13 dropped sm_70.
- Every kernel change reports registers, spill stores/loads, stack and smem (`bin/sm70-report.py`, or `--ptxas-options=-v` on the V100 box). Spill > 0 is a failing result.
- Never write the N×N score matrix to global memory in the fused path.
- Correctness before speed: compare against the reference in `tests/`. Do not loosen tolerances or edit existing tests to pass; report the real error.
- Do not invent benchmark numbers. Anything not run on a V100 is "not run".
- Read `docs/INVARIANTS.md` before changing kernel behaviour; changing an invariant needs a manager decision.
- Never read or commit `.env`, tokens, model weights, or raw `*.ncu-rep`/`*.nsys-rep`.
- Workers: no git commands that write; leave changes uncommitted. The manager commits.
- Stay in the ticket's file scope. Out-of-scope need → STOP and explain.

## Commands

```bash
# dev machine (no CUDA compiler: CPU targets only). Setup: docs/TOOLCHAIN.md
export PATH=$PWD/.venv/bin:$PATH   # once per shell
cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure
python bin/sm70-report.py src/attention/<file>.cu   # static regs/spill/smem for sm_70
bin/check-forbidden.sh && shellcheck -x bin/*.sh

# V100 box
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=70 && cmake --build build && ctest --test-dir build --output-on-failure
bin/profile-ncu.sh <binary> [args...]                        # text summary into profiles/
```

## Conventions

- Files < ~1,500 lines; one kernel family per file.
- Tile sizes are template parameters (`BR`, `BC`, `D`, warps); no magic numbers inline.
- Tensor layout, mask and causal semantics are defined in `include/volta_fa/attention.h`; do not reinterpret them per kernel.
- Every CUDA API call and launch is checked (`VFA_CUDA_CHECK`).
- Comments explain *why* (layout, padding, occupancy trade-off), not what.
- Commits say why each test exists. Worker-written code: "Implemented by <worker> from ticket <name>; verified by <manager>".

## Docs map

[ARCHITECTURE](docs/ARCHITECTURE.md) · [INVARIANTS](docs/INVARIANTS.md) · [ROADMAP](docs/ROADMAP.md) · [WORKFLOW](docs/WORKFLOW.md) · [PROFILING](docs/PROFILING.md) · [TOOLCHAIN](docs/TOOLCHAIN.md) · [research/](docs/research/) · [REVIEW.md](REVIEW.md) · [prompts/](prompts/)
