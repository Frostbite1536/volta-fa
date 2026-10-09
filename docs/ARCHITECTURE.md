# Architecture

## Components

```
            llama.cpp (third_party/, pinned commit)
                  │  ggml CUDA backend hook
                  ▼
        ┌─────────────────────┐        ┌──────────────────────┐
        │ include/volta_fa/   │        │ src/p2p/             │
        │ public C API        │        │ peer access, copies, │
        └─────────┬───────────┘        │ overlap w/ compute   │
                  ▼                    └──────────┬───────────┘
        ┌─────────────────────┐                   │
        │ src/attention/      │◄── tile/launch ───┘ (split-KV / TP partials)
        │ fused TC kernels    │
        └─────────────────────┘
```

| Component | Responsibility | Depends on | Interface |
| --- | --- | --- | --- |
| `src/attention/` | Fused QKᵀ → online softmax → PV on tensor cores (WMMA or m8n8k4); KV tiling in shared memory | CUDA, `include/` | `volta_fa_attention(...)` launch wrappers |
| `src/p2p/` | Enable peer access, async peer copies, event-based overlap with compute, host-staged fallback | CUDA runtime | `volta_fa_p2p_*` |
| `include/volta_fa/attention.h` | Tensor layout, mask/causal semantics, reference API: the contract every kernel and test shares | none | header |
| `tests/` | CPU reference, tiled numerics model, parity tests across shapes | `include/` | ctest |
| `bench/` | Microbenchmarks, baseline comparison, result files | both + llama.cpp | `bin/bench.sh` |
| llama.cpp integration | Patch/fork exposing the kernels behind a flag | all | kept as a patch series in `third_party/patches/` |

## Data flow (attention)

1. Q tile (BR rows) loaded to shared memory/registers; loop over K/V tiles (BC rows).
2. `S = Q·Kᵀ` on tensor cores, FP32 accumulate in registers.
3. Online softmax: update running max `m`, sum `l` (FP32), rescale accumulator.
4. `P` cast to FP16 → `O += P·V` (through shared memory with WMMA; it can stay in registers with m8n8k4).
5. Write O (and LSE if needed). No N×N tensor reaches global memory (INV-NUM-001).

## Key design decisions (to be filled in with evidence)

| Decision | Status | Evidence |
| --- | --- | --- |
| WMMA vs PTX m8n8k4 | open, leaning m8n8k4 (known fragment layout; llama.cpp precedent) | design doc + static reports |
| Tile sizes (BR, BC) per head dim (64/80/96/128/256) | open | planning calculator, then ncu |
| FP16 vs FP32 accumulate for S | open | error vs. reference + speed |
| Bank-conflict padding scheme for K/V shared tiles | open | ncu `l1tex__data_bank_conflicts` |
| Single-kernel vs. split-KV for long context / small batch | open | occupancy at 32k+ |
| P2P: `cudaMemcpyPeerAsync` vs. IPC handles | open; IPC only matters across processes, and llama.cpp is one process | Phase 3 topology tests |

## Failure modes and limits

- No V100 → CPU tests and static sm_70 reports only; hardware results are "not run" (INV-PROC-002).
- PCIe topology determines P2P bandwidth (x8 vs x16, shared switch, ACS settings). Measure with `nvidia-smi topo -m` and a bandwidth test before designing overlap.
- Register pressure vs. occupancy is the main performance cliff; see docs/PROFILING.md.
