# Profiling loop

Run on the V100 box. Commit **text summaries** to `profiles/`; raw reports stay local.

## Phase 0 / system view

```bash
nsys profile --stats=true -o profiles/raw/llama-prefill \
  ./third_party/llama.cpp/build/bin/llama-cli -m model.gguf -f prompt.txt -n 64 -ngl 99 -fa on
nvidia-smi topo -m
```

Look for: time split between attention and matmul kernels, gaps between kernels, PCIe copy time in dual-GPU runs.

## Kernel view (ncu)

```bash
bin/profile-ncu.sh ./build/bench/bench_attention --seq 32768 --d 128
```

Use an Nsight Compute release that still supports sm_70 (see docs/TOOLCHAIN.md). Metrics to read (check names against your version with `ncu --query-metrics`):

| Question | Metric |
| --- | --- |
| Tensor cores busy? | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` (Volta name; differs on newer archs) |
| Memory-bound? | `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes.sum` |
| Occupancy limiter | `sm__warps_active.avg.pct_of_peak_sustained_active`, launch stats (regs/thread, smem/block) |
| Bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`, source-level view with `-lineinfo` |
| Spills | `bin/sm70-report.py` or ptxas `-v`; at runtime `l1tex__t_sectors_pipe_lsu_mem_local*` |

## Feeding results back to a model

Paste only: kernel name, launch config, regs/thread, spill bytes, smem/block, achieved occupancy, tensor-pipe %, DRAM %, top bank-conflict source lines. Use `prompts/profile-feedback.md`. Do not paste whole reports.

## Rules

- Same input, same clocks (`nvidia-smi -lgc` if permitted), warm-up runs discarded.
- 3+ interleaved rounds, report medians and spread.
- Record GPU model, driver, CUDA version, commit, baseline commit with every summary.
