# Cloud V100 session runbook

Goal of a few rented hours: run the Phase 0 kit and the PR #27997 A/B early, so that the
hardware week starts with data. Everything below runs without models first; models are
the last, optional step.

## 1. Pick the instance

- **Prefer V100 32 GB PCIe** (same part as the home box: 1380 MHz boost, 250 W). SXM2 parts boost
  to 1530 MHz at 300 W and run ~10% faster; 16 GB parts limit model size. Any V100 is fine for
  kernel-level numbers, but `env.txt` must record which one, and SXM2 numbers are not the home
  box's numbers.
- One GPU is enough. Multi-V100 cloud instances usually have NVLink, which says nothing about
  the home box's PCIe P2P; skip the `multi` step there.
- Ubuntu 22.04 or 24.04 image with the NVIDIA driver installed (R535–R580 all run CUDA 12.8
  sm_70 binaries). Root or sudo. ~60 GB disk (llama.cpp builds + ccache + optional models).

## 2. Move the repo (it has no remote)

```bash
# laptop
git -C ~/Github/volta-fa bundle create /tmp/volta-fa.bundle --all
scp /tmp/volta-fa.bundle <host>:
# instance
git clone volta-fa.bundle volta-fa && cd volta-fa
```

## 3. Set up (~15–25 min, mostly the CUDA download)

```bash
bench/cloud/setup.sh
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

It installs CUDA 12.8 (not the driver), ninja, ccache and NCCL, checks out llama.cpp at
`bench/LLAMA_COMMIT`, reports whether this Nsight Compute can profile GV100, and runs the
volta-fa CPU tests.

## 4. Runs, in priority order

| # | Command | Needs | Time (rough) | Answers |
| --- | --- | --- | --- | --- |
| 1 | `STEPS="env build support test perf" bench/phase0.sh` | nothing | build 15–30 min, runs ~10 min | which FA cases Volta supports/fails; kernel timings incl. D=256 decode and quantised KV |
| 2 | `LLAMA=third_party/llama-superfast-v100 EXTRA_CMAKE="-DGGML_CUDA_CUB_3DOT2=ON -DGGML_NATIVE=OFF" STEPS="env build support test perf" bench/phase0.sh` | nothing | one more build | the same kernel-level numbers for the superfast fork (pinned in `bench/SUPERFAST_COMMIT`): does it beat upstream, and on which shapes? Add the gcc-14 flags setup printed if the default compiler fails |
| 3 | `bench/pr27997/run.sh` | nothing | 4 builds (ccache makes 2–4 fast) | `pr` vs upstream vs `comb64`/`pr_comb64`: regs/stack, correctness, D=256 prefill kernel time |
| 4 | `MODEL=… bench/pr27997/run.sh` | a D=256 GGUF | +llama-bench per variant | end-to-end prefill at 4k/16k/64k depth |
| 5 | `bench/phase0.sh` with `bench/models.txt`, for upstream and (with `LLAMA=…`) the fork | 2–4 GGUFs | ~30–60 min each | depth / KV-quant / small-batch / ubatch sweeps; small-batch at depth is where the fork re-routes decode to tensor cores |
| 6 | `bin/profile-ncu.sh …` | Volta-capable ncu + profiling permission | as needed | only if step 3 of setup said ncu works |

Use `tmux` so a dropped SSH session does not kill a run.

## 5. Bring results home

```bash
# instance
tar czf results.tgz bench/results
# laptop
scp <host>:volta-fa/results.tgz . && tar xzf results.tgz -C ~/Github/volta-fa
```

Tables for the write-up:

```bash
R=bench/results/pr27997-<stamp>
python3 bench/analyze.py tbo $R/upstream.perf.sql $R/pr.perf.sql $R/comb64.perf.sql $R/pr_comb64.perf.sql
python3 bench/analyze.py tbo bench/results/phase0-llama.cpp-*/fa-perf.sql bench/results/phase0-llama-superfast-v100-*/fa-perf.sql --labels upstream,superfast
python3 bench/analyze.py lb <upstream>.jsonl <fork>.jsonl --labels upstream,superfast
```

Commit the text/sql/jsonl results (they are small). Never commit models or raw `.ncu-rep`.
Note the instance type, GPU variant (PCIe/SXM2, 16/32 GB) and price in `bench/BASELINE.md`.
