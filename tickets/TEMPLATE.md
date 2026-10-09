# Ticket @NAME@: @ONE-LINE GOAL@

Your current directory is a git worktree of volta-fa (branch `@BRANCH@`, based on `main` @ `@SHA@`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Volta sm_70 ONLY, CUDA 12.x. Tensor cores via `nvcuda::wmma` (16x16x16 half) or PTX `mma.sync.aligned.m8n8k4`. Do NOT use BF16, FP8, cp.async, ldmatrix, mma m16n8k*, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=@REPO@/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Never install anything. Build into `build/` inside this worktree.
- For every kernel you touch run `python bin/sm70-report.py <file.cu>` and paste its table. Spill must be 0.
- Never write the NxN score matrix to global memory (INV-NUM-001). Layout/mask/causal semantics are fixed by `include/volta_fa/attention.h`.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no V100 on this machine: anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
@WHY. Link the invariant IDs. Give file:line anchors. For optimisation tickets paste the profile summary (kernel, launch config, regs, spill, smem, occupancy, tensor-pipe %, DRAM %, top bank-conflict lines).@

## Design (manager decided, do not change)
@tile sizes, smem layout/padding, warp->tile mapping, register/smem budget, target occupancy@

## Files you may touch
- @path@: @what may change@
Nothing else.

## Steps
1. @...@

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build`
- `ctest --test-dir build -R @focused tests@ --output-on-failure`
- `bin/check-forbidden.sh`
- `python bin/sm70-report.py @kernel files@` (kernel tickets only)

## Done means
@exact state@. Final report: files changed; each command with its result line; sm70-report table; which negative controls failed as expected; what was NOT verified. Do not decide whether to ship.
