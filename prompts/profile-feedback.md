# Profile feedback prompt

Paste into the manager (or into a ticket Background section).

```
Kernel: <name>, launch <grid>x<block>, dyn smem <bytes>
sm70-report: regs=<n> spill_stores=<n> spill_loads=<n> stack=<n> smem=<n>
ncu: achieved occupancy <x>% (theoretical <y>%, limiter: <regs|smem|warps>), tensor pipe <x>%, DRAM <x>% of peak,
     bank conflicts <n> at <file:line ...>
Baseline: <commit> <metric>.

Constraints: Volta sm_70; wmma 16x16x16 or PTX mma.m8n8k4 only (no m16n8k*/ldmatrix/cp.async/BF16). Invariants: docs/INVARIANTS.md.
Task: propose ONE change targeting the largest measured limiter (e.g. smaller BR to cut registers, padding K tile rows by 8 halves to remove conflicts). State the expected effect on each metric above so it can be falsified by the next run.
```

One change per iteration, so the next profile attributes the effect.
