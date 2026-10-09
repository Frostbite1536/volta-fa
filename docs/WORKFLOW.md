# Multi-model workflow (adapted from opus-sol)

Source: `/home/p51/Github/opus-sol/PLAYBOOK.md`. Same shape (manager writes narrow tickets, worker implements, manager verifies and owns git), adapted for GPU work where **the profiler is the judge**, not either model.

## Roles

| Manager (Opus 5.5) | Worker (GPT-6.1 Sol via Codex) |
| --- | --- |
| Phase decisions, kernel design (tiling, smem layout, register budget), ticket writing | Scaffolding: CMake, test harness, FP32 reference, bench drivers, P2P boilerplate, llama.cpp patch plumbing |
| Interprets ncu/nsys output, decides next optimisation | First-draft kernels from a precise design ticket; mechanical refactors; rebases |
| Verifies on hardware, reads critical diffs, owns git | Stops and explains when a rule blocks it |
| Reviews kernels, anything touching numerics or invariants | Routine read-only reviews (`bin/worker-review.sh`) |

The pasted plan suggested Opus writes all kernels and Sol only scaffolds. Keep that as the default for the **hot** kernels, but let Sol draft kernels from a manager-written design ticket when the design is already decided; Opus then does the profile-driven rewrites.

## The hardware loop (replaces "run the tests" for kernel work)

1. Manager writes ticket from `tickets/TEMPLATE.md` including the register/smem budget and target occupancy.
2. `bin/worker.sh <name> <worktree> <ticket>` (background, stdin closed).
3. Manager runs `bin/sm70-report.py` (works without a GPU) or builds with `-v` on the box; reads regs, spill, smem. Spill > 0 → back to worker/self with the numbers.
4. `ctest` for parity. Failing parity blocks everything else.
5. `bin/profile-ncu.sh` → text summary in `profiles/<kernel>-<date>.txt`.
6. Feed **only the summary** (key metrics + flagged source lines) back into the next ticket using `prompts/profile-feedback.md`.
7. Repeat until the exit criterion for the phase is met or the gap is shown not to exist.
8. Commit with the profile summary referenced in the message.

## Review tiers

| Change | Review |
| --- | --- |
| CMake, docs, bench drivers, test plumbing | Worker read-only + CI |
| Any kernel, numerics, masking, softmax | Manager-spawned independent reviewer (fresh context), told the invariants to check |
| P2P lifetime/streams/events | Independent reviewer (races hide here) |
| Performance claims | Manager re-measures; single runs are not evidence (3+ interleaved rounds, medians) |

## Known failure modes to put in every kernel ticket

- **Ampere bias**: models reach for `mma.sync.m16n8k16`, `ldmatrix`, `cp.async`, BF16. The rules block forbids them; `bin/sm70-report.py` (compiles at compute_70) and `bin/check-forbidden.sh` catch what slips through. Note the inverse trap: m8n8k4 *is* legal on Volta, so don't let a reviewer "fix" it to WMMA.
- **Register spills / low occupancy** while the code looks perfect. Read ptxas first.
- **Shared-memory bank conflicts**: give the model the ncu-flagged lines, not a guess.
- **Invented numbers**: workers must write "not run" when no V100 was available.
- **Tests edited to pass**: manager reads every edit to an existing test.

## Mechanics carried over from opus-sol

- Stdin must be closed for `codex exec` (`< /dev/null`).
- Do not pass `--add-dir`; worker cannot write outside its worktree.
- Worker cannot commit; manager commits after verifying.
- Worktrees live in `.worktrees/` (git-excluded). Tickets repeat the rules block verbatim; workers do not read CLAUDE.md on their own.
- Receipts land in `state/receipts/` (gitignored); read them with `tail`/`grep`, not whole.

## Setup note

`bin/env.sh` holds machine-specific paths. `WORKER_MODEL` is copied from opus-sol (`gpt-6.1-sol`). Worktrees: `git worktree add .worktrees/<name> -b <branch> main`. The worker builds into its own `build/` inside the worktree using the main checkout's `.venv` by absolute path; tickets spell this out.
