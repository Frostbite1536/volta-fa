# llama-superfast-v100 `feature/ridgefill`: one rule worth keeping (desk research, 2026-11-06)

Desk research only: nothing here was run by us. All numbers are the fork's own, from its commit messages and
`RIDGEFILL.md`.

Source: `andreasknopke/llama-superfast-v100` branch `feature/ridgefill` at `0eb1f47` (RidgeFill commits `ac9b6e1`,
`78bebea`, `a27a2cf`, `0eb1f47`, 2026-11-03).

## Verdict

**Nothing in this branch is worth adopting for volta-fa except one cache-invalidation rule (below).**

RidgeFill is a lossy, model-level prefill approximation. For tokens before an exact tail, it runs only layers
0–15 and fills the K/V of later layers from a fitted linear projector. Its own doc says the output is not
identical to stock and "expect some quality loss". That fails reference parity (INV-NUM-003) by design, and it adds
no Volta kernel work.

The long-context prefill drop it works around is already partly addressed exactly. The branch's Volta D=256 FA
configs equal our patch 0003 rows (up to the stage count, which has no effect on Volta).

## The rule worth keeping

**What happened.** With RidgeFill on, the server re-read the whole prompt on every chat step:
- 17.8K-token prompt, steps adding one line each;
- 12.1–15.0 s per step before the fix, 73–150 ms after (fork's V100 measurements, `78bebea`).

**The cause.** Two bookkeeping checks dropped the cached prefix even though no cached cell's contents had changed:
- a fill boundary that moves on every step by construction;
- a reordered selection of exact rows.

Qwen3.8-27B is a hybrid model: its recurrent (GDN) state cannot rewind partially. So for it, any invalidation
dropped the entire cache rather than a suffix. The fix invalidates only when what the cached range must contain
changes (another mode or another vtail setting).

**The rule.** On hybrid or recurrent models, a cache invalidation that does not reflect a real change in cached
contents costs a full prompt re-read, not a suffix. Any future change of ours that keys or invalidates the KV cache
must invalidate only on content changes.

**When it applies.** None of our patches 0001–0008 touch the prompt cache, and our end-to-end numbers come from
`llama-bench`, which has no prompt cache. The rule applies only if we ever touch the cache or benchmark multi-turn
through `llama-server`. In that case, check that the server's `cache_n` timing equals the shared prefix length;
that is how the fork saw the bug (`cache_n` was 0).
