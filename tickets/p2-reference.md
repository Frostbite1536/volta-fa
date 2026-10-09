# Ticket p2-reference: CPU attention reference, tiled numerics model, and tests

Your current directory is a git worktree of volta-fa (branch `p2-reference`, based on `main` @ `01faf22`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Volta sm_70 ONLY, CUDA 12.x. (This ticket is CPU-only C++17; no CUDA code.)
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/volta-fa/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Never install anything. Build into `build/` inside this worktree.
- Layout/mask/softcap/causal semantics are FIXED by `include/volta_fa/attention.h`. Do not change that header. If it cannot express something you need, STOP and explain.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- No third-party test framework and no downloads: write a minimal assertion helper yourself.
- Never invent numbers. Report the error statistics your code actually printed.
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
We cannot run GPU code for a month. Before any kernel exists we want:
1. an exact reference everyone trusts (`attention_reference`, double accumulation), and
2. a CPU model of the fused-kernel algorithm (`attention_tiled_model`) that reproduces its rounding points (FP16 P, optional FP16 QK accumulate, FP32 running max/sum, per-tile rescale, FP16 output). Its error against the reference tells us what tolerance a correct GPU kernel can be held to, so tolerances come from measurement, not guesses.

Read first: `include/volta_fa/attention.h` (the contract, including the logit formula, GQA mapping and bottom-right causal rule), `docs/INVARIANTS.md` (INV-NUM-001..004).

## Files you may touch
- `src/reference/CMakeLists.txt`: new; static library `volta_fa_ref` (CPU), public include dir `include/`.
- `src/reference/attention_reference.cpp`: new; `validate`, `AttnParams::effective_scale`, `attention_reference`, `round_to_fp16`.
- `src/reference/attention_tiled_model.cpp`: new; `attention_tiled_model`.
- `tests/CMakeLists.txt`: new; registers tests with `add_test`.
- `tests/check.h`: new; minimal CHECK helpers.
- `tests/test_fp16.cpp`, `tests/test_reference.cpp`, `tests/test_tiled_model.cpp`: new.
- `tests/error_budget.cpp`: new; executable (registered as a test that only fails on NaN/crash) that prints the error table described below.
Nothing else. The top-level `CMakeLists.txt` already adds `src/reference` and `tests` if they exist.

## Requirements

### round_to_fp16
Bit-exact IEEE binary16 round-to-nearest-even, including subnormals, overflow to ±inf (values ≥ 65520 round to inf), NaN preserved. Do not use `_Float16` or compiler extensions (this header will later be compiled by nvcc's host pass); do it with integer bit manipulation.

### attention_reference
Straightforward, double accumulation, two-pass softmax (max, then sum) per row. Must handle: GQA, d_qk != d_v, mask null or not, softcap, causal with seq_q < seq_kv, fully masked rows (O=0, LSE=-inf, no NaN), and `lse == nullptr`. Call `validate` first; on invalid params, throw `std::invalid_argument` with the reason.

### validate
Reject: non-positive dims, n_head_q % n_head_kv != 0, negative softcap, non-finite scale. Accept seq_q > seq_kv (rows with no visible key under causal are just fully masked).

### attention_tiled_model
Same outputs as the reference, but computed the way a fused kernel does:
- For each (batch, q head), process query rows in blocks of `br` and K/V in tiles of `bc`, in increasing kv order; tails (seq not multiple of br/bc) handled.
- Logits in float: dot product in float; if `qk_accum_fp16`, round the running sum to FP16 after every 16-element chunk of d_qk. Then scale/softcap/mask/causal exactly as the contract says.
- Per row: running max `m` and sum `l` in float; for each tile: `m_new = max(m, max_j x_j)`; `alpha = exp(m - m_new)`; `p_j = exp(x_j - m_new)` (rounded to FP16 if `p_fp16`); `l = l*alpha + sum_j p_j` (sum of the possibly-rounded p, in float); `acc = acc*alpha + sum_j p_j * v_j` with v_j taken as given (float); careful with `m = -inf` before any visible key (no `-inf - -inf` NaN).
- Final `O = acc / l` (0 if l == 0), rounded to FP16 if `o_fp16_out`; `LSE = m + log(l)` (−inf if l == 0).
- Skip work for tiles fully masked by causal (it must not change results).

### Tests (one per behaviour; each must fail if the behaviour is broken)
- `test_fp16`: known bit patterns: 1.0, 65504, 65520→inf, 6.1e-5 (min normal), 5.96e-8 (min subnormal), a halfway case for RNE ties-to-even, -0.0, NaN.
- `test_reference`:
  - hand-computed tiny case (seq 2, d 2) with exact expected output;
  - uniform logits → O = mean of V rows;
  - GQA mapping (n_head_q=8, n_head_kv=2): head h must use kv head h/4 (construct V so the wrong mapping gives a different answer);
  - causal bottom-right with seq_q=3, seq_kv=5 (row 0 sees keys 0..2);
  - fully masked row via mask → O=0, LSE=-inf, no NaN anywhere;
  - softcap matches the formula on a case where tanh saturates;
  - large logits (scale such that |x| ~ 1e4) → finite output equal to the argmax V row;
  - validate rejects each invalid case.
- `test_tiled_model`: with all rounding flags off, matches the reference within 1e-5 abs on a matrix of shapes, including tails (seq_kv=1, 63, 65, 1000 with bc=64), br/bc of 16/32/64/128, causal on/off, GQA, d 64/80/128/256, seq_q < seq_kv, and a fully masked row. Results must be identical with and without the causal-tile skip (expose nothing new; test it by comparing causal=true against causal=false plus an equivalent explicit mask).

### error_budget (prints, does not assert tolerances)
Inputs rounded through `round_to_fp16`, seeded `std::mt19937`, fixed seeds. For each config in {p_fp16 only; p_fp16+o_fp16_out; qk_accum_fp16+p_fp16+o_fp16_out} and each input distribution in {Q,K,V ~ N(0,1); Q,K ~ N(0,1) with 4 "outlier" channels ×20; logits scaled so softmax is near one-hot} and shapes {d=64,128,256} × {seq_kv=512, 4096} (seq_q=128, causal and not): print one row with max abs error, max rel error (relative to max|O_ref| of that case), and mean abs error of O vs the reference, plus max abs LSE error. Also print the number of non-finite values (must be 0 except LSE=-inf on fully masked rows; any other non-finite value makes the test fail). Keep total runtime under ~60 s on a laptop (reduce batch/heads to 1 if needed).

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build` with no warnings from our sources (`-Wall -Wextra` is on)
- `ctest --test-dir build --output-on-failure`
- `bin/check-forbidden.sh`
- Negative controls: temporarily break (a) the GQA mapping, (b) causal alignment to top-left, (c) the `m=-inf` guard in the tiled model; confirm the matching test fails each time; restore.

## Done means
All files above exist, build clean, ctest green. Final report: files changed; each command with its result line; the full error_budget table as printed; which negative controls failed as expected; anything you could not do. Do not choose tolerances; the manager does that from your table.
