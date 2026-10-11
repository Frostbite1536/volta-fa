# Ticket p9-qmm-table: weight-shape and matmul-bandwidth tabulator for the P-9 design

Your current directory is a git worktree of volta-fa (branch `p9-qmm`, based on `main` @ `8802adb`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Volta sm_70 ONLY, CUDA 12.x. (This ticket has no GPU code.)
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/volta-fa/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Never install anything. Build into `build/` inside this worktree.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no V100 on this machine: anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
P-9 is a design for a Volta tensor-core "skinny" quantised matmul (1-16 tokens). The design doc needs tables of
(a) which quantised weight shapes a model has and how many bytes an evaluation reads, and (b) measured
`test-backend-ops perf -o MUL_MAT` times turned into effective weight bandwidth. Inputs exist:
- `bench/results/gguf-tensors-20261011/dump-*.txt`: `gguf_dump.py` output. Tensor lines look like
  `      9:   16777216 |  4096,  4096,     1,     1 | Q8_0    | blk.0.attn_output.weight`
  (index, element count, ne0..ne3, type, name). ne0 = K (input dim), ne1 = N (output rows).
- Perf SQL from test-backend-ops (`--output sql`): the format is what `bench/analyze.py` already parses
  (`load_tbo_one`, `load_tbo`, `parse_params`; a comma-joined file list = rounds, summarised as median/min/max/n).
  MUL_MAT op_params look like `type_a=q4_K,type_b=f32,m=17408,n=4,k=5120,bs=[1,1],nr=[1,1],...`
  (m = N output rows, k = K, n = tokens). Reuse `bench/analyze.py` by importing it (see how
  `bench/patches/check.py` does `sys.path.insert` + `import analyze`); do not copy its parser.

## Design (manager decided, do not change)
New script `bin/qmatmul-table.py`, Python 3 standard library only, three subcommands, Markdown tables on stdout:
1. `shapes DUMP [DUMP...]`: per dump file (model), the matmul weights = 2-D tensors (ne1 > 1, ne2 == ne3 == 1)
   that are not F32 and whose name does not start with `token_embd`. Group by (type, N, K, role) where role is the
   name with a leading `blk.<i>.` removed (e.g. `ffn_up.weight`). Columns: role, type, K, N, count, MiB (total).
   Then per-type totals (elements, MiB) and the total bytes of all matmul weights, with `output.weight` also shown
   separately. Bytes per weight from a table of ggml block sizes (bytes per block / elements per block):
   Q4_0 18/32, Q4_1 20/32, Q5_0 22/32, Q5_1 24/32, Q8_0 34/32, Q2_K 84/256, Q3_K 110/256, Q4_K 144/256,
   Q5_K 176/256, Q6_K 210/256, IQ4_NL 18/32, IQ4_XS 136/256, IQ3_S 110/256, F16 2/1, BF16 2/1. Unknown type: print it
   in an "unknown type" line and exclude it from byte totals (never guess).
2. `perf SQL_GROUP [SQL_GROUP...] [--labels a,b]`: for MUL_MAT rows, one table per input group: rows (type_a, m, k),
   columns n (sorted), cell = median µs, and a second table with effective weight GB/s = bytes(type_a, m*k) / time
   (bytes from the same block table). If a group has several rounds, also print the max (max-min)/median spread.
3. `step DUMP SQL_GROUP --tokens 1,2,4,8,16`: estimated matmul time per evaluation for that model at each token count
   = sum over the model's (type, N, K) groups of count x measured median time for (type, m=N, k=K, n). Groups without
   a measured shape are estimated with the same type's median GB/s at that n over the measured shapes (state this in
   the output) and listed; if the type has no measurement at all, report it as "not measured" and exclude it.
   Print coverage = bytes of exactly measured groups / total bytes. Exclude `output.weight` from the sum by default
   (`--with-output` includes it) and say so in the output.

## Files you may touch
- `bin/qmatmul-table.py`: new.
- `tests/fixtures/qmm-dump.txt`, `tests/fixtures/qmm-r1.sql`, `tests/fixtures/qmm-r2.sql`: new, small synthetic fixtures
  (a few tensor lines incl. one F32 norm, one token_embd, one unknown type; a few MUL_MAT rows; values chosen so the
  expected numbers are easy to check by hand).
- `tests/CMakeLists.txt`: add tests next to `analyze_lb_rounds`, inside the same `if` block, with
  PASS_REGULAR_EXPRESSION checks of hand-computed values for each subcommand.
Nothing else.

## Steps
1. Read `bench/analyze.py` (loaders) and `tests/CMakeLists.txt` (how the analyze tests are registered).
2. Write the script and fixtures; compute the expected values by hand in comments in the fixture or CMake file.
3. Run it on the real dumps: `python bin/qmatmul-table.py shapes bench/results/gguf-tensors-20261011/dump-*.txt`
   and paste the output in your report (no SQL for the real models exists yet; that is fine).

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build`
- `ctest --test-dir build --output-on-failure` (all tests, including the existing ones)
- `bin/check-forbidden.sh`
- `python -m py_compile bin/qmatmul-table.py`

## Done means
The script and tests exist, all ctest tests pass. Final report: files changed; each command with its result line;
the `shapes` output for the two real dumps; what was NOT verified. Do not decide whether to ship.
