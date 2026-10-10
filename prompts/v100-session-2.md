# Guided V100 session 2: patch series A/B (prompt for a fresh Claude Code session)

Paste the block below into a new session started in `~/Github/volta-fa`.

```
You are guiding me (the repo owner) through a rented cloud V100 session for ~/Github/volta-fa, and you run
the commands yourself once I give you SSH access. Goal: measure llama.cpp patch series 0001-0006
(third_party/patches/) on a real V100 with bench/patches/run.sh, bring the results home, analyse them against
the acceptance criteria, and record the findings.

Before anything else, read: CLAUDE.md, docs/CLOUD_V100.md (§0 is the quick path for this session),
docs/design/P6-small-batch-gqa.md and docs/design/P8-quantised-kv-decode.md ("Accept if" sections),
docs/research/phase0-cloud-2026-10-10.md (session 1 results), prompts/v100-session-2.md (this file: the
checklist and rules below). Then give me a short plan and the rental instructions.

Follow the checklist and rules in prompts/v100-session-2.md.
```

## Checklist for the guiding session

1. **Rent** (owner does this on vast.ai; guide them):
   - Filter: 1× Tesla V100, **PCIE 32 GB** preferred (same part as the home box). SXM2 is ~10% faster and must
     be labelled as such. ≥ 80 GB disk (7 llama.cpp builds + ccache + ~25 GB of models), Ubuntu 22.04/24.04,
     driver R535–R580. Session 1 paid $0.163/hr.
   - Use the owner's saved custom template: image `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-<date>`,
     on-start script **keeps `entrypoint.sh`** (it installs the SSH key; clearing it broke SSH in session 1).
     Vast's stock CUDA templates filtered out every V100 in session 1. Direct SSH connection on.
   - The owner pastes the "Direct ssh" line. Connect with
     `ssh -i ~/.ssh/vastai_ed25519 -o IdentitiesOnly=yes -p <PORT> root@<IP>`. If direct fails, try the proxy
     line. Verify with `nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv`.
2. **Move the repo and set up** (docs/CLOUD_V100.md §0): bundle with `--all`, scp, clone, start `tmux new -s vfa`,
   `bench/cloud/setup.sh`, `export PATH=/usr/local/cuda-12.8/bin:$PATH`. Start the two model downloads in a
   second tmux window right away; they finish while the builds run.
3. **Run 1 (no models):** `bench/patches/run.sh 2>&1 | tee patches-kernel.log`
   (conversion microbench → 7 builds → full FLASH_ATTN_EXT per variant, MUL_MAT for upstream and conv8 →
   3 interleaved kernel rounds). Expect `upstream` correctness to **abort** (known Volta crash, session 1 §4).
   Every patched variant must be **0 FAIL**. The first build takes 15–30 min, later ones are mostly ccache.
   - Check progress with `tail -5 bench/results/patches-*/run.log` rather than streaming build output.
   - When run 1 ends, **copy its results home immediately** (tar + scp, §0), before run 2.
4. **Run 2 (models, llama-bench only):**
   `STEPS="build lb" MODELS="/root/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf /root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf" bench/patches/run.sh 2>&1 | tee patches-models.log`
   (~1.5–2.5 h). Copy results home again when it ends.
5. **Tell the owner to destroy the instance** in the Vast UI once results are on the laptop and verified
   (`ls bench/results/patches-*`). There is no persistent volume.
6. **Analyse on the laptop** (`summary.md` in each results dir; `python3 bench/analyze.py` for re-cuts):
   - P-6 acceptance: 0 new failures; GQA 6 at 8/16 tokens, 32K: `p6cfg` ≥ 1.3× upstream; GQA 4/8 within noise.
     Also report the SMALL_B decision data (padded `<8,8>` vs upstream `<16,2>` at 16 tokens) and B ≤ 4.
   - P-8 acceptance: 0 new failures (FA and MUL_MAT); `qkv` ≥ 2× `p6cfg` at 1 token, 32K, GQA 4 (q8_0 and
     q4_0); `conv8` ≥ 1.5× `qkv` at 8 tokens, 32K, quantised; Llama-3.1-8B q8_0 tg64 at 16K ≥ +25% over upstream;
     no f16 case outside noise. Microbench: 0005/0006 bit-identical, V100 speedups.
   - Change 4 gate (P-8 doc): if `qkv` at 1 token, 32K takes ≥ 1.6× as long at GQA 8 as at GQA 4 (equal KV
     bytes), recommend designing GQA-packed VEC.
   - Compare with the model predictions (`python bin/qkv-decode-model.py`, `python bin/smallbatch-model.py`)
     and say where the model was wrong.
7. **Record:** `docs/research/patches-cloud-<date>.md` (same style as phase0-cloud-2026-10-10.md: hardware, what
   ran, tables with medians and ranges, what passed/failed against each criterion, what is not measured). Update
   docs/ROADMAP.md, the "Status" lines of both design docs, and the memory file. Commit the results text files
   and docs (the owner's repo, local git, no remote).

## Rules

- Never invent or extrapolate numbers. Anything not run is "not run". Report failures with their output.
- If a patched variant has FAIL > 0 or crashes: stop, rerun the failing case alone with
  `test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p '<params>'`, and diagnose before spending more rented
  time. Do not edit tests or tolerances to pass.
- Variance: decode on this host had ±5–20% run-to-run spread in session 1; claim nothing inside the spread.
  Warm-GPU vs cool-GPU runs differ (session 1: a 56 °C run was the outlier); `run.log` records clocks and
  temperature per round.
- Never read or commit `.env`, tokens or model weights; do not commit raw `.ncu-rep`/`.nsys-rep`.
- llama.cpp's contribution rules require AI-use disclosure and forbid AI-written issue/PR text, comments and
  replies. Do not draft or post anything to llama.cpp. Give the owner the facts (numbers, patch files,
  repro commands) so they can write their own posts.
- Ask before anything that costs money or is hard to undo (renting, destroying, large downloads beyond the two
  models). Keep the owner posted with short status lines during long runs.
