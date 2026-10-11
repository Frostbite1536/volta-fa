# volta-fa in plain language

*What was done, why, and what it achieved, written for readers without a GPU programming background. Every number
here was measured on real hardware; the technical details and raw data are linked at the end.*

## The short version

We made a popular free AI program, **llama.cpp**, run noticeably faster on an older but still capable graphics card,
the **NVIDIA Tesla V100**. There are eight changes in all:

- One fixes a crash.
- The others remove specific slowdowns.

On two real AI models, the biggest results were:

| situation | speed-up |
| --- | --- |
| Working on 8–16 words ("tokens") at a time | about **38% faster** (Qwen, a 27-billion-parameter model) and **27% faster** (Llama, 8 billion) |
| Writing text with a compressed memory, deep into a long conversation | **27% faster** (compression level q8_0) to **56% faster** (stronger compression, q4_0) |
| Reading very long documents (64,000 tokens of context) | about **45% faster** |

The largest change also makes the arithmetic *more* accurate than before, not less.

## Background, in everyday terms

**The graphics card.** AI models run on GPUs (graphics cards), because GPUs do thousands of small calculations at
once. The V100 came out in 2017. It is no longer the newest, but it has 32 GB of fast memory and special
"tensor core" circuits for AI maths, and it is widely available second-hand. The owner is building a machine with
two of them.

**llama.cpp.** A widely used open-source program that runs AI language models on your own computer. Most of its
speed work targets newer cards, so older cards like the V100 get less attention. That gap is where this project
works.

**Tokens.** Models read and write text in small pieces called tokens (roughly a word or part of a word). Sometimes
the program handles one token at a time, as when it writes a reply. Sometimes it handles a handful at once:

- **speculative decoding:** a small model guesses several words ahead and the big model checks them in one go;
- several users at once;
- reading a prompt in chunks.

Those "small batches" of 2–16 tokens turned out to be a weak spot.

**The two big jobs inside a model.**
- *Attention* is how the model looks back over everything said so far.
- *Matrix multiplication* is the bulk arithmetic of applying the model's learned weights.

Both have to move huge amounts of data from the card's memory, so speed is mostly about **moving data efficiently**,
not about raw calculating power.

**Compression ("quantisation").** Model weights are usually stored in compressed form (for example 4–8 bits per
number instead of 16) so big models fit in memory. The model's short-term memory of the conversation, the **KV
cache**, can be compressed too. Compression saves memory, but the data has to be unpacked before or while it is used,
and unpacking costs time.

## How the work was done

- The owner's own V100 machine arrives around November 2026. Until then, work happened in two places:
  - **On a laptop without a suitable GPU:** code, tests, and checks that predict how the code will behave on a V100
    without running it.
  - **On a rented V100 in the cloud** (Vast.ai), in sessions on 10–11 October 2026, for every real measurement.
- The code was written by AI agents (Claude as the manager, with other AI models as workers on some tasks). The
  owner set the goals and made the decisions.
- **Strict rules** applied throughout:
  - No speed claim without a measurement on a real V100.
  - Every change is checked for correctness against a reference before it is timed.
  - Comparisons are made in the same session, alternating old and new versions, so that a warm card or a busy
    machine doesn't skew the result.
- Changes are kept as small "patches" against a fixed version of llama.cpp, so they could later be offered to the
  llama.cpp project. llama.cpp's rules require any submission to disclose AI use and to be written by a person, so
  any submission is the owner's to write.

## What we found first

The project began with a wrong assumption: that llama.cpp had no fast attention for the V100. It does. So the first
cloud session simply **measured** where the existing code was slow, rather than writing a replacement. It found
four problems:

1. **A crash:** certain small-batch requests made llama.cpp stop with an error on the V100.
2. **Small batches with "grouped" attention** read the same data several times over.
3. **A compressed conversation memory** was being fully unpacked on every single word written, which cost more time
   than the attention itself.
4. **Long documents** used settings designed for newer cards, which overloaded the V100's working storage.

A later investigation found a fifth and bigger opportunity: in the matrix arithmetic, not in attention at all
(see change 8).

## The eight changes

**1. Crash fix.** Two parts of the program disagreed about which version of a calculation to use. On the V100, one
of those versions doesn't exist, so the program stopped. Now both parts use the same rule. *Result: the crash is
gone, and the full attention test suite passes.*

**2. Grouping work for odd group sizes.** In many models, several "question" heads share one set of memory
("grouped" attention). When the group size was an odd number like 3, 5, 6 or 7, the V100 code read the shared memory
several times. Now it pads the group to a convenient size and reads the memory once. *Result, together with change 3:
attention up to 3.9× faster in those cases; Qwen 6–20% faster overall on small batches with a long conversation. Two small slowdowns
remain in rare short-context cases.*

**3. Better settings for big attention heads.** For one common model shape, the V100 borrowed settings from newer
cards, which overflowed its fast working storage and spilled into slow memory. New settings fit. *Result: that
attention step is 2.3–2.65× faster; reading a 64,000-token document with Qwen is 45% faster.* An existing public
proposal (PR #27997) already covers part of this.

**4. Choosing the right tool for one-word-at-a-time with compressed memory.** For writing one word, the program now
uses a method that reads the compressed memory directly, instead of unpacking all of it first. Another card family
(Turing) already did this. *Result: that step 2–3× faster.*

**5 and 6. Faster unpacking.** Where unpacking is still needed, it now keeps more of the card busy (5) and moves data
in bigger pieces (6). The output is identical bit for bit. *Result: unpacking 1.7–6.9× faster.* These two changes
help every NVIDIA card, not just the V100, but so far they have only been measured on the V100 and on an older
laptop GPU.

**7. Sharing memory reads between heads when writing.** When writing one word with a compressed memory, each head in
a group re-read and re-unpacked the same shared memory. Now 2–4 heads share one read. *Result, together with
changes 4–6: Llama writes 27% faster at long context with q8_0 compression and 56% faster with q4_0, close to the
speed of uncompressed memory (5% behind it instead of 25%).* One rare case became 2–5% slower. The first version of
this change was actually slower; it took a second design to make it pay.

**8. A new engine for the main arithmetic at small batches.** This is the biggest change.
- *What we noticed:* when handling 5–16 tokens at once, the V100 moved model weights at only a third to a half of its
  memory speed. One path unpacked the weights again for every token. The other used a calculating method the V100
  has no special hardware for.
- *What the new code does:* it reads each compressed weight once, unpacks it with a few exact tricks, and feeds the
  V100's tensor cores (the hardware built for exactly this maths).
- *Speed:* Qwen is **34–39% faster** at 8–16 tokens and Llama **27% faster** at 12–16 tokens. Nothing changes for
  1–4 tokens.
- *Accuracy:* answers are **6–16× closer** to a full-precision reference than the old method's. The old method
  squeezed the model's working numbers into 8 bits; the new one keeps about 11.

## An honest note about one safety feature

The plan for change 8 promised a safety step. Before converting the model's working numbers to the card's compact
16-bit format (which can't hold values above 65,504), each group of 256 numbers would be scaled into a safe range.
Scaling by a power of two is exact, so it loses nothing. During development that step was left out, because the test
models never came close to the limit, and the plan wasn't updated to say so. A separate review caught the
difference. On the owner's instruction the step was restored and measured:

- it costs Qwen 1–2% of the speed gain;
- it costs Llama nothing;
- it leaves accuracy unchanged;
- extreme test inputs that previously produced wrong results now come out right.

We also found that the *existing* llama.cpp method has a similar limit with very large inputs. That is a separate
finding the owner may report.

## What didn't work or is still open

- **Some ideas lost on the real card,** like the first version of change 7. Those measurements are kept rather than
  hidden.
- **A few small slowdowns remain** in rare cases (changes 2 and 7). They are listed in the documents.
- **Other graphics cards:** changes 5, 6 and 7 touch code that every NVIDIA card uses, but they were only timed on
  the V100. They need checking on newer cards before being offered to llama.cpp.
- **A faster private version exists.** A community version of llama.cpp built specifically for the V100
  (llama-superfast-v100) is still faster for small-batch Qwen. Its extra speed comes from rearranging the model's
  weights in memory when the model loads, plus model-specific shortcuts. Copying that would mean rebuilding most of
  that version, and llama.cpp is unlikely to accept it. The recommendation is to stop here and offer the cleaner
  changes upstream.
- **Two cards working together** (the owner's machine will have two V100s): the software groundwork exists, but it
  can only be tested once the real machine arrives.

## What happens next

1. **Upstreaming:** the owner decides which changes to offer to llama.cpp and writes those submissions. A per-patch
   fact sheet supports that: [PATCH_DATASHEETS.md](PATCH_DATASHEETS.md).
2. **The home machine (about November 2026):** two-card experiments, and the smaller leftover items: 2–4 token
   batches, two more compression formats, and combining matrix steps that share an input.
3. **Sibling projects** apply the same method to two other cards: the NVIDIA P100 (`~/Github/P100`) and the Intel
   Arc A770 (`~/Github/A770`).

## Small glossary

- **GPU / graphics card:** the chip that runs the model's maths in parallel.
- **V100 / Volta:** the NVIDIA card (2017) and its chip family; "sm_70" is its technical code name.
- **Tensor cores:** circuits in the GPU built for the kind of matrix maths AI models use.
- **Token:** a piece of text, roughly a word.
- **Batch:** how many tokens are processed in one step.
- **Attention:** the part of the model that looks back over the conversation.
- **Matrix multiplication (matmul):** applying the model's weights; most of the work.
- **Quantisation:** storing numbers in fewer bits to save memory.
- **KV cache:** the model's stored memory of the conversation so far.
- **Kernel:** one small program that runs on the GPU.
- **Patch:** a file describing a change to llama.cpp's code.
- **Upstream:** the official llama.cpp project.
- **FP16 / FP32:** 16-bit and 32-bit number formats; FP16 is compact but has a smaller range.

## Where the details are

- Plan and history: [ROADMAP.md](ROADMAP.md)
- Measurements:
  - [first cloud session](research/phase0-cloud-2026-10-10.md)
  - [changes 1–6](research/patches-cloud-2026-10-10.md)
  - [changes 2 and 7](research/pad-rule-vecpack-v100-2026-10-10.md)
  - [where the faster version's speed comes from](research/qwen-smallbatch-and-vecpack-heads-v100-2026-10-11.md)
- Design of change 8 and all its measurements: [design/P9-small-batch-qmatmul.md](design/P9-small-batch-qmatmul.md)
- Raw data: `bench/results/`
