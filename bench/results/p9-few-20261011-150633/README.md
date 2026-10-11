# p9-few: nsys of few-row and split-K launches before the reduction fix (Phase 2, third part)

Kernel and memory-time summaries per shape (Q4_K/Q8_0, N = 48 / 4096 / 17408, 8 and 16 tokens, two K-split settings
`pd1`/`pd2`) for `p9-dev6.diff`. Showed the split-K last-block reduction as a fixed cost per split launch; see
"Phase 2, third part" in `docs/design/P9-small-batch-qmatmul.md`. Raw `.nsys-rep`/`.sqlite` files were not copied.
