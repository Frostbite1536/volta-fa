# p9-scale (first activation-scaling run, superseded)

First build of the per-256-slice scaling (`p9-scale.diff`, before the accumulate/scale-load optimisation) against the
unscaled kernel. The perf SQL files have no P-9 shapes (the perf test line was not inserted); the llama-bench runs
showed the first scaled build 1.5-2.6% slower than unscaled on Qwen, which led to the optimised per-slice version
and the per-token alternative compared in `p9-scale2-*`. `*.new-vs-old.txt`: KL divergence of the scaled against the
unscaled build (1.3e-5 / 1.8e-5), not against the FP32 reference; `p9-scale2-*` has the reference comparison.
