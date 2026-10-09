// Parity tolerances for GPU attention kernels vs attention_reference (INV-NUM-003).
// Changing these is a manager decision; record it in docs/INVARIANTS.md.
//
// Derivation (2026-10-09, tests/error_budget.cpp): the tiled model with FP32 QK accumulate,
// FP16 P and FP16 output (the "PO" rows) had a worst element error of ~1.0e-3 and worst LSE
// error of ~5.2e-4 across normal, outlier-channel and near-one-hot inputs, d = 64/128/256,
// seq_kv = 512/4096, causal on/off. The error is dominated by rounding O to FP16 (half an
// ulp is |O| * 2^-11). The bounds below leave ~2x headroom for GPU summation order and
// exp2 approximations. error_budget asserts the model stays inside them, so a tolerance
// that a correct kernel cannot meet fails CI instead of being discovered on hardware.
//
// FP16 QK accumulation ("QPO") reached 0.46 absolute error with outlier channels and is
// not allowed (INV-NUM-005); there is deliberately no tolerance for it.
#pragma once

#include <cmath>

namespace volta_fa {

constexpr double kOutAbsTol = 1e-3;
constexpr double kOutRelTol = 2e-3;
constexpr double kLseAbsTol = 1e-3;

inline bool out_within_tolerance(double got, double ref) {
    return std::abs(got - ref) <= kOutAbsTol + kOutRelTol * std::abs(ref);
}

inline bool lse_within_tolerance(double got, double ref) {
    if (std::isinf(ref) && ref < 0) return std::isinf(got) && got < 0;
    return std::abs(got - ref) <= kLseAbsTol;
}

}  // namespace volta_fa
