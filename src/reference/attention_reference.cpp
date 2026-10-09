#include "volta_fa/attention.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace volta_fa {

float AttnParams::effective_scale() const {
    return scale == 0.0f ? 1.0f / std::sqrt(static_cast<float>(d_qk)) : scale;
}

bool validate(const AttnParams& p, std::string* why) {
    const char* reason = nullptr;
    if (p.batch <= 0 || p.n_head_q <= 0 || p.n_head_kv <= 0 ||
        p.seq_q <= 0 || p.seq_kv <= 0 || p.d_qk <= 0 || p.d_v <= 0) {
        reason = "all dimensions must be positive";
    } else if (p.n_head_q % p.n_head_kv != 0) {
        reason = "n_head_q must be divisible by n_head_kv";
    } else if (p.softcap < 0.0f || !std::isfinite(p.softcap)) {
        reason = "softcap must be finite and non-negative";
    } else if (!std::isfinite(p.scale)) {
        reason = "scale must be finite";
    }
    if (why) *why = reason ? reason : "";
    return reason == nullptr;
}

float round_to_fp16(float x) {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "requires binary32 float");
    static_assert(std::numeric_limits<float>::is_iec559, "requires IEEE floating point");
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    const std::uint32_t sign = bits & 0x80000000u;
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    const std::uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 0xffu) return x;  // Preserve infinity and NaN (including payload).

    const int e = static_cast<int>(exponent) - 127;
    std::uint32_t half = 0;
    if (e >= -25) {
        if (e > 15) {
            half = 0x7c00u;
        } else {
            const std::uint32_t significand = fraction | 0x800000u;
            const int shift = e < -14 ? -e - 1 : 13;
            std::uint32_t rounded = significand >> shift;
            const std::uint32_t remainder = significand & ((1u << shift) - 1u);
            const std::uint32_t midpoint = 1u << (shift - 1);
            if (remainder > midpoint || (remainder == midpoint && (rounded & 1u))) ++rounded;
            if (e < -14) {
                half = rounded;  // A carry to 0x400 becomes the smallest normal.
            } else {
                // The carry from the 11-bit significand increments the exponent,
                // including the 65520 tie which rounds to infinity.
                half = (static_cast<std::uint32_t>(e + 15) << 10) + rounded - 0x400u;
            }
        }
    }

    const std::uint32_t he = half >> 10;
    std::uint32_t hf = half & 0x3ffu;
    std::uint32_t result = sign;
    if (he == 31) {
        result |= 0x7f800000u;
    } else if (he != 0) {
        result |= ((he + 112u) << 23) | (hf << 13);
    } else if (hf != 0) {
        int normal_exponent = -14;
        while ((hf & 0x400u) == 0) {
            hf <<= 1;
            --normal_exponent;
        }
        result |= (static_cast<std::uint32_t>(normal_exponent + 127) << 23) |
                  ((hf & 0x3ffu) << 13);
    }
    float output;
    std::memcpy(&output, &result, sizeof(output));
    return output;
}

void attention_reference(const AttnParams& p, const float* q, const float* k, const float* v,
                         const float* mask, float* o, float* lse) {
    std::string why;
    if (!validate(p, &why)) throw std::invalid_argument(why);
    const double scale = p.effective_scale();
    const double neg_inf = -std::numeric_limits<double>::infinity();
    std::vector<double> logits(p.seq_kv);
    std::vector<double> acc(p.d_v);
    for (int b = 0; b < p.batch; ++b) {
        for (int h = 0; h < p.n_head_q; ++h) {
            const int kh = h / (p.n_head_q / p.n_head_kv);
            const std::size_t kv_base = (static_cast<std::size_t>(b) * p.n_head_kv + kh) * p.seq_kv;
            for (int i = 0; i < p.seq_q; ++i) {
                const std::size_t row = (static_cast<std::size_t>(b) * p.n_head_q + h) * p.seq_q + i;
                double m = neg_inf;
                for (int j = 0; j < p.seq_kv; ++j) {
                    double x = neg_inf;
                    if (!p.causal || j <= i + (p.seq_kv - p.seq_q)) {
                        double dot = 0.0;
                        for (int d = 0; d < p.d_qk; ++d) {
                            dot += static_cast<double>(q[row * p.d_qk + d]) *
                                   k[(kv_base + j) * p.d_qk + d];
                        }
                        x = dot * scale;
                        if (p.softcap > 0.0f) x = p.softcap * std::tanh(x / p.softcap);
                        if (mask) x += mask[static_cast<std::size_t>(i) * p.seq_kv + j];
                    }
                    logits[j] = x;
                    m = std::max(m, x);
                }
                double sum = 0.0;
                std::fill(acc.begin(), acc.end(), 0.0);
                if (m != neg_inf) {
                    for (int j = 0; j < p.seq_kv; ++j) {
                        const double weight = std::exp(logits[j] - m);
                        sum += weight;
                        for (int d = 0; d < p.d_v; ++d) {
                            acc[d] += weight * v[(kv_base + j) * p.d_v + d];
                        }
                    }
                }
                for (int d = 0; d < p.d_v; ++d) {
                    o[row * p.d_v + d] = sum == 0.0 ? 0.0f : static_cast<float>(acc[d] / sum);
                }
                if (lse) lse[row] = sum == 0.0 ? static_cast<float>(neg_inf) : static_cast<float>(m + std::log(sum));
            }
        }
    }
}

}  // namespace volta_fa
