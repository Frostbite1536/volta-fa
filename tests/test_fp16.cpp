#include "check.h"
#include "volta_fa/attention.h"

#include <limits>

using volta_fa::round_to_fp16;

namespace {
void known_patterns() {
    CHECK(float_bits(round_to_fp16(1.0f)) == 0x3f800000u);
    CHECK(float_bits(round_to_fp16(65504.0f)) == 0x477fe000u);
    CHECK(float_bits(round_to_fp16(65520.0f)) == 0x7f800000u);
    CHECK(float_bits(round_to_fp16(-65520.0f)) == 0xff800000u);
    CHECK(float_bits(round_to_fp16(65519.0f)) == 0x477fe000u);
    CHECK(float_bits(round_to_fp16(6.103515625e-5f)) == 0x38800000u);
    // 6.1e-5 is just below the midpoint to the smallest normal.
    CHECK(float_bits(round_to_fp16(6.1e-5f)) == 0x387fc000u);
    CHECK(float_bits(round_to_fp16(5.96e-8f)) == 0x33800000u);
    CHECK(float_bits(round_to_fp16(1.0f + std::ldexp(1.0f, -11))) == 0x3f800000u);
    CHECK(float_bits(round_to_fp16(1.0f + 3.0f * std::ldexp(1.0f, -11))) == 0x3f804000u);
    CHECK(float_bits(round_to_fp16(std::ldexp(1.0f, -25))) == 0u);
    CHECK(float_bits(round_to_fp16(-std::ldexp(1.0f, -25))) == 0x80000000u);
    CHECK(float_bits(round_to_fp16(-0.0f)) == 0x80000000u);
    CHECK(float_bits(round_to_fp16(std::numeric_limits<float>::denorm_min())) == 0u);
    CHECK(float_bits(round_to_fp16(std::numeric_limits<float>::max())) == 0x7f800000u);
    CHECK(float_bits(round_to_fp16(-std::numeric_limits<float>::infinity())) == 0xff800000u);
    const std::uint32_t nan_bits = 0xffc12345u;
    float nan;
    std::memcpy(&nan, &nan_bits, sizeof(nan));
    CHECK(std::isnan(round_to_fp16(nan)));
    CHECK(float_bits(round_to_fp16(nan)) == nan_bits);
}

// Construct the exact half value arithmetically, independently of the converter.
float half_value(unsigned bits) {
    const unsigned exponent = bits >> 10;
    const unsigned fraction = bits & 1023u;
    return exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24) :
        std::ldexp(static_cast<float>(1024u + fraction), static_cast<int>(exponent) - 25);
}

void exhaustive_finite_and_ties() {
    for (unsigned bits = 0; bits <= 0x7bffu; ++bits) {
        const float value = half_value(bits);
        CHECK(float_bits(round_to_fp16(value)) == float_bits(value));
        CHECK(float_bits(round_to_fp16(-value)) == float_bits(-value));
        if (bits < 0x7bffu) {
            const float next = half_value(bits + 1);
            const float midpoint = (value + next) * 0.5f;
            const float even = (bits & 1u) ? next : value;
            CHECK(float_bits(round_to_fp16(midpoint)) == float_bits(even));
            CHECK(float_bits(round_to_fp16(-midpoint)) == float_bits(-even));
            CHECK(float_bits(round_to_fp16(std::nextafter(midpoint, value))) == float_bits(value));
            CHECK(float_bits(round_to_fp16(std::nextafter(midpoint, next))) == float_bits(next));
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    return run_tests(argc, argv, {{"patterns", known_patterns}, {"exhaustive", exhaustive_finite_and_ties}});
}
