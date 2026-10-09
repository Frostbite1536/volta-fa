#include "check.h"
#include "volta_fa/attention.h"

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

using namespace volta_fa;

namespace {
constexpr float inf = std::numeric_limits<float>::infinity();
void compare(const AttnParams& p, const TiledModelConfig& cfg, std::mt19937& rng) {
    std::uniform_real_distribution<float> random(-0.4f, 0.4f);
    const std::size_t rows = static_cast<std::size_t>(p.batch) * p.n_head_q * p.seq_q;
    const std::size_t kv_rows = static_cast<std::size_t>(p.batch) * p.n_head_kv * p.seq_kv;
    std::vector<float> q(rows * p.d_qk), k(kv_rows * p.d_qk), v(kv_rows * p.d_v);
    for (auto* data : {&q, &k, &v}) for (float& x : *data) x = random(rng);
    std::vector<float> mask(static_cast<std::size_t>(p.seq_q) * p.seq_kv, 0.0f);
    for (int i = 0; i < p.seq_q; ++i) {
        for (int j = 0; j < p.seq_kv; ++j) {
            mask[static_cast<std::size_t>(i) * p.seq_kv + j] = i == 1 || (i + j) % 11 == 0 ? -inf : 0.03f * (j % 7);
        }
    }
    std::vector<float> ref(rows * p.d_v), out(ref.size()), ref_lse(rows), out_lse(rows);
    attention_reference(p, q.data(), k.data(), v.data(), mask.data(), ref.data(), ref_lse.data());
    attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), mask.data(), out.data(), out_lse.data());
    for (std::size_t i = 0; i < ref.size(); ++i) CHECK(near(out[i], ref[i], fp32_abs_tolerance));
    for (std::size_t i = 0; i < rows; ++i) {
        if (ref_lse[i] == -inf) CHECK(out_lse[i] == -inf);
        else CHECK(near(out_lse[i], ref_lse[i], fp32_abs_tolerance));
    }
    if (p.causal) {
        auto explicit_params = p;
        explicit_params.causal = false;
        for (int i = 0; i < p.seq_q; ++i) {
            for (int j = 0; j < p.seq_kv; ++j) {
                if (j > i + (p.seq_kv - p.seq_q)) mask[static_cast<std::size_t>(i) * p.seq_kv + j] = -inf;
            }
        }
        std::vector<float> explicit_out(out.size()), explicit_lse(rows);
        attention_tiled_model(explicit_params, cfg, q.data(), k.data(), v.data(), mask.data(),
                              explicit_out.data(), explicit_lse.data());
        for (std::size_t i = 0; i < out.size(); ++i) CHECK(float_bits(out[i]) == float_bits(explicit_out[i]));
        for (std::size_t i = 0; i < rows; ++i) CHECK(float_bits(out_lse[i]) == float_bits(explicit_lse[i]));
    }
}

void matrix() {
    std::mt19937 rng(20261009);
    for (int dim : {64, 80, 128, 256}) {
        for (int nk : {1, 63, 65, 1000}) {
            for (int br : {16, 32, 64, 128}) {
                for (int bc : {16, 32, 64, 128}) {
                    for (bool causal : {false, true}) {
                        AttnParams p;
                        p.seq_q = nk == 63 ? br + 1 : 7;
                        p.seq_kv = nk;
                        p.d_qk = dim;
                        p.d_v = dim == 80 ? 13 : dim;
                        p.n_head_q = 4;
                        p.n_head_kv = 2;
                        p.causal = causal;
                        p.softcap = dim == 80 ? 0.7f : 0.0f;
                        if (dim == 64 && nk == 65 && br == 16 && bc == 16) {
                            p.batch = 2;
                            p.n_head_q = 8;
                        }
                        compare(p, {br, bc, false, false, false}, rng);
                    }
                }
            }
        }
    }
}

void masked() {
    AttnParams p;
    p.seq_q = 2;
    p.seq_kv = 65;
    p.d_qk = 1;
    p.d_v = 2;
    std::vector<float> q(2, 1), k(65, 1), v(130, 3), mask(130, -inf);
    // First row stays fully masked; second has its first visible key in tile 2.
    mask[65 + 64] = 0;
    for (bool rounded : {false, true}) {
        float o[4], lse[2];
        const TiledModelConfig cfg{16, 64, rounded, rounded, rounded};
        attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), mask.data(), o, lse);
        CHECK(o[0] == 0 && o[1] == 0 && lse[0] == -inf);
        CHECK(o[2] == 3 && o[3] == 3 && lse[1] == 1);
        for (float x : o) CHECK(std::isfinite(x));
        for (float x : lse) CHECK(!std::isnan(x));
        float without_lse[4];
        attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), mask.data(), without_lse, nullptr);
        for (int i = 0; i < 4; ++i) CHECK(o[i] == without_lse[i]);
        // The causal skip is also bit-identical with all flags enabled.
        auto causal_params = p;
        causal_params.causal = true;
        float causal_o[4], causal_lse[2];
        attention_tiled_model(causal_params, cfg, q.data(), k.data(), v.data(), mask.data(), causal_o, causal_lse);
        // The mask already excludes row 0's causal-hidden key 64.
        for (int i = 0; i < 4; ++i) CHECK(float_bits(o[i]) == float_bits(causal_o[i]));
        for (int i = 0; i < 2; ++i) CHECK(float_bits(lse[i]) == float_bits(causal_lse[i]));
    }
}

void rounding() {
    AttnParams p;
    p.seq_kv = 2;
    p.d_qk = 33;
    p.d_v = 1;
    p.scale = 1;
    std::vector<float> q(33, 1), k(66, 0);
    k[0] = 1;
    k[16] = std::ldexp(1.0f, -11);
    k[32] = std::ldexp(1.0f, -11);
    const float v[] = {0, 1};
    float out[1], lse[1], plain[1];
    attention_tiled_model(p, {16, 16, true, false, false}, q.data(), k.data(), v, nullptr, out, lse);
    CHECK(near(out[0], 1.0 / (1.0 + std::exp(1.0)), analytic_tight_abs_tolerance));
    CHECK(near(lse[0], 1.0 + std::log1p(std::exp(-1.0)), analytic_tight_abs_tolerance));
    attention_tiled_model(p, {16, 16, false, false, false}, q.data(), k.data(), v, nullptr, plain, nullptr);
    CHECK(std::abs(out[0] - plain[0]) > 1e-4f);

    // Two half-ULPs in the same chunk must add before the FP16 rounding.
    k[17] = k[16];
    k[32] = 0;
    attention_tiled_model(p, {16, 16, true, false, false}, q.data(), k.data(), v, nullptr, out, nullptr);
    CHECK(near(out[0], 1.0 / (1.0 + std::exp(1.0 + std::ldexp(1.0, -10))), analytic_tight_abs_tolerance));

    p.d_qk = 1;
    const float q1[] = {1}, k1[] = {0, std::log(3.0f)}, v1[] = {1, 0};
    const float rounded_p = round_to_fp16(std::exp(-k1[1]));
    const float expected = rounded_p / (1.0f + rounded_p);
    attention_tiled_model(p, {16, 16, false, true, false}, q1, k1, v1, nullptr, out, lse);
    CHECK(out[0] == expected);
    CHECK(lse[0] == k1[1] + std::log(1.0f + rounded_p));
    attention_tiled_model(p, {16, 16, false, false, false}, q1, k1, v1, nullptr, plain, nullptr);
    CHECK(std::abs(out[0] - plain[0]) > 1e-5f);
    attention_tiled_model(p, {16, 16, false, true, true}, q1, k1, v1, nullptr, out, nullptr);
    CHECK(out[0] == round_to_fp16(expected));
    CHECK(out[0] != expected);

    // Once an earlier tile set m=4, the next tile's P is exp(-4),
    // rounded relative to the running max rather than that tile's own max.
    const float k2[] = {4, 0};
    const float weight = round_to_fp16(std::exp(-4.0f));
    attention_tiled_model(p, {16, 1, false, true, false}, q1, k2, v, nullptr, out, lse);
    CHECK(out[0] == weight / (1.0f + weight));
    CHECK(lse[0] == 4.0f + std::log(1.0f + weight));
}

void validation() {
    AttnParams p;
    for (TiledModelConfig cfg : {TiledModelConfig{0, 64, false, false, false},
                                 TiledModelConfig{64, -1, false, false, false}}) {
        bool threw = false;
        try {
            attention_tiled_model(p, cfg, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
    p.d_v = 0;
    std::string reason;
    CHECK(!validate(p, &reason));
    bool threw = false;
    try {
        attention_tiled_model(p, {}, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    } catch (const std::invalid_argument& error) {
        threw = true;
        CHECK(reason == error.what());
    }
    CHECK(threw);
}
}  // namespace

int main(int argc, char** argv) {
    return run_tests(argc, argv, {{"matrix", matrix}, {"masked", masked}, {"rounding", rounding}, {"validation", validation}});
}
