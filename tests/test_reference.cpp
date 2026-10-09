#include "check.h"
#include "volta_fa/attention.h"

#include <limits>
#include <vector>

using namespace volta_fa;

namespace {
constexpr float inf = std::numeric_limits<float>::infinity();

AttnParams small(int nq, int nk, int dq, int dv) {
    AttnParams p;
    p.seq_q = nq;
    p.seq_kv = nk;
    p.d_qk = dq;
    p.d_v = dv;
    p.scale = 1.0f;
    return p;
}

void tiny() {
    const auto p = small(2, 2, 2, 2);
    const float q[] = {1, 0, 0, 1};
    const float k[] = {std::log(3.0f), 0, 0, std::log(3.0f)};
    const float v[] = {2, 4, 6, 8};
    float o[4], lse[2];
    attention_reference(p, q, k, v, nullptr, o, lse);
    const float expected[] = {3, 5, 5, 7};  // Weights (3/4,1/4) then (1/4,3/4).
    for (int i = 0; i < 4; ++i) CHECK(near(o[i], expected[i], analytic_abs_tolerance));
    for (float x : lse) CHECK(near(x, std::log(4.0), analytic_abs_tolerance));
    float without_lse[4];
    attention_reference(p, q, k, v, nullptr, without_lse, nullptr);
    for (int i = 0; i < 4; ++i) CHECK(o[i] == without_lse[i]);
}

void uniform() {
    auto p = small(2, 3, 2, 3);  // Different QK and V dimensions.
    p.scale = 0.0f;
    CHECK(near(p.effective_scale(), 1.0 / std::sqrt(2.0), analytic_tight_abs_tolerance));
    const float q[] = {0, 0, 0, 0};
    const float k[] = {1, 2, 3, 4, 5, 6};
    const float v[] = {1, 2, 6, 3, 4, 0, 8, 9, -3};
    float o[6], lse[2];
    attention_reference(p, q, k, v, nullptr, o, lse);
    for (int r = 0; r < 2; ++r) {
        CHECK(o[r * 3] == 4.0f);
        CHECK(o[r * 3 + 1] == 5.0f);
        CHECK(o[r * 3 + 2] == 1.0f);
        CHECK(near(lse[r], std::log(3.0), analytic_tight_abs_tolerance));
    }
}

void gqa() {
    auto p = small(2, 3, 1, 2);
    p.batch = 2;
    p.n_head_q = 8;
    p.n_head_kv = 2;
    std::vector<float> q(2 * 8 * 2, 0.0f), k(2 * 2 * 3, 0.0f), v(2 * 2 * 3 * 2);
    for (int b = 0; b < 2; ++b) {
        for (int h = 0; h < 2; ++h) {
            for (int j = 0; j < 3; ++j) {
                v[((b * 2 + h) * 3 + j) * 2] = 100.0f * b + 10.0f * h + j;
                v[((b * 2 + h) * 3 + j) * 2 + 1] = -100.0f * b - 10.0f * h - j;
            }
        }
    }
    std::vector<float> o(2 * 8 * 2 * 2);
    attention_reference(p, q.data(), k.data(), v.data(), nullptr, o.data(), nullptr);
    for (int b = 0; b < 2; ++b) {
        for (int h = 0; h < 8; ++h) {
            for (int i = 0; i < 2; ++i) {
                const float expected = 100.0f * b + 10.0f * (h / 4) + 1;
                CHECK(o[((b * 8 + h) * 2 + i) * 2] == expected);
                CHECK(o[((b * 8 + h) * 2 + i) * 2 + 1] == -expected);
            }
        }
    }
}

void double_accumulation() {
    const auto p = small(1, 2, 3, 1);
    const float q[] = {1, 1, 1}, k[] = {1e8f, 1, -1e8f, 0, 0, 0}, v[] = {0, 1};
    float o[1];
    attention_reference(p, q, k, v, nullptr, o, nullptr);
    CHECK(near(o[0], 1.0 / (1.0 + std::exp(1.0)), analytic_tight_abs_tolerance));
    const auto uniform_p = small(1, 3, 1, 1);
    const float zero_q[] = {0}, zero_k[] = {0, 0, 0}, cancelling_v[] = {1e8f, 1, -1e8f};
    attention_reference(uniform_p, zero_q, zero_k, cancelling_v, nullptr, o, nullptr);
    CHECK(near(o[0], 1.0 / 3.0, analytic_tight_abs_tolerance));
}

void causal() {
    auto p = small(3, 5, 1, 1);
    p.causal = true;
    const float q[] = {0, 0, 0}, k[] = {0, 0, 0, 0, 0}, v[] = {0, 2, 4, 6, 8};
    float o[3], lse[3];
    attention_reference(p, q, k, v, nullptr, o, lse);
    for (int i = 0; i < 3; ++i) {
        CHECK(o[i] == 2.0f + i);
        CHECK(near(lse[i], std::log(3.0 + i), analytic_tight_abs_tolerance));
    }
    // seq_q > seq_kv is valid and begins with rows having no visible keys.
    p.seq_q = 5;
    p.seq_kv = 3;
    float more_o[5], more_lse[5];
    attention_reference(p, k, q, v, nullptr, more_o, more_lse);
    for (int i = 0; i < 2; ++i) {
        CHECK(more_o[i] == 0);
        CHECK(more_lse[i] == -inf);
    }
    CHECK(more_o[2] == 0 && more_o[3] == 1 && more_o[4] == 2);
}

void masked() {
    const auto p = small(2, 3, 1, 2);
    const float q[] = {1, 1}, k[] = {1, 2, 3}, v[] = {1, 2, 3, 4, 5, 6};
    const float mask[] = {-inf, -inf, -inf, -inf, 0, -inf};
    float o[4], lse[2];
    attention_reference(p, q, k, v, mask, o, lse);
    CHECK(o[0] == 0 && o[1] == 0 && lse[0] == -inf);
    CHECK(o[2] == 3 && o[3] == 4 && lse[1] == 2);
    for (float x : o) CHECK(std::isfinite(x));
    for (float x : lse) CHECK(!std::isnan(x));
}

void softcap() {
    auto p = small(1, 2, 1, 1);
    p.scale = 2.0f;
    p.softcap = 3.0f;
    const float q[] = {10}, k[] = {10, -10}, v[] = {2, 9}, mask[] = {-4, 1};
    float o[1], lse[1];
    attention_reference(p, q, k, v, mask, o, lse);
    const double a = 3 * std::tanh(200.0 / 3) - 4;
    const double b = 3 * std::tanh(-200.0 / 3) + 1;
    const double sum = std::exp(a) + std::exp(b);
    CHECK(near(o[0], (2 * std::exp(a) + 9 * std::exp(b)) / sum, analytic_abs_tolerance));
    CHECK(near(lse[0], std::log(sum), analytic_tight_abs_tolerance));
}

void large() {
    auto p = small(1, 3, 1, 2);
    p.scale = 10000;
    const float q[] = {1}, k[] = {-1, 1, 0}, v[] = {1, 2, 7, -3, 9, 10};
    float o[2], lse[1];
    attention_reference(p, q, k, v, nullptr, o, lse);
    CHECK(o[0] == 7 && o[1] == -3 && lse[0] == 10000);
    for (float x : o) CHECK(std::isfinite(x));
}

void rejected(const AttnParams& p) {
    std::string reason;
    CHECK(!validate(p, &reason));
    CHECK(!reason.empty());
    CHECK(!validate(p));
    bool threw = false;
    try {
        attention_reference(p, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    } catch (const std::invalid_argument& error) {
        threw = true;
        CHECK(reason == error.what());
    }
    CHECK(threw);
}

void validation() {
    int AttnParams::*dimensions[] = {&AttnParams::batch, &AttnParams::n_head_q, &AttnParams::n_head_kv,
        &AttnParams::seq_q, &AttnParams::seq_kv, &AttnParams::d_qk, &AttnParams::d_v};
    for (auto dim : dimensions) {
        for (int bad : {0, -1}) {
            AttnParams p;
            p.*dim = bad;
            rejected(p);
        }
    }
    AttnParams p;
    p.n_head_q = 3;
    p.n_head_kv = 2;
    rejected(p);
    p = AttnParams{};
    p.softcap = -1;
    rejected(p);
    for (float bad : {inf, -inf, std::numeric_limits<float>::quiet_NaN()}) {
        p = AttnParams{};
        p.scale = bad;
        rejected(p);
        p = AttnParams{};
        p.softcap = bad;
        rejected(p);
    }
    p = AttnParams{};
    p.seq_q = 5;
    p.seq_kv = 3;
    p.causal = true;
    std::string reason = "old reason";
    CHECK(validate(p, &reason));
    CHECK(reason.empty());
    p.scale = -2;
    CHECK(validate(p));
    CHECK(p.effective_scale() == -2);
}
}  // namespace

int main(int argc, char** argv) {
    return run_tests(argc, argv, {{"tiny", tiny}, {"uniform", uniform}, {"double", double_accumulation}, {"gqa", gqa}, {"causal", causal},
        {"masked", masked}, {"softcap", softcap}, {"large", large}, {"validation", validation}});
}
