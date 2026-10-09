#include "volta_fa/attention.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace volta_fa {

void attention_tiled_model(const AttnParams& p, const TiledModelConfig& cfg, const float* q, const float* k,
                           const float* v, const float* mask, float* o, float* lse) {
    std::string why;
    if (!validate(p, &why)) throw std::invalid_argument(why);
    if (cfg.br <= 0 || cfg.bc <= 0) throw std::invalid_argument("br and bc must be positive");
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float scale = p.effective_scale();
    const int block_rows = std::min(cfg.br, p.seq_q);
    std::vector<float> m(block_rows), l(block_rows);
    std::vector<float> acc(static_cast<std::size_t>(block_rows) * p.d_v);
    std::vector<float> tile_acc(p.d_v);
    std::vector<float> logits(std::min(cfg.bc, p.seq_kv));
    for (int b = 0; b < p.batch; ++b) {
        for (int h = 0; h < p.n_head_q; ++h) {
            const int kh = h / (p.n_head_q / p.n_head_kv);
            const std::size_t kv_base = (static_cast<std::size_t>(b) * p.n_head_kv + kh) * p.seq_kv;
            const std::size_t q_base = (static_cast<std::size_t>(b) * p.n_head_q + h) * p.seq_q;
            for (int ib = 0; ib < p.seq_q;) {
                const int nr = std::min(cfg.br, p.seq_q - ib);
                std::fill(m.begin(), m.end(), neg_inf);
                std::fill(l.begin(), l.end(), 0.0f);
                std::fill(acc.begin(), acc.end(), 0.0f);
                for (int jb = 0; jb < p.seq_kv;) {
                    const int nc = std::min(cfg.bc, p.seq_kv - jb);
                    // Even the last query row in this block cannot see this tile.
                    if (p.causal && jb > ib + nr - 1 + (p.seq_kv - p.seq_q)) break;
                    for (int r = 0; r < nr; ++r) {
                        const int i = ib + r;
                        if (p.causal && jb > i + (p.seq_kv - p.seq_q)) continue;
                        const std::size_t row = q_base + i;
                        float tile_max = neg_inf;
                        for (int c = 0; c < nc; ++c) {
                            const int j = jb + c;
                            float x = neg_inf;
                            if (!p.causal || j <= i + (p.seq_kv - p.seq_q)) {
                                float dot = 0.0f;
                                for (int d = 0; d < p.d_qk; ++d) {
                                    dot += q[row * p.d_qk + d] * k[(kv_base + j) * p.d_qk + d];
                                    if (cfg.qk_accum_fp16 && ((d + 1) % 16 == 0 || d + 1 == p.d_qk)) {
                                        dot = round_to_fp16(dot);
                                    }
                                }
                                x = dot * scale;
                                if (p.softcap > 0.0f) x = p.softcap * std::tanh(x / p.softcap);
                                if (mask) x += mask[static_cast<std::size_t>(i) * p.seq_kv + j];
                            }
                            logits[c] = x;
                            tile_max = std::max(tile_max, x);
                        }
                        const float m_new = std::max(m[r], tile_max);
                        if (m_new == neg_inf) continue;  // No visible key yet; avoid -inf - -inf.
                        const float alpha = m[r] == neg_inf ? 0.0f : std::exp(m[r] - m_new);
                        float tile_sum = 0.0f;
                        float* row_acc = acc.data() + static_cast<std::size_t>(r) * p.d_v;
                        std::fill(tile_acc.begin(), tile_acc.end(), 0.0f);
                        for (int c = 0; c < nc; ++c) {
                            float weight = std::exp(logits[c] - m_new);
                            if (cfg.p_fp16) weight = round_to_fp16(weight);
                            tile_sum += weight;
                            const float* value = v + (kv_base + jb + c) * p.d_v;
                            for (int d = 0; d < p.d_v; ++d) tile_acc[d] += weight * value[d];
                        }
                        for (int d = 0; d < p.d_v; ++d) row_acc[d] = row_acc[d] * alpha + tile_acc[d];
                        l[r] = l[r] * alpha + tile_sum;
                        m[r] = m_new;
                    }
                    jb += nc;
                }
                for (int r = 0; r < nr; ++r) {
                    const std::size_t row = q_base + ib + r;
                    for (int d = 0; d < p.d_v; ++d) {
                        float value = l[r] == 0.0f ? 0.0f : acc[static_cast<std::size_t>(r) * p.d_v + d] / l[r];
                        if (cfg.o_fp16_out) value = round_to_fp16(value);
                        o[row * p.d_v + d] = value;
                    }
                    if (lse) lse[row] = l[r] == 0.0f ? neg_inf : m[r] + std::log(l[r]);
                }
                ib += nr;
            }
        }
    }
}

}  // namespace volta_fa
