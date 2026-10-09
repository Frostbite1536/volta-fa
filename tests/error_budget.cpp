#include "volta_fa/attention.h"
#include "volta_fa/tolerance.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace volta_fa;

int main() {
    try {
        const char* distributions[] = {"normal", "outlier", "peaked"};
        const char* names[] = {"P", "PO", "QPO"};
        const TiledModelConfig configs[] = {{64, 64, false, true, false},
            {64, 64, false, true, true}, {64, 64, true, true, true}};
        std::size_t total_nonfinite = 0;
        std::size_t po_out_of_tol = 0;  // config "PO" models a correct kernel; it must meet tolerance.h
        std::cout << "seq_q=128 batch=heads=1 br=bc=64; inputs rounded to FP16\n"
                  << "seed=12345+distribution_index*100000+d*100+seq_kv (mt19937)\n"
                  << "normal: scale=1/sqrt(d); outlier: Q,K channels 0..3 x20; "
                     "peaked: scale=64/sqrt(d); V~N(0,1) throughout\n"
                  << "max_rel=max_abs/max|O_ref|; nonfinite counts both reference and model\n"
                  << "cfg dist d kv causal max_abs max_rel mean_abs lse_abs nonfinite\n";
        std::cout << std::scientific << std::setprecision(6);
        for (int distribution = 0; distribution < 3; ++distribution) {
            for (int dim : {64, 128, 256}) {
                for (int nk : {512, 4096}) {
                    AttnParams p;
                    p.seq_q = 128;
                    p.seq_kv = nk;
                    p.d_qk = p.d_v = dim;
                    if (distribution == 2) p.scale = 64.0f / std::sqrt(static_cast<float>(dim));
                    const unsigned seed = 12345u + static_cast<unsigned>(distribution) * 100000u +
                                          static_cast<unsigned>(dim) * 100u + static_cast<unsigned>(nk);
                    std::mt19937 rng(seed);
                    std::normal_distribution<float> normal(0.0f, 1.0f);
                    std::vector<float> q(static_cast<std::size_t>(p.seq_q) * dim);
                    std::vector<float> k(static_cast<std::size_t>(nk) * dim), v(k.size());
                    for (auto* data : {&q, &k}) {
                        for (std::size_t i = 0; i < data->size(); ++i) {
                            float x = normal(rng);
                            if (distribution == 1 && i % dim < 4) x *= 20.0f;
                            (*data)[i] = round_to_fp16(x);
                        }
                    }
                    for (float& x : v) x = round_to_fp16(normal(rng));
                    std::vector<float> ref(q.size()), out(q.size()), ref_lse(p.seq_q), out_lse(p.seq_q);
                    for (bool causal : {false, true}) {
                        p.causal = causal;
                        attention_reference(p, q.data(), k.data(), v.data(), nullptr, ref.data(), ref_lse.data());
                        for (int config = 0; config < 3; ++config) {
                            attention_tiled_model(p, configs[config], q.data(), k.data(), v.data(), nullptr,
                                                  out.data(), out_lse.data());
                            double max_abs = 0, max_ref = 0, sum_abs = 0, lse_abs = 0;
                            std::size_t nonfinite = 0;
                            for (std::size_t i = 0; i < ref.size(); ++i) {
                                nonfinite += !std::isfinite(ref[i]);
                                nonfinite += !std::isfinite(out[i]);
                                const double error = std::abs(static_cast<double>(out[i]) - ref[i]);
                                max_abs = std::max(max_abs, error);
                                max_ref = std::max(max_ref, std::abs(static_cast<double>(ref[i])));
                                sum_abs += error;
                                if (config == 1 && !out_within_tolerance(out[i], ref[i])) ++po_out_of_tol;
                            }
                            for (int i = 0; i < p.seq_q; ++i) {
                                // Every row here has visible keys, including under causal.
                                nonfinite += !std::isfinite(ref_lse[i]);
                                nonfinite += !std::isfinite(out_lse[i]);
                                lse_abs = std::max(lse_abs, std::abs(static_cast<double>(out_lse[i]) - ref_lse[i]));
                                if (config == 1 && !lse_within_tolerance(out_lse[i], ref_lse[i])) ++po_out_of_tol;
                            }
                            total_nonfinite += nonfinite;
                            const double max_rel = max_ref == 0 ? 0 : max_abs / max_ref;
                            std::cout << names[config] << ' ' << distributions[distribution] << ' ' << dim << ' '
                                      << nk << ' ' << static_cast<int>(causal) << ' ' << max_abs << ' ' << max_rel << ' '
                                      << sum_abs / ref.size() << ' ' << lse_abs << ' ' << nonfinite << '\n';
                        }
                    }
                }
            }
        }
        std::cout << "total_nonfinite=" << total_nonfinite << " po_out_of_tolerance=" << po_out_of_tol << '\n';
        return total_nonfinite == 0 && po_out_of_tol == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
