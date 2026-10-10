// dequant_nc.cu: launch geometry of llama.cpp's non-contiguous dequantise-to-F16 kernel (P-8, Change 2).
//
// TILE and MMA FlashAttention convert a quantised K/V cache view to F16 before every call
// (fattn-common.cuh launch_fattn -> ggml_get_to_fp16_nc_cuda -> convert.cu dequantize_block_cuda).
// That launcher uses 256-thread blocks along the row (2 values per thread) and one block per row,
// so a D-element row keeps only D/2 of 256 threads busy (D=64: 32/256) and launches one block per
// (token, head). This replicates the kernel body byte for byte and times the upstream geometry
// against patch 0005 (same body, several rows per block) and patch 0006 (q4_0/q8_0 kernel, 8 values per
// thread), plus a device-to-device copy of the F16 output size as a bandwidth reference.
//
//   nvcc -O3 -std=c++17 -arch=sm_70 bench/micro/dequant_nc.cu -o build/dequant_nc && build/dequant_nc
//
// Layout = llama.cpp's FA K/V view of the KV cache: ne = [D, n_kv, n_head_kv],
// row stride (token) = n_head_kv*D values, head stride = D values. Output is contiguous F16.
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

constexpr int QK8_0 = 32, QR8_0 = 1;
constexpr int QK4_0 = 32, QR4_0 = 2;
struct block_q8_0 { half d; int8_t qs[QK8_0]; };
struct block_q4_0 { half d; uint8_t qs[QK4_0 / 2]; };
static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q4_0) == 18, "ggml block sizes");

// dequantize.cuh at llama.cpp 79e2e74
__device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q8_0 * x = (const block_q8_0 *) vx;
    const float d = x[ib].d;
    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];
    v.x *= d;
    v.y *= d;
}
__device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q4_0 * x = (const block_q4_0 *) vx;
    const float d = x[ib].d;
    const int vui = x[ib].qs[iqs];
    v.x = vui & 0xF;
    v.y = vui >> 4;
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}
typedef void (*dequantize_kernel_t)(const void *, const int64_t, const int, float2 &);

// convert.cu dequantize_block at 79e2e74, except that the row index includes threadIdx.y
// (identical to upstream when blockDim.y == 1). fast_div_modulo replaced by / and % (same result).
template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
__global__ void dequantize_block(const void * __restrict__ vx, half * __restrict__ y,
        const int64_t ne00, const int64_t ne01, const int64_t ne0203, const int64_t ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);
    if (i00 >= ne00) {
        return;
    }
    for (int64_t i01 = int64_t(blockIdx.y)*blockDim.y + threadIdx.y; i01 < ne01; i01 += int64_t(gridDim.y)*blockDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const int64_t i02 = i0203 % ne02;
            const int64_t i03 = i0203 / ne02;
            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;
            const int64_t ib = ibx0 + i00/qk;
            const int64_t iqs = (i00%qk)/qr;
            const int64_t iybs = i00 - i00%qk;
            const int64_t y_offset = qr == 1 ? 1 : qk/2;
            float2 v;
            dequantize_kernel(vx, ib, iqs, v);
            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            y[iy0 + 0]        = __float2half(v.x);
            y[iy0 + y_offset] = __float2half(v.y);
        }
    }
}

// third_party/patches/0006 dequantize_block_q_0_nc (8 values per thread, 2-byte aligned loads, 8/16-byte
// stores), dst_t = half; fast_div_modulo replaced by / and %, ggml_cuda_memcpy_1 by plain short/uint2/uint4 copies.
template <bool q8>
__global__ void dequantize_block_q_0_nc(const void * __restrict__ vx, half * __restrict__ y,
        const int ne00, const int ne01, const int64_t ne0203, const int64_t ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int nth_row = ne00/8;
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne01*nth_row) {
        return;
    }
    const int i01 = i / nth_row;
    const int i00 = 8*(i - i01*nth_row);
    const int iqs = i00 % 32;
    for (int64_t i0203 = blockIdx.y; i0203 < ne0203; i0203 += gridDim.y) {
        const int64_t ib = (i0203 / ne02)*s03 + (i0203 % ne02)*s02 + i01*s01 + i00/32;
        half * yb = y + (i0203*ne01 + i01)*ne00 + (i00 - iqs);
        if constexpr (q8) {
            const block_q8_0 * x = (const block_q8_0 *) vx + ib;
            const float d = x->d;
            int8_t q[8];
            for (int k = 0; k < 4; ++k) ((short *) q)[k] = ((const short *) (x->qs + iqs))[k];
            alignas(16) half v[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] = __float2half(float(q[k])*d);
            *(uint4 *) (yb + iqs) = *(const uint4 *) v;
        } else {
            const block_q4_0 * x = (const block_q4_0 *) vx + ib;
            const float d = x->d;
            uint8_t q[4];
            for (int k = 0; k < 2; ++k) ((short *) q)[k] = ((const short *) (x->qs + iqs/2))[k];
            alignas(16) half lo[4];
            alignas(16) half hi[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                lo[k] = __float2half((float(q[k] & 0xF) - 8.0f)*d);
                hi[k] = __float2half((float(q[k] >>  4) - 8.0f)*d);
            }
            *(uint2 *) (yb +      iqs/2) = *(const uint2 *) lo;
            *(uint2 *) (yb + 16 + iqs/2) = *(const uint2 *) hi;
        }
    }
}

constexpr int BLOCK = 256; // CUDA_DEQUANTIZE_BLOCK_SIZE

template <int qk, int qr, dequantize_kernel_t dk>
void launch(bool rows_per_block, const void * vx, half * y, int64_t ne00, int64_t ne01, int64_t ne02,
            int64_t s01, int64_t s02, int64_t s03) {
    int bx = BLOCK;
    if (rows_per_block) {
        // Patch 0005: just enough x threads for one row (warp multiple), the rest of the block on more rows.
        bx = 32;
        while (bx < BLOCK && 2*bx < ne00) bx *= 2;
    }
    const int by = BLOCK / bx;
    const dim3 grid((ne00 + 2*bx - 1) / (2*bx), (int) std::min<int64_t>((ne01 + by - 1) / by, 65535), (int) std::min<int64_t>(ne02, 65535));
    dequantize_block<qk, qr, dk><<<grid, dim3(bx, by), 0>>>(vx, y, ne00, ne01, ne02, ne02, s01, s02, s03);
}

struct Shape { const char * name; int D, n_kv, nh; };

template <int qk, int qr, dequantize_kernel_t dk, typename block_t>
void run(const char * tname, const Shape & s, int iters) {
    const int64_t nblk_row = s.D / qk;                       // blocks per head row
    const int64_t nblk = nblk_row * s.nh * s.n_kv;           // whole view (token-major cache rows)
    std::vector<block_t> h(nblk);
    uint32_t r = 12345;
    for (auto & b : h) {
        r = r*1664525u + 1013904223u;
        b.d = __float2half(0.01f + (r >> 8) * 1e-9f);
        for (size_t i = 0; i < sizeof(b.qs); ++i) { r = r*1664525u + 1013904223u; ((uint8_t *) b.qs)[i] = r >> 24; }
    }
    void * dx; half * y0; half * y1;
    const size_t ny = size_t(s.D) * s.n_kv * s.nh;
    CK(cudaMalloc(&dx, nblk * sizeof(block_t)));
    CK(cudaMalloc(&y0, ny * sizeof(half)));
    CK(cudaMalloc(&y1, ny * sizeof(half)));
    CK(cudaMemcpy(dx, h.data(), nblk * sizeof(block_t), cudaMemcpyHostToDevice));
    // strides in blocks: token row = nh*D values, head = D values
    const int64_t s01 = nblk_row * s.nh, s02 = nblk_row, s03 = nblk;

    // g: 0 upstream (79e2e74), 1 patch 0005 geometry, 2 patch 0006 kernel, 3 D2D memcpy of the F16 size (bandwidth reference)
    half * y2;
    CK(cudaMalloc(&y2, ny * sizeof(half)));
    auto go = [&](int g) {
        switch (g) {
            case 0: case 1: launch<qk, qr, dk>(g, dx, g ? y1 : y0, s.D, s.n_kv, s.nh, s01, s02, s03); break;
            case 2: dequantize_block_q_0_nc<qr == 1><<<dim3((int64_t(s.n_kv)*(s.D/8) + 255)/256, s.nh), 256>>>(
                        dx, y2, s.D, s.n_kv, s.nh, s.nh, s01, s02, s03); break;
            case 3: CK(cudaMemcpyAsync(y2, y1, ny * sizeof(half), cudaMemcpyDeviceToDevice)); break;
        }
    };
    float ms[4];
    for (int g = 0; g < 4; ++g) {
        go(g); // warm-up
        CK(cudaGetLastError());
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) go(g);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms[g], a, b));
        ms[g] /= iters;
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
        if (g == 2) {  // check 0006 before the memcpy overwrites y2
            std::vector<uint16_t> o0(ny), o2(ny);
            CK(cudaMemcpy(o0.data(), y0, ny * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(o2.data(), y2, ny * 2, cudaMemcpyDeviceToHost));
            if (std::memcmp(o0.data(), o2.data(), ny * 2) != 0) { std::printf("%s %s: 0006 MISMATCH\n", tname, s.name); std::exit(1); }
        }
    }
    std::vector<uint16_t> o0(ny), o1(ny);
    CK(cudaMemcpy(o0.data(), y0, ny * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o1.data(), y1, ny * 2, cudaMemcpyDeviceToHost));
    const bool same = std::memcmp(o0.data(), o1.data(), ny * 2) == 0;
    const double bytes = nblk * sizeof(block_t) + ny * sizeof(half);
    std::printf("%-5s %-24s D=%3d kv=%6d nh=%d | upstream %7.1f us %5.1f GB/s | 0005 rows/block %7.1f us %5.2fx | 0006 vec8 %7.1f us %5.2fx | f16 D2D copy %7.1f us | %s\n",
        tname, s.name, s.D, s.n_kv, s.nh, ms[0]*1e3, bytes/ms[0]/1e6, ms[1]*1e3, ms[0]/ms[1], ms[2]*1e3, ms[0]/ms[2], ms[3]*1e3,
        same ? "outputs bit-identical" : "MISMATCH");
    CK(cudaFree(dx)); CK(cudaFree(y0)); CK(cudaFree(y1)); CK(cudaFree(y2));
    if (!same) std::exit(1);
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s (sm_%d%d, %d SMs); one K or V view per call, %d iterations\n", p.name, p.major, p.minor, p.multiProcessorCount, iters);
    const Shape shapes[] = {
        {"test-backend-ops D=64",   64,  7680, 8},   // the 7.2x case in phase0-cloud-2026-10-10 §3
        {"Llama-3.1-8B @16K",      128, 16384, 8},
        {"test-backend-ops D=256", 256, 16384, 2},   // the 2.4x case
        {"Qwen3.8-27B @32K",       256, 32768, 4},
    };
    for (const Shape & s : shapes) {
        const size_t need = size_t(s.D) * s.n_kv * s.nh * (3*2 + 2); // two F16 outputs + input, generous
        size_t fr, tot;
        CK(cudaMemGetInfo(&fr, &tot));
        if (need > fr) { std::printf("skip %s (needs %zu MB)\n", s.name, need >> 20); continue; }
        run<QK8_0, QR8_0, dequantize_q8_0, block_q8_0>("q8_0", s, iters);
        run<QK4_0, QR4_0, dequantize_q4_0, block_q4_0>("q4_0", s, iters);
    }
    return 0;
}
