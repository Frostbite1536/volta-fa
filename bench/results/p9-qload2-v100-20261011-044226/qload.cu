// P-9 Phase 0: measure native versus fragment-order quantised-weight streaming
// for the Volta weight-layout decision (no model weights or external kernel code).
// Build/run: nvcc -O3 -std=c++17 -arch=sm_70 bench/micro/qload.cu -o build/qload && build/qload [iters]
// Laptop: nvcc -O3 -std=c++17 -arch=sm_52 -DQLOAD_NO_MMA bench/micro/qload.cu -o build/qload52
// copy: native D2D reference (also writes); stream: contiguous 16-byte reads.
// native: lane l streams row 32t+l, using the largest aligned load width.
// native_smem/native_smem_mma: Q4_K/Q5_K, full-line native loads followed by
// a warp-private shared-memory transpose, then row XOR or the same decode + MMA.
// repacked: lane l reads its 16 bytes in each contiguous 512-byte tile chunk;
// rows are zero-padded to 16 bytes. Both layouts split each row across P*S warps.
// native_mma/repacked_mma: Q4_K/Q5_K, the decode + tensor-core work of the real kernel per 16 bytes (32 codes):
// 16 HFMA2 and 8 m8n8k4 MMA per lane, with runtime constants. B (activations) stays in registers, so this is a
// lower bound on the per-weight work (no activation loads, no scales/mins).
// This is an arithmetic/load microbenchmark, not a quantised matmul correctness test.
// Decimal MB/GB/s use native bytes for EVERY variant, including the padded copy.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

// Opaque byte blocks: only their sizes and weight counts matter for this experiment.
struct Q4_K   { uint8_t bytes[144]; };
struct Q5_K   { uint8_t bytes[176]; };
struct Q6_K   { uint8_t bytes[210]; };
struct IQ4_XS { uint8_t bytes[136]; };
struct Q8_0   { uint8_t bytes[34]; };
static_assert(sizeof(Q4_K) == 144, "Q4_K block size");
static_assert(sizeof(Q5_K) == 176, "Q5_K block size");
static_assert(sizeof(Q6_K) == 210, "Q6_K block size");
static_assert(sizeof(IQ4_XS) == 136, "IQ4_XS block size");
static_assert(sizeof(Q8_0) == 34, "Q8_0 block size");

__device__ __forceinline__ uint4 load16(const uint8_t * p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

template<int W>
__device__ __forceinline__ uint32_t load_xor(const uint8_t * p) {
    if constexpr (W == 16) {
        const uint4 v = load16(p);
        return v.x ^ v.y ^ v.z ^ v.w;
    } else if constexpr (W == 8) {
        uint32_t a, b;
        asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(a), "=r"(b) : "l"(p) : "memory");
        return a ^ b;
    } else if constexpr (W == 4) {
        uint32_t a;
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(a) : "l"(p) : "memory");
        return a;
    } else {
        uint16_t a;
        asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(a) : "l"(p) : "memory");
        return a;
    }
}

__global__ void stream(const uint8_t * data, size_t bytes, uint32_t * out) {
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    uint32_t checksum = 0;
    // All requested native buffer sizes are multiples of 16.
    for (size_t i = tid; i < bytes / 16; i += stride) checksum ^= load_xor<16>(data + i * 16);
    out[tid] = checksum;
}

// Segment boundaries are ceil(units/(P*S)) load units apart; the final segment
// is shorter (or empty). Native output is indexed [segment][row], as is repacked.
template<int W>
__global__ void native(const uint8_t * data, size_t pitch, int N, int P, uint32_t * out) {
    const int S = blockDim.x / 32;
    const int tile = blockIdx.x / P;
    const int segment = (blockIdx.x % P) * S + threadIdx.x / 32;
    const int row = tile * 32 + threadIdx.x % 32;
    const size_t units = pitch / W;
    const size_t length = (units + P * S - 1) / (P * S);
    const size_t end = min(units, (segment + 1) * length);
    uint32_t checksum = 0;
    for (size_t i = segment * length; i < end; ++i) checksum ^= load_xor<W>(data + row * pitch + i * W);
    out[size_t(segment) * N + row] = checksum;
}

__global__ void repacked(const uint8_t * data, size_t padded_pitch, int N, int P, uint32_t * out) {
    const int S = blockDim.x / 32;
    const int tile = blockIdx.x / P;
    const int segment = (blockIdx.x % P) * S + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const size_t chunks = padded_pitch / 16;
    const size_t length = (chunks + P * S - 1) / (P * S);
    const size_t end = min(chunks, (segment + 1) * length);
    const uint8_t * base = data + size_t(tile) * 32 * padded_pitch;
    uint32_t checksum = 0;
    for (size_t c = segment * length; c < end; ++c) checksum ^= load_xor<16>(base + (c * 32 + lane) * 16);
    out[size_t(segment) * N + tile * 32 + lane] = checksum;
}

constexpr int SMEM_UNITS = 8;
constexpr int SMEM_ROW_UNITS = SMEM_UNITS + 1;
constexpr size_t SMEM_WARP_BYTES = 32 * SMEM_ROW_UNITS * sizeof(uint4);

__device__ __forceinline__ uint4 load16_smem(const uint4 * p) {
    uint4 v;
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(address) : "memory");
    return v;
}

__device__ __forceinline__ void stage_native(const uint8_t * tile_data, size_t pitch,
                                            size_t step, int count, uint4 * staging) {
    const int lane = threadIdx.x % 32;
    const int unit = lane % SMEM_UNITS;
    // Each instruction reads four complete 128-byte row lines in a full step.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (unit < count) {
            const int row = 4 * i + lane / SMEM_UNITS;
            const uint4 v = load16(tile_data + size_t(row) * pitch + (step + unit) * 16);
            const uint32_t address = static_cast<uint32_t>(
                __cvta_generic_to_shared(staging + row * SMEM_ROW_UNITS + unit));
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                         : : "r"(address), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
        }
    }
    // The 144-byte (36-word) row stride advances each row by four banks.
    // Each LDS.128 transaction covers eight lanes: their four-word vectors
    // occupy all 32 banks once. A 128-byte row stride would alias those banks.
    __syncwarp();
    // Keep staging simple; software pipelining is a follow-up experiment.
}

__global__ void native_smem(const uint8_t * data, size_t pitch, int N, int P, uint32_t * out) {
    extern __shared__ uint4 smem[];
    const int S = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int tile = blockIdx.x / P;
    const int segment = (blockIdx.x % P) * S + warp;
    const int lane = threadIdx.x % 32;
    const size_t units = pitch / 16;
    const size_t length = (units + P * S - 1) / (P * S);
    const size_t end = min(units, (segment + 1) * length);
    uint4 * staging = smem + warp * 32 * SMEM_ROW_UNITS;
    const uint8_t * tile_data = data + size_t(tile) * 32 * pitch;
    uint32_t checksum = 0;
    for (size_t step = segment * length; step < end; step += SMEM_UNITS) {
        const int count = int(min(size_t(SMEM_UNITS), end - step));
        stage_native(tile_data, pitch, step, count, staging);
        for (int unit = 0; unit < count; ++unit) {
            const uint4 v = load16_smem(staging + lane * SMEM_ROW_UNITS + unit);
            checksum ^= v.x ^ v.y ^ v.z ^ v.w;
        }
        __syncwarp(); // All reads finish before this warp overwrites its buffer.
    }
    out[size_t(segment) * N + tile * 32 + lane] = checksum;
}

#ifndef QLOAD_NO_MMA
__host__ __device__ __forceinline__ half2 half2_from_bits(uint32_t bits) {
    __half2_raw raw;
    raw.x = uint16_t(bits);
    raw.y = uint16_t(bits >> 16);
    return half2(raw);
}

__host__ __device__ __forceinline__ uint32_t half2_bits(half2 value) {
    const __half2_raw raw = value;
    return uint32_t(raw.x) | (uint32_t(raw.y) << 16);
}

__device__ __forceinline__ void decode_mma(uint32_t x, half2 s, half2 z,
                                         uint32_t b0, uint32_t b1, float (&acc)[8]) {
    // All 8 codes of the word, as the real kernel decodes them: 4 half2 -> 2 mma (4 K values each per lane).
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        const uint32_t c0 = ((x >> (8*pair))     & 0x000F000Fu) | 0x64006400u;
        const uint32_t c1 = ((x >> (8*pair + 4)) & 0x000F000Fu) | 0x64006400u;
        const uint32_t a0 = half2_bits(__hfma2(half2_from_bits(c0), s, z));
        const uint32_t a1 = half2_bits(__hfma2(half2_from_bits(c1), s, z));
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
              "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    }
}

// Two distinct template instantiations give one kernel for each MMA layout.
template<bool Repacked>
__global__ void layout_mma(const uint8_t * data, size_t pitch, int N, int P,
                          half2 s, half2 z, uint32_t b0, uint32_t b1, float * out) {
    const int S = blockDim.x / 32;
    const int tile = blockIdx.x / P;
    const int segment = (blockIdx.x % P) * S + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const size_t chunks = pitch / 16;
    const size_t length = (chunks + P * S - 1) / (P * S);
    const size_t end = min(chunks, (segment + 1) * length);
    float acc[8] = {};
    for (size_t c = segment * length; c < end; ++c) {
        const size_t offset = Repacked ? size_t(tile) * 32 * pitch + (c * 32 + lane) * 16
                                      : size_t(tile * 32 + lane) * pitch + c * 16;
        const uint4 v = load16(data + offset);
        decode_mma(v.x, s, z, b0, b1, acc);
        decode_mma(v.y, s, z, b0, b1, acc);
        decode_mma(v.z, s, z, b0, b1, acc);
        decode_mma(v.w, s, z, b0, b1, acc);
    }
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum += acc[i];
    out[size_t(segment) * N + tile * 32 + lane] = sum;
}

__global__ void native_smem_mma(const uint8_t * data, size_t pitch, int N, int P,
                               half2 s, half2 z, uint32_t b0, uint32_t b1, float * out) {
    extern __shared__ uint4 smem[];
    const int S = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int tile = blockIdx.x / P;
    const int segment = (blockIdx.x % P) * S + warp;
    const int lane = threadIdx.x % 32;
    const size_t units = pitch / 16;
    const size_t length = (units + P * S - 1) / (P * S);
    const size_t end = min(units, (segment + 1) * length);
    uint4 * staging = smem + warp * 32 * SMEM_ROW_UNITS;
    const uint8_t * tile_data = data + size_t(tile) * 32 * pitch;
    float acc[8] = {};
    for (size_t step = segment * length; step < end; step += SMEM_UNITS) {
        const int count = int(min(size_t(SMEM_UNITS), end - step));
        stage_native(tile_data, pitch, step, count, staging);
        for (int unit = 0; unit < count; ++unit) {
            const uint4 v = load16_smem(staging + lane * SMEM_ROW_UNITS + unit);
            decode_mma(v.x, s, z, b0, b1, acc);
            decode_mma(v.y, s, z, b0, b1, acc);
            decode_mma(v.z, s, z, b0, b1, acc);
            decode_mma(v.w, s, z, b0, b1, acc);
        }
        __syncwarp();
    }
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum += acc[i];
    out[size_t(segment) * N + tile * 32 + lane] = sum;
}
#endif

struct Type { const char * name; int block_bytes, weights; bool mma; };
struct Shape { int N, K; };
enum Variant { COPY, STREAM, NATIVE, REPACKED, NATIVE_MMA, REPACKED_MMA,
               NATIVE_SMEM, NATIVE_SMEM_MMA, VARIANTS };
const char * const names[] = {"copy", "stream", "native", "repacked", "native_mma", "repacked_mma",
                             "native_smem", "native_smem_mma"};
struct Result { const char * type; int N, K, w; size_t bytes; Variant variant; int S, P; double us, gbps; };
struct Summary { const char * type; int N, K; double best[VARIANTS] = {}; bool pass = false; };

// A single event pair is reused, but every measured launch has its own sample.
template<typename Launch>
double median_us(Launch launch, int iters) {
    for (int i = 0; i < 3; ++i) launch();
    CK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
    std::vector<float> times(iters);
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(start));
        launch();
        CK(cudaEventRecord(stop));
        CK(cudaEventSynchronize(stop));
        CK(cudaEventElapsedTime(&times[i], start, stop));
    }
    CK(cudaEventDestroy(start)); CK(cudaEventDestroy(stop));
    std::sort(times.begin(), times.end());
    const int m = iters / 2;
    return 1000.0 * (iters % 2 ? times[m] : (double(times[m - 1]) + times[m]) / 2);
}

bool run(const Type & type, Shape shape, int iters, std::vector<Result> & results, Summary & summary) {
    const int N = shape.N, K = shape.K;
    const size_t pitch = size_t(K / type.weights) * type.block_bytes;
    const size_t padded_pitch = (pitch + 15) / 16 * 16;
    const size_t bytes = size_t(N) * pitch, padded_bytes = size_t(N) * padded_pitch;
    int w = 16;
    while (pitch % w) w /= 2;
    std::printf("%s N=%d K=%d: row_pitch=%zu B, w=%d, native=%zu B, repacked_padded=%zu B (%.6f MB)\n",
                type.name, N, K, pitch, w, bytes, padded_bytes, padded_bytes / 1e6);
    std::vector<uint8_t> host(bytes), packed(padded_bytes, 0);
    uint32_t rng = 12345;
    for (uint8_t & b : host) { rng = rng * 1664525u + 1013904223u; b = rng >> 24; }
    std::vector<uint32_t> cpu(N, 0);
    for (int row = 0; row < N; ++row) {
        // CPU reference: 32-bit words, with a zero-extended u16 tail.
        for (size_t j = 0; j < pitch; j += 4) {
            uint32_t word = 0;
            std::memcpy(&word, host.data() + row * pitch + j, std::min(size_t(4), pitch - j));
            cpu[row] ^= word;
        }
        for (size_t c = 0; c < padded_pitch / 16; ++c) {
            const size_t start = c * 16;
            const size_t dest = size_t(row / 32) * 32 * padded_pitch + (c * 32 + row % 32) * 16;
            if (start < pitch) std::memcpy(packed.data() + dest, host.data() + row * pitch + start,
                                          std::min(size_t(16), pitch - start));
        }
    }
    uint8_t * d_native, * d_packed, * d_copy;
    uint32_t * d_xor;
    CK(cudaMalloc(&d_native, bytes)); CK(cudaMalloc(&d_packed, padded_bytes));
    CK(cudaMalloc(&d_copy, bytes)); CK(cudaMalloc(&d_xor, size_t(N) * 32 * sizeof(uint32_t)));
    CK(cudaMemcpy(d_native, host.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_packed, packed.data(), padded_bytes, cudaMemcpyHostToDevice));
#ifndef QLOAD_NO_MMA
    float * d_float;
    CK(cudaMalloc(&d_float, size_t(N) * 32 * sizeof(float)));
    const half2 scale = __floats2half2_rn(0.0625f, 0.0625f);
    const half2 zero = __floats2half2_rn(-64.0f, -64.0f);
    const uint32_t b = half2_bits(__floats2half2_rn(0.125f, 0.125f));
#endif
    auto launch = [&](Variant v, int S, int P) {
        const int blocks = (N / 32) * P, threads = 32 * S;
        switch (v) {
            case COPY: CK(cudaMemcpyAsync(d_copy, d_native, bytes, cudaMemcpyDeviceToDevice)); break;
            case STREAM: stream<<<blocks, threads>>>(d_native, bytes, d_xor); break;
            case NATIVE:
                switch (w) {
                    case 16: native<16><<<blocks, threads>>>(d_native, pitch, N, P, d_xor); break;
                    case 8: native<8><<<blocks, threads>>>(d_native, pitch, N, P, d_xor); break;
                    case 4: native<4><<<blocks, threads>>>(d_native, pitch, N, P, d_xor); break;
                    case 2: native<2><<<blocks, threads>>>(d_native, pitch, N, P, d_xor); break;
                }
                break;
            case REPACKED: repacked<<<blocks, threads>>>(d_packed, padded_pitch, N, P, d_xor); break;
            case NATIVE_SMEM:
                native_smem<<<blocks, threads, S * SMEM_WARP_BYTES>>>(d_native, pitch, N, P, d_xor); break;
#ifndef QLOAD_NO_MMA
            case NATIVE_MMA: layout_mma<false><<<blocks, threads>>>(d_native, pitch, N, P, scale, zero, b, b, d_float); break;
            case REPACKED_MMA: layout_mma<true><<<blocks, threads>>>(d_packed, padded_pitch, N, P, scale, zero, b, b, d_float); break;
            case NATIVE_SMEM_MMA:
                native_smem_mma<<<blocks, threads, S * SMEM_WARP_BYTES>>>(d_native, pitch, N, P, scale, zero, b, b, d_float); break;
#endif
            default: std::fprintf(stderr, "unsupported variant\n"); std::exit(1);
        }
        CK(cudaGetLastError());
    };
    // Verify every XOR layout for every geometry before any timing of this shape.
    bool pass = true;
    for (int S : {4, 8}) for (int P : {1, 2, 4}) for (Variant v : {NATIVE, REPACKED, NATIVE_SMEM}) {
        if (v == NATIVE_SMEM && !type.mma) continue;
        launch(v, S, P);
        std::vector<uint32_t> partial(size_t(N) * S * P);
        CK(cudaMemcpy(partial.data(), d_xor, partial.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int row = 0; row < N; ++row) {
            uint32_t checksum = 0;
            for (int segment = 0; segment < S * P; ++segment) checksum ^= partial[size_t(segment) * N + row];
            if (checksum != cpu[row]) {
                std::printf("FAIL %s N=%d K=%d %s S=%d P=%d row=%d CPU=%08x GPU=%08x\n",
                            type.name, N, K, names[v], S, P, row, cpu[row], checksum);
                pass = false;
                break;
            }
        }
    }
    std::printf("%s %s N=%d K=%d: native/repacked%s row XOR, all 6 S/P combinations%s\n",
                pass ? "PASS" : "FAIL", type.name, N, K, type.mma ? "/native_smem" : "",
                pass ? "" : "; timing skipped");
    summary = Summary{type.name, N, K, {}, pass};
    if (pass) {
#ifndef QLOAD_NO_MMA
        if (type.mma) {
            for (int S : {4, 8}) for (int P : {1, 2, 4}) for (Variant v : {NATIVE_MMA, REPACKED_MMA, NATIVE_SMEM_MMA}) {
                launch(v, S, P);
                std::vector<float> partial(size_t(N) * S * P);
                CK(cudaMemcpy(partial.data(), d_float, partial.size() * sizeof(float), cudaMemcpyDeviceToHost));
                double sum = 0;
                for (float value : partial) sum += value;
                std::printf("MMA sum %s N=%d K=%d %s S=%d P=%d: %.9g\n", type.name, N, K, names[v], S, P, sum);
            }
        }
#endif
        auto measure = [&](Variant v, int S, int P) {
            const double us = median_us([&]() { launch(v, S, P); }, iters);
            const double gbps = bytes / us / 1000;
            results.push_back({type.name, N, K, w, bytes, v, S, P, us, gbps});
            summary.best[v] = std::max(summary.best[v], gbps);
        };
        // Baselines run once, using S=8/P=1 for the stream kernel.
        measure(COPY, 0, 0);
        measure(STREAM, 8, 1);
        for (int S : {4, 8}) for (int P : {1, 2, 4}) {
            measure(NATIVE, S, P); measure(REPACKED, S, P);
            if (type.mma) measure(NATIVE_SMEM, S, P);
#ifndef QLOAD_NO_MMA
            if (type.mma) {
                measure(NATIVE_MMA, S, P); measure(REPACKED_MMA, S, P); measure(NATIVE_SMEM_MMA, S, P);
            }
#endif
        }
    }
#ifndef QLOAD_NO_MMA
    CK(cudaFree(d_float));
#endif
    CK(cudaFree(d_native)); CK(cudaFree(d_packed)); CK(cudaFree(d_copy)); CK(cudaFree(d_xor));
    return pass;
}

int main(int argc, char ** argv) {
    int iters = 20;
    if (argc > 2) { std::fprintf(stderr, "usage: %s [positive iters]\n", argv[0]); return 1; }
    if (argc == 2) {
        char * end;
        errno = 0;
        const long parsed = std::strtol(argv[1], &end, 10);
        if (errno || end == argv[1] || *end || parsed <= 0 || parsed > INT_MAX) {
            std::fprintf(stderr, "iters must be a positive integer\n"); return 1;
        }
        iters = int(parsed);
    }
    cudaDeviceProp device;
    CK(cudaGetDeviceProperties(&device, 0));
#ifndef QLOAD_NO_MMA
    if (device.major != 7 || device.minor != 0) {
        std::fprintf(stderr, "MMA benchmark requires sm_70; use -DQLOAD_NO_MMA for the laptop check\n"); return 1;
    }
#endif
    std::printf("device: %s (sm_%d%d, %d SMs); %d timed launches, 3 warm-ups per variant/geometry\n",
                device.name, device.major, device.minor, device.multiProcessorCount, iters);
#ifdef QLOAD_NO_MMA
    std::printf("QLOAD_NO_MMA: MMA variants not run. Laptop/Maxwell bandwidth is NOT V100 bandwidth. V100: not run.\n");
#endif
    std::printf("Decimal MB/GB/s; GB/s = native bytes / median time (copy also writes). No L2 flush.\n");
    const Type types[] = {{"Q4_K", sizeof(Q4_K), 256, true}, {"Q5_K", sizeof(Q5_K), 256, true},
                          {"Q6_K", sizeof(Q6_K), 256, false}, {"IQ4_XS", sizeof(IQ4_XS), 256, false},
                          {"Q8_0", sizeof(Q8_0), 32, false}};
    const Shape k_shapes[] = {{17408, 5120}, {5120, 17408}, {10240, 5120}, {5120, 6144}};
    const Shape q8_shapes[] = {{14336, 4096}, {4096, 14336}, {4096, 4096}};
    std::vector<Result> results;
    std::vector<Summary> summaries;
    bool pass = true;
    for (const Type & type : types) {
        const Shape * shapes = type.weights == 32 ? q8_shapes : k_shapes;
        const int count = type.weights == 32 ? 3 : 4;
        for (int i = 0; i < count; ++i) {
            Summary summary;
            if (!run(type, shapes[i], iters, results, summary)) pass = false;
            summaries.push_back(summary);
        }
    }
    std::printf("\n| type | N | K | MB | w | variant | S | P | median µs | GB/s |\n");
    std::printf("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |\n");
    for (const Result & r : results) {
        std::printf("| %s | %d | %d | %.6f | %d | %s | %d | %d | %.3f | %.3f |\n",
                    r.type, r.N, r.K, r.bytes / 1e6, r.w, names[r.variant], r.S, r.P, r.us, r.gbps);
    }
    std::printf("\nBest GB/s per shape (ratios use independently best S/P geometries; — = not run):\n\n");
    std::printf("| type | N | K | stream | native | repacked | native_mma | repacked_mma | native_smem | native_smem_mma | native/repacked | native_mma/repacked_mma | native_smem/repacked | native_smem_mma/repacked_mma |\n");
    std::printf("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for (const Summary & s : summaries) {
        std::printf("| %s | %d | %d |", s.type, s.N, s.K);
        for (int v = STREAM; v < VARIANTS; ++v) {
            if (s.best[v] > 0) std::printf(" %.3f |", s.best[v]); else std::printf(" — |");
        }
        if (s.pass) std::printf(" %.4f |", s.best[NATIVE] / s.best[REPACKED]); else std::printf(" — |");
        if (s.best[REPACKED_MMA] > 0) std::printf(" %.4f |", s.best[NATIVE_MMA] / s.best[REPACKED_MMA]);
        else std::printf(" — |");
        if (s.best[NATIVE_SMEM] > 0) std::printf(" %.4f |", s.best[NATIVE_SMEM] / s.best[REPACKED]);
        else std::printf(" — |");
        if (s.best[NATIVE_SMEM_MMA] > 0) std::printf(" %.4f |\n", s.best[NATIVE_SMEM_MMA] / s.best[REPACKED_MMA]);
        else std::printf(" — |\n");
    }
    return pass ? 0 : 1;
}
