// DRAM bandwidth of the weight access patterns of mul_mat_q_volta (V100): each warp owns 32 rows and per step loads
// SB superblocks (bytes B each) of every row into registers (16-byte loads, consecutive lanes on consecutive units of
// a row), then stores them to shared memory, with the next step's loads issued before the store (depth 1), as the
// kernel does. Variants: native layout (row pitch = K/256*B) vs tile-contiguous layout (a warp's 32 rows x SB
// superblocks are one contiguous run), SB = 1, 2, 4. nvcc -O3 -arch=sm_70 bwtest.cu -o bwtest
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void stream(const uint4 * __restrict__ w, size_t n, unsigned * out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x*size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x)*blockDim.x) acc += __ldg(w + i).x;
    if (acc == 0x12345678u) out[0] = acc;
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

template <int B, int SB, bool TILED, int NR>
__global__ void __launch_bounds__(NR*32, 1) bw(const char * __restrict__ w, int N, int nsb, int64_t pitch, unsigned * out) {
    constexpr int U = B*SB/16; // units per row per step
    extern __shared__ uint4 sm[];
    uint4 * stage = sm + (threadIdx.x/32)*32*U;
    const int lane = threadIdx.x % 32;
    const int tile = blockIdx.x*NR + threadIdx.x/32;
    const int P = gridDim.y, per = (nsb/SB + P - 1)/P;
    const int s0 = blockIdx.y*per, s1 = min(nsb/SB, s0 + per);
    auto addr = [&](int s, int f) -> const uint4 * {
        const int r = f / U, u = f % U;
        if (TILED) return (const uint4 *)(w + (int64_t(tile)*(nsb/SB) + s)*32*U*16 + int64_t(r)*U*16 + u*16);
        return (const uint4 *)(w + int64_t(min(tile*32 + r, N - 1))*pitch + int64_t(s)*SB*B + u*16);
    };
    uint4 reg[U];
    unsigned acc = 0;
    if (s0 < s1) {
#pragma unroll
        for (int i = 0; i < U; ++i) reg[i] = __ldg(addr(s0, 32*i + lane));
    }
    for (int s = s0; s < s1; ++s) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < U; ++i) stage[32*i + lane] = reg[i];
        if (s + 1 < s1) {
#pragma unroll
            for (int i = 0; i < U; ++i) reg[i] = __ldg(addr(s + 1, 32*i + lane));
        }
        __syncthreads();
        acc += stage[(lane*7) % (32*U)].x;
    }
    if (acc == 0x12345678u) out[0] = acc;
}

template <int B, int SB, bool TILED, int NR>
static int run(const char * w, int N, int K, unsigned * out, int P) {
    const int nsb = K/256;
    const int64_t pitch = int64_t(nsb)*B;
    // padded so that as many blocks fit per SM as with the real kernel (2 of 4 warps, 4 of 2 warps)
    const size_t smem = std::max<size_t>(NR == 4 ? 40*1024 : 24*1024, size_t(NR)*32*(B*SB/16)*16);
    auto k = bw<B, SB, TILED, NR>;
    if (smem > 48*1024) CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, NR*32, smem));
    dim3 grid(N/(NR*32), P);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) k<<<grid, NR*32, smem>>>(w, N, nsb, pitch, out);
    CK(cudaDeviceSynchronize());
    const int it = 50;
    cudaEventRecord(a);
    for (int i = 0; i < it; ++i) k<<<grid, NR*32, smem>>>(w, N, nsb, pitch, out);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    const double bytes = double(N)*pitch;
    printf("B=%3d SB=%d %-6s NR=%d P=%d occ=%d: %7.1f us  %6.1f GB/s\n", B, SB, TILED ? "tiled" : "native", NR, P, occ,
           1e3*ms/it, bytes/(1e-3*ms/it)/1e9);
    return 0;
}

int main() {
    const int N = 17408, K = 5120;
    char * w; unsigned * out;
    CK(cudaMalloc(&w, size_t(N)*K)); CK(cudaMalloc(&out, 4));
    CK(cudaMemset(w, 1, size_t(N)*K));
    // Q4_K-like (144 B), Q5_K-like (176 B), Q8_0-like (272 B)
    run<144, 1, false, 4>(w, N, K, out, 1); run<144, 2, false, 4>(w, N, K, out, 1); run<144, 4, false, 4>(w, N, K, out, 1);
    run<144, 1, true, 4>(w, N, K, out, 1);  run<144, 2, true, 4>(w, N, K, out, 1);
    run<144, 1, false, 2>(w, N, K, out, 1); run<144, 2, false, 2>(w, N, K, out, 1); run<144, 1, true, 2>(w, N, K, out, 1);
    run<176, 1, false, 4>(w, N, K, out, 1); run<176, 2, false, 4>(w, N, K, out, 1); run<176, 1, true, 4>(w, N, K, out, 1);
    run<272, 1, false, 4>(w, N, K, out, 1); run<272, 2, false, 4>(w, N, K, out, 1); run<272, 1, true, 4>(w, N, K, out, 1);
    // pure streaming reference: grid-stride 16-byte loads over the whole buffer
    {
        const size_t n = size_t(N)*(K/256)*144/16;
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        for (int i = 0; i < 3; ++i) stream<<<80*8, 256>>>((const uint4 *) w, n, out);
        cudaEventRecord(a);
        for (int i = 0; i < 50; ++i) stream<<<80*8, 256>>>((const uint4 *) w, n, out);
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b);
        printf("stream: %7.1f us  %6.1f GB/s\n", 1e3*ms/50, n*16.0/(1e-3*ms/50)/1e9);
    }
    return 0;
}
