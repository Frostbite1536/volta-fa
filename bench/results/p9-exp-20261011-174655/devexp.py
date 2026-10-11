# Experiment switches on top of devmods.py (not part of the patch): GGML_CUDA_MMQ_VOLTA_NOLOAD (all row tiles read rows
# 0..31: weights come from L2, results wrong; timing only), GGML_CUDA_MMQ_VOLTA_MB (instances with more blocks per SM),
# GGML_CUDA_MMQ_VOLTA_NOCOMPUTE (loads and staging only, no decode/mma; results wrong; timing only).
def edit(p, pairs):
    s = open(p).read()
    for a, b, cnt in pairs:
        assert s.count(a) == cnt, (p, a[:60], s.count(a))
        s = s.replace(a, b)
    open(p, 'w').write(s)
edit('ggml/src/ggml-cuda/mmq-volta.cu', [
("""template <ggml_type type, int ntt, int NR>
static __global__ void __launch_bounds__(NR*WARP_SIZE, 1) mul_mat_q_volta(""", """template <ggml_type type, int ntt, int NR, int MB = 1, int NC = 0> // DEV MB NC
static __global__ void __launch_bounds__(NR*WARP_SIZE, MB) mul_mat_q_volta(""", 1),
("""template <ggml_type type, int ntt, int NR>
static void mul_mat_q_volta_launch(""", """template <ggml_type type, int ntt, int NR, int MB = 1, int NC = 0> // DEV MB NC
static void mul_mat_q_volta_launch(""", 1),
("""    const size_t smem = size_t(NR)*WARP_SIZE*mmq_volta_type<type>::stage
        + size_t(T)*MMQ_VOLTA_ACT_STRIDE*sizeof(half) + size_t(T)*8*sizeof(half);""", """    // DEV NC: shared memory padded so the same number of blocks per SM fit as with the real kernel (2 / 4)
    const size_t smem = std::max<size_t>(NC ? (ntt == 2 ? 40*1024 : 24*1024) : 0, size_t(NR)*WARP_SIZE*mmq_volta_type<type>::stage
        + size_t(T)*MMQ_VOLTA_ACT_STRIDE*sizeof(half) + size_t(T)*8*sizeof(half)); // DEV""", 1),
("mul_mat_q_volta<type, ntt, NR>", "mul_mat_q_volta<type, ntt, NR, MB, NC>", 3),
("""        if constexpr (type == GGML_TYPE_Q6_K) {
            mmq_volta_superblock_q6_k""", """        if constexpr (NC == 1) { // DEV
            tot[0][0] += __uint_as_float(reinterpret_cast<const uint32_t *>(stage)[lane]); // DEV
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            mmq_volta_superblock_q6_k""", 1),
("""        vx, xp, xm, dst, ws.ptr, counters.ptr, N, K, n, row_pitch, s1d);""",
 """        vx, xp, xm, dst, ws.ptr, counters.ptr, getenv("GGML_CUDA_MMQ_VOLTA_NOLOAD") ? std::min(N, 32) : N, K, n, row_pitch, s1d); // DEV""", 1),
("""    if (n <= 8 && dev_nr == 4""", """    static const int dev_mb = getenv("GGML_CUDA_MMQ_VOLTA_MB") ? atoi(getenv("GGML_CUDA_MMQ_VOLTA_MB")) : 1; // DEV
    static const bool dev_nc = getenv("GGML_CUDA_MMQ_VOLTA_NOCOMPUTE") != nullptr; // DEV
    if (dev_nc && n <= 8) { // DEV
        mul_mat_q_volta_launch<type, 1, 2, 1, 1>(ctx, vx, xp.get(), xm.ptr, d, N, K, n, src0->nb[1], s1d); // DEV
    } else if (dev_nc && N % (4*WARP_SIZE) == 0) { // DEV
        mul_mat_q_volta_launch<type, 2, 4, 1, 1>(ctx, vx, xp.get(), xm.ptr, d, N, K, n, src0->nb[1], s1d); // DEV
    } else if (dev_mb > 1 && n <= 8) { // DEV
        mul_mat_q_volta_launch<type, 1, 2, 6>(ctx, vx, xp.get(), xm.ptr, d, N, K, n, src0->nb[1], s1d); // DEV
    } else if (dev_mb > 1 && N % (4*WARP_SIZE) == 0) { // DEV
        mul_mat_q_volta_launch<type, 2, 4, 3>(ctx, vx, xp.get(), xm.ptr, d, N, K, n, src0->nb[1], s1d); // DEV
    } else if (n <= 8 && dev_nr == 4""", 1),
])
