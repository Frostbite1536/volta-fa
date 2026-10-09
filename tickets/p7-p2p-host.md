# Ticket p7-p2p-host: topology probe, peer copy with staged fallback, bandwidth tool

Your current directory is a git worktree of volta-fa (branch `p7-p2p-host`, based on `main` @ `cec21bb`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Volta sm_70 ONLY, CUDA 12.x. This ticket is HOST code only (C++17 calling the CUDA runtime API). Write no `__global__` kernels and no `.cu` files.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/volta-fa/.venv/bin:$PATH;`. Never install anything. Build into `build/` inside this worktree. There is no nvcc here: configure with `cmake -S . -B build -G Ninja -DVFA_CUDART_ROOT=/home/p51/Github/volta-fa/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime`.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen or edit existing tests. Do not touch `include/volta_fa/attention.h`, `include/volta_fa/tolerance.h`, `src/reference/`, or existing test files.
- Never invent numbers. This machine has ONE Maxwell GPU (Quadro M2200) and your sandbox may not be able to open it at all: tests that need a GPU must SKIP (exit code 77, registered with `SKIP_RETURN_CODE 77`) when `cudaGetDeviceCount` fails or returns 0, and you must report which tests actually ran vs skipped. Peer paths need 2 GPUs: "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
Phase 3 (docs/ROADMAP.md) is about two V100 PCIe cards without NVLink. Before any overlap work we need: what the topology is, whether peer access works (it can fail on consumer boards / IOMMU setups, see the `iommu=pt` note in docs/TOOLCHAIN.md), a copy primitive that uses P2P when possible and a pinned-host staged path otherwise, and a bandwidth tool to run on day one. Read docs/INVARIANTS.md INV-P2P-001..003 first: capability check + fallback with the active path visible; streams/events, never device-wide syncs in the copy path; staging buffers reused only after an event proves the previous use finished.

## API (manager decided; implement exactly, you may add private helpers)

`include/volta_fa/check.h`:
```cpp
// Throws std::runtime_error("<file>:<line>: <expr>: <cudaGetErrorString>") on failure.
#define VFA_CUDA_CHECK(expr) ...
```

`include/volta_fa/p2p.h`, namespace `volta_fa::p2p`:
```cpp
struct PeerInfo { bool can_access = false; int perf_rank = -1; bool native_atomics = false; };
struct Topology {
    int device_count = 0;
    std::vector<std::string> name, pci_bus_id;
    std::vector<std::vector<PeerInfo>> peer;   // peer[src][dst]; diagonal left default
};
Topology probe();                          // 0 devices -> device_count = 0, no throw
std::string describe(const Topology& t);   // human-readable table: devices, bus ids, peer matrix

enum class Path { Peer, Staged };
const char* path_name(Path p);

bool p2p_disabled_by_env();                // true iff env VFA_DISABLE_P2P is set to a non-empty value other than "0"

class PeerLink {                           // one direction: src_dev -> dst_dev
public:
    PeerLink(int src_dev, int dst_dev, std::size_t chunk_bytes = 4u << 20);
    ~PeerLink();                           // waits for its own outstanding work (events), frees staging
    PeerLink(const PeerLink&) = delete; PeerLink& operator=(const PeerLink&) = delete;
    Path path() const;
    // Enqueue a copy of `bytes` from src (device memory on src_dev) to dst (device memory on dst_dev).
    // `stream` is a stream on src_dev. The copy starts after all work already enqueued on `stream`,
    // and work enqueued on `stream` afterwards starts only after the copy has fully landed in dst.
    // The host is never blocked except to wait for a staging buffer that is still in use.
    void copy_async(void* dst, const void* src, std::size_t bytes, cudaStream_t stream);
};
```
Semantics:
- Path selection: `Peer` iff `src_dev != dst_dev`, `cudaDeviceCanAccessPeer(dst_dev→src_dev)` (and the reverse direction as needed for the copy) is true, and `!p2p_disabled_by_env()`. Enable peer access in the constructor; treat `cudaErrorPeerAccessAlreadyEnabled` as success and clear it with `cudaGetLastError()`. Leave access enabled in the destructor (other code may rely on it; say so in a comment). `src_dev == dst_dev` is allowed and always uses `Staged` (this is how the staged path gets tested on a single GPU).
- `Peer`: `cudaMemcpyPeerAsync` on `stream`.
- `Staged`: two pinned host buffers (`cudaHostAlloc` with `cudaHostAllocPortable`) of `chunk_bytes`, a private stream on dst_dev, and events. For chunk i: D2H on `stream` into buffer i%2 (first waiting on the event that says buffer i%2's previous H2D finished), record event; H2D on the private dst stream after `cudaStreamWaitEvent` on that event; record "buffer free" event. At the end, record a completion event on the dst stream and make `stream` wait on it. No `cudaDeviceSynchronize`, no `cudaStreamSynchronize` in this path. `bytes == 0` is a no-op.
- Every CUDA call goes through `VFA_CUDA_CHECK`. Save and restore the caller's current device (`cudaGetDevice`/`cudaSetDevice`) in every public function that changes it.

## CMake
- `src/p2p/CMakeLists.txt`: static library `volta_fa_p2p` from `p2p.cpp`, public include dir `include/`, linked to a cudart target.
- Top-level `CMakeLists.txt`: you may add, after the reference/tests blocks, a block that defines the cudart target and adds `src/p2p` and `bench`: if `VFA_HAVE_CUDA` use `find_package(CUDAToolkit REQUIRED)` and `CUDA::cudart`; else if cache path `VFA_CUDART_ROOT` is set and contains `include/cuda_runtime.h` and `lib/libcudart.so.12`, create an IMPORTED SHARED target from those; else `message(STATUS ...)` and skip the P2P targets (CPU-only builds must keep working with no CUDA at all; verify by also configuring a second build dir without `VFA_CUDART_ROOT`).
- `tests/CMakeLists.txt`: append (do not modify existing lines) `test_p2p` only `if(TARGET volta_fa_p2p)`, with `SKIP_RETURN_CODE 77`.
- `bench/CMakeLists.txt`: `p2p_bandwidth` executable.

## Files you may touch
- `include/volta_fa/check.h`, `include/volta_fa/p2p.h`: new.
- `src/p2p/p2p.cpp`, `src/p2p/CMakeLists.txt`: new.
- `tests/test_p2p.cpp`: new; `tests/CMakeLists.txt`: append only.
- `bench/p2p_bandwidth.cpp`, `bench/CMakeLists.txt`: new.
- `CMakeLists.txt`: the block described above only.
Nothing else.

## Tests (`test_p2p`, one per behaviour, each must fail if the behaviour breaks)
1. `probe`: device_count matches `cudaGetDeviceCount`; `describe` mentions every device's bus id.
2. `env`: `p2p_disabled_by_env()` for unset / "" / "0" / "1" (use setenv/unsetenv).
3. `staged_sizes`: device 0 → device 0 (staged), chunk_bytes = 4096, sizes 0, 1, 4095, 4096, 4097, 3.5×4096, 1 MiB; random bytes; verify dst exactly.
4. `stream_order_before`: `cudaMemsetAsync(src, 0xAB, n, stream)` then `copy_async` on the same stream, then D2H of dst on the same stream and `cudaStreamSynchronize(stream)`: dst must be all 0xAB. (Proves the copy waits for prior work and later work waits for the copy, with no device sync.)
5. `buffer_reuse`: many back-to-back copies of different data through one PeerLink with a small chunk (forces staging-buffer reuse); every destination correct.
6. `peer` (only if ≥ 2 devices and peer access is possible, else print "not run" and skip that case): 0→1 and 1→0 correctness with Path::Peer, and with VFA_DISABLE_P2P=1 the same pairs use Path::Staged and are correct.
Each case runnable by name like the existing tests (`test_p2p <case>`), registered as separate ctest entries.

## Bandwidth tool (`p2p_bandwidth`)
`p2p_bandwidth [--sizes 64K,1M,16M,256M] [--iters N] [--chunk 4M] [--csv]`: prints `describe(probe())`, then for every ordered device pair (including same-device staged as a smoke test) and each available path (Peer if possible, Staged always): unidirectional GB/s and per-copy latency (µs) per size; for pairs a≠b also bidirectional (a→b and b→a concurrently, two PeerLinks, two streams) aggregate GB/s. Warm-up iterations discarded; timing with host clock around a batch of N copies followed by synchronizing the involved streams. CSV columns: `path,src,dst,mode,bytes,iters,gbps,latency_us`.

## Must pass before you finish
- Configure + build with `-DVFA_CUDART_ROOT=...` (no warnings from our sources), and a second configure + build WITHOUT it (P2P targets skipped, CPU tests still build).
- `ctest --test-dir build --output-on-failure` (all pass or skip; report which skipped and why)
- `bin/check-forbidden.sh`
- Negative controls: (a) remove the final "stream waits on completion event" step → which test fails (if the GPU is reachable); (b) make the env check ignore "0" → `env` fails; (c) break chunk tail handling → `staged_sizes` fails. Restore after each.

## Done means
Files above exist; both builds clean; ctest green or skipped with reasons. Final report: files changed; each command with its result line; for each test whether it RAN or SKIPPED; negative-control results; `p2p_bandwidth --sizes 64K,1M --iters 5` output if the GPU was reachable (else "not run: <error>"); anything you could not do.
