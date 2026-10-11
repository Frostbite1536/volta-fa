#include "check.h"
#include "volta_fa/check.h"
#include "volta_fa/p2p.h"

#include <algorithm>
#include <cstdlib>
#include <random>

namespace {
using volta_fa::p2p::Path;
using volta_fa::p2p::PeerLink;

struct Skip : std::runtime_error { using std::runtime_error::runtime_error; };

struct EnvGuard {
    bool present;
    std::string value;
    EnvGuard() : present(std::getenv("VFA_DISABLE_P2P") != nullptr),
                 value(present ? std::getenv("VFA_DISABLE_P2P") : "") {}
    ~EnvGuard() {
        if (present) setenv("VFA_DISABLE_P2P", value.c_str(), 1);
        else unsetenv("VFA_DISABLE_P2P");
    }
};

int devices() {
    int count = 0;
    try { VFA_CUDA_CHECK(cudaGetDeviceCount(&count)); }
    catch (const std::exception& e) { throw Skip(e.what()); }
    if (count == 0) throw Skip("cudaGetDeviceCount returned 0");
    return count;
}

// Keep device ownership explicit in the host-only tests as well as the library.
struct Resources {
    int original = 0, device;
    void* src = nullptr;
    void* dst = nullptr;
    unsigned char* host = nullptr;
    cudaStream_t stream{};
    Resources(int dev, std::size_t bytes) : device(dev) {
        VFA_CUDA_CHECK(cudaGetDevice(&original));
        VFA_CUDA_CHECK(cudaSetDevice(device));
        VFA_CUDA_CHECK(cudaMalloc(&src, bytes));
        VFA_CUDA_CHECK(cudaMalloc(&dst, bytes));
        VFA_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocPortable));
        VFA_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Resources() {
        try {
            VFA_CUDA_CHECK(cudaSetDevice(device));
            VFA_CUDA_CHECK(cudaStreamSynchronize(stream));
            VFA_CUDA_CHECK(cudaStreamDestroy(stream));
            VFA_CUDA_CHECK(cudaFree(src));
            VFA_CUDA_CHECK(cudaFree(dst));
            VFA_CUDA_CHECK(cudaFreeHost(host));
            VFA_CUDA_CHECK(cudaSetDevice(original));
        } catch (const std::exception& e) { std::cerr << e.what() << '\n'; std::terminate(); }
    }
};

std::vector<unsigned char> random_bytes(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<unsigned char> data(n);
    for (auto& byte : data) byte = static_cast<unsigned char>(rng());
    return data;
}

void probe() {
    const int count = devices();
    int original = 0, after = -1;
    VFA_CUDA_CHECK(cudaGetDevice(&original));
    const auto topology = volta_fa::p2p::probe();
    VFA_CUDA_CHECK(cudaGetDevice(&after));
    CHECK(after == original);
    CHECK(topology.device_count == count);
    CHECK(topology.name.size() == static_cast<std::size_t>(count));
    CHECK(topology.pci_bus_id.size() == static_cast<std::size_t>(count));
    CHECK(topology.peer.size() == static_cast<std::size_t>(count));
    const auto description = volta_fa::p2p::describe(topology);
    for (int i = 0; i < count; ++i) {
        char bus[32]{};
        VFA_CUDA_CHECK(cudaDeviceGetPCIBusId(bus, sizeof(bus), i));
        CHECK(topology.pci_bus_id[i] == bus);
        CHECK(description.find(bus) != std::string::npos);
        CHECK(topology.peer[i].size() == static_cast<std::size_t>(count));
        CHECK(!topology.peer[i][i].can_access);
        CHECK(topology.peer[i][i].perf_rank == -1);
        CHECK(!topology.peer[i][i].native_atomics);
    }
    std::cout << description;
}

void env() {
    EnvGuard guard;
    CHECK(unsetenv("VFA_DISABLE_P2P") == 0);
    CHECK(!volta_fa::p2p::p2p_disabled_by_env());
    for (const char* value : {"", "0", "1", "false", "00"}) {
        CHECK(setenv("VFA_DISABLE_P2P", value, 1) == 0);
        CHECK(volta_fa::p2p::p2p_disabled_by_env() ==
              (std::string(value) != "" && std::string(value) != "0"));
    }
    CHECK(std::string(volta_fa::p2p::path_name(Path::Peer)) == "Peer");
    CHECK(std::string(volta_fa::p2p::path_name(Path::Staged)) == "Staged");
}

void staged_sizes() {
    devices();
    constexpr std::size_t maximum = 1u << 20;
    Resources r(0, maximum);
    PeerLink link(0, 0, 4096);
    CHECK(link.path() == Path::Staged);
    link.copy_async(nullptr, nullptr, 0, r.stream);
    const auto data = random_bytes(maximum, 7);
    VFA_CUDA_CHECK(cudaMemcpy(r.src, data.data(), maximum, cudaMemcpyHostToDevice));
    // A pageable host-to-device cudaMemcpy may return before the DMA completes, and the copies below run on
    // non-blocking streams that do not order behind it.
    VFA_CUDA_CHECK(cudaDeviceSynchronize());
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{4095}, std::size_t{4096},
                          std::size_t{4097}, std::size_t{14336}, maximum}) {
        // Poison the entire allocation and check the guard after each tail too.
        VFA_CUDA_CHECK(cudaMemsetAsync(r.dst, 0xD3, maximum, r.stream));
        link.copy_async(r.dst, r.src, n, r.stream);
        VFA_CUDA_CHECK(cudaMemcpyAsync(r.host, r.dst, maximum, cudaMemcpyDeviceToHost, r.stream));
        VFA_CUDA_CHECK(cudaStreamSynchronize(r.stream));
        CHECK(std::equal(data.begin(), data.begin() + n, r.host));
        CHECK(std::all_of(r.host + n, r.host + maximum, [](unsigned char c) { return c == 0xD3; }));
    }
    // Destruction must wait for its own work without the caller synchronizing.
    {
        PeerLink retiring(0, 0, 4096);
        retiring.copy_async(r.dst, r.src, maximum, r.stream);
    }
    VFA_CUDA_CHECK(cudaMemcpy(r.host, r.dst, maximum, cudaMemcpyDeviceToHost));
    CHECK(std::equal(data.begin(), data.end(), r.host));
}

void stream_order_before() {
    devices();
    constexpr std::size_t n = 1u << 20;
    Resources r(0, n);
    // One large chunk leaves the entire H2D pending when copy_async returns.
    PeerLink link(0, 0, n);
    CHECK(link.path() == Path::Staged);
    for (int iteration = 0; iteration < 32; ++iteration) {
        VFA_CUDA_CHECK(cudaMemsetAsync(r.dst, 0, n, r.stream));
        VFA_CUDA_CHECK(cudaMemsetAsync(r.src, 0xAB, n, r.stream));
        link.copy_async(r.dst, r.src, n, r.stream);
        VFA_CUDA_CHECK(cudaMemcpyAsync(r.host, r.dst, n, cudaMemcpyDeviceToHost, r.stream));
        VFA_CUDA_CHECK(cudaStreamSynchronize(r.stream));
        CHECK(std::all_of(r.host, r.host + n, [](unsigned char c) { return c == 0xAB; }));
    }
}

void buffer_reuse() {
    devices();
    constexpr std::size_t n = 3 * 257 + 13, copies = 64, total = n * copies;
    Resources r(0, total);
    const auto data = random_bytes(total, 29);
    VFA_CUDA_CHECK(cudaMemcpy(r.src, data.data(), total, cudaMemcpyHostToDevice));
    // A pageable host-to-device cudaMemcpy may return before the DMA completes, and the copies below run on
    // non-blocking streams that do not order behind it.
    VFA_CUDA_CHECK(cudaDeviceSynchronize());
    VFA_CUDA_CHECK(cudaMemsetAsync(r.dst, 0, total, r.stream));
    PeerLink link(0, 0, 257);
    CHECK(link.path() == Path::Staged);
    for (std::size_t i = 0; i < copies; ++i)
        link.copy_async(static_cast<char*>(r.dst) + i * n,
                        static_cast<char*>(r.src) + i * n, n, r.stream);
    VFA_CUDA_CHECK(cudaMemcpyAsync(r.host, r.dst, total, cudaMemcpyDeviceToHost, r.stream));
    VFA_CUDA_CHECK(cudaStreamSynchronize(r.stream));
    CHECK(std::equal(data.begin(), data.end(), r.host));
}

void peer() {
    if (devices() < 2) throw Skip("not run: peer paths need at least two GPUs");
    int forward = 0, reverse = 0;
    VFA_CUDA_CHECK(cudaDeviceCanAccessPeer(&forward, 0, 1));
    VFA_CUDA_CHECK(cudaDeviceCanAccessPeer(&reverse, 1, 0));
    if (!forward || !reverse) throw Skip("not run: bidirectional peer capability unavailable");
    EnvGuard env_guard;
    for (bool staged : {false, true}) {
        if (staged) CHECK(setenv("VFA_DISABLE_P2P", "1", 1) == 0);
        else CHECK(unsetenv("VFA_DISABLE_P2P") == 0);
        for (int src : {0, 1}) {
            const int dst = 1 - src;
            constexpr std::size_t n = 3 * 4096 + 17;
            Resources source(src, n), destination(dst, n);
            const auto data = random_bytes(n, 41 + src);
            VFA_CUDA_CHECK(cudaSetDevice(src));
            VFA_CUDA_CHECK(cudaMemcpy(source.src, data.data(), n, cudaMemcpyHostToDevice));
            // A pageable host-to-device cudaMemcpy may return before the DMA completes, and the copies below run on
            // non-blocking streams that do not order behind it.
            VFA_CUDA_CHECK(cudaDeviceSynchronize());
            VFA_CUDA_CHECK(cudaSetDevice(dst));
            {
                PeerLink link(src, dst, 4096);
                CHECK(link.path() == (staged ? Path::Staged : Path::Peer));
                int current = -1;
                VFA_CUDA_CHECK(cudaGetDevice(&current));
                CHECK(current == dst);
                link.copy_async(destination.dst, source.src, n, source.stream);
                VFA_CUDA_CHECK(cudaGetDevice(&current));
                CHECK(current == dst);
            }
            int current = -1;
            VFA_CUDA_CHECK(cudaGetDevice(&current));
            CHECK(current == dst);
            VFA_CUDA_CHECK(cudaMemcpy(destination.host, destination.dst, n, cudaMemcpyDeviceToHost));
            CHECK(std::equal(data.begin(), data.end(), destination.host));
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    const std::vector<TestCase> cases{{"probe", probe}, {"env", env}, {"staged_sizes", staged_sizes},
        {"stream_order_before", stream_order_before}, {"buffer_reuse", buffer_reuse}, {"peer", peer}};
    if (argc > 2) { std::cerr << "usage: test_p2p [case]\n"; return 1; }
    bool ran = false, skipped = false;
    for (const auto& test : cases) {
        if (argc == 2 && test.first != std::string(argv[1])) continue;
        try { test.second(); std::cout << "PASS " << test.first << '\n'; ran = true; }
        catch (const Skip& e) { std::cout << "SKIP " << test.first << ": " << e.what() << '\n'; skipped = true; }
        catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    }
    if (!ran && !skipped) { std::cerr << "unknown case\n"; return 1; }
    return skipped ? 77 : 0;
}
