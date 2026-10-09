#include "volta_fa/p2p.h"
#include "volta_fa/check.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

namespace volta_fa::p2p {
namespace {
class DeviceGuard {
public:
    DeviceGuard() { VFA_CUDA_CHECK(cudaGetDevice(&original_)); }
    void set(int device) { VFA_CUDA_CHECK(cudaSetDevice(device)); }
    void restore() { set(original_); restored_ = true; }
    ~DeviceGuard() {
        if (!restored_) {
            try { restore(); }
            catch (const std::exception& e) { std::cerr << e.what() << '\n'; }
        }
    }
private:
    int original_ = 0;
    bool restored_ = false;
};

bool enable_peer(int device, int peer, DeviceGuard& guard) {
    guard.set(device);
    cudaError_t status = cudaSuccess;
    VFA_CUDA_CHECK((status = cudaDeviceEnablePeerAccess(peer, 0),
                   status == cudaErrorPeerAccessAlreadyEnabled ||
                   status == cudaErrorPeerAccessUnsupported || status == cudaErrorTooManyPeers
                       ? cudaSuccess : status));
    if (status != cudaSuccess) {
        // The expected status is handled above; clear the runtime's last error.
        VFA_CUDA_CHECK((cudaGetLastError(), cudaSuccess));
    }
    return status == cudaSuccess || status == cudaErrorPeerAccessAlreadyEnabled;
}

template<class F> void cleanup(F&& f) noexcept {
    try { f(); }
    catch (const std::exception& e) { std::cerr << "P2P cleanup: " << e.what() << '\n'; }
}
}  // namespace

bool p2p_disabled_by_env() {
    const char* value = std::getenv("VFA_DISABLE_P2P");
    return value && *value && std::strcmp(value, "0") != 0;
}

const char* path_name(Path p) { return p == Path::Peer ? "Peer" : "Staged"; }

Topology probe() {
    Topology t;
    cudaError_t status = cudaSuccess;
    VFA_CUDA_CHECK((status = cudaGetDeviceCount(&t.device_count),
                   status == cudaErrorNoDevice ? cudaSuccess : status));
    if (status == cudaErrorNoDevice) {
        VFA_CUDA_CHECK((cudaGetLastError(), cudaSuccess));
        t.device_count = 0;
    }
    if (t.device_count == 0) return t;
    DeviceGuard guard;
    t.peer.resize(t.device_count, std::vector<PeerInfo>(t.device_count));
    for (int src = 0; src < t.device_count; ++src) {
        cudaDeviceProp prop{};
        VFA_CUDA_CHECK(cudaGetDeviceProperties(&prop, src));
        char bus_id[32]{};
        VFA_CUDA_CHECK(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), src));
        t.name.emplace_back(prop.name);
        t.pci_bus_id.emplace_back(bus_id);
        for (int dst = 0; dst < t.device_count; ++dst) {
            if (src == dst) continue;
            auto& info = t.peer[src][dst];
            int access = 0, atomics = 0;
            VFA_CUDA_CHECK(cudaDeviceCanAccessPeer(&access, src, dst));
            info.can_access = access != 0;
            VFA_CUDA_CHECK(cudaDeviceGetP2PAttribute(&info.perf_rank, cudaDevP2PAttrPerformanceRank, src, dst));
            VFA_CUDA_CHECK(cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, src, dst));
            info.native_atomics = atomics != 0;
        }
    }
    guard.restore();
    return t;
}

std::string describe(const Topology& t) {
    std::ostringstream out;
    out << "Devices: " << t.device_count << '\n';
    for (int i = 0; i < t.device_count; ++i)
        out << "  " << i << ": " << t.name.at(i) << "  PCI " << t.pci_bus_id.at(i) << '\n';
    out << "Peer matrix (row=src, column=dst; access/rank/native-atomics):\n  src\\dst";
    for (int dst = 0; dst < t.device_count; ++dst) out << '\t' << dst;
    out << '\n';
    for (int src = 0; src < t.device_count; ++src) {
        out << "  " << src;
        for (int dst = 0; dst < t.device_count; ++dst) {
            const auto& p = t.peer.at(src).at(dst);
            if (src == dst) out << "\t-";
            else out << '\t' << (p.can_access ? "yes" : "no") << '/' << p.perf_rank
                     << '/' << (p.native_atomics ? "yes" : "no");
        }
        out << '\n';
    }
    return out.str();
}

struct PeerLink::Impl {
    int src, dst;
    std::size_t chunk;
    Path active = Path::Staged;
    void* buffer[2]{};
    cudaEvent_t ready[2]{}, free[2]{}, complete{};
    bool ready_recorded[2]{}, free_recorded[2]{}, complete_recorded = false;
    cudaStream_t dst_stream{};

    Impl(int s, int d, std::size_t c) : src(s), dst(d), chunk(c) {}
    ~Impl() {
        cleanup([&] {
            DeviceGuard guard;
            // Each event covers this link's own work, including partial enqueues.
            guard.set(src);
            for (int i = 0; i < 2; ++i)
                if (ready_recorded[i]) cleanup([&] { VFA_CUDA_CHECK(cudaEventSynchronize(ready[i])); });
            guard.set(dst);
            for (int i = 0; i < 2; ++i)
                if (free_recorded[i]) cleanup([&] { VFA_CUDA_CHECK(cudaEventSynchronize(free[i])); });
            guard.set(active == Path::Peer ? src : dst);
            if (complete_recorded) cleanup([&] { VFA_CUDA_CHECK(cudaEventSynchronize(complete)); });
            if (complete) cleanup([&] { VFA_CUDA_CHECK(cudaEventDestroy(complete)); });
            guard.set(src);
            for (auto event : ready)
                if (event) cleanup([&] { VFA_CUDA_CHECK(cudaEventDestroy(event)); });
            guard.set(dst);
            for (auto event : free)
                if (event) cleanup([&] { VFA_CUDA_CHECK(cudaEventDestroy(event)); });
            if (dst_stream) cleanup([&] { VFA_CUDA_CHECK(cudaStreamDestroy(dst_stream)); });
            for (auto ptr : buffer)
                if (ptr) cleanup([&] { VFA_CUDA_CHECK(cudaFreeHost(ptr)); });
            // Leave peer access enabled: other code may rely on it.
            guard.restore();
        });
    }
};

PeerLink::PeerLink(int src_dev, int dst_dev, std::size_t chunk_bytes) {
    if (chunk_bytes == 0) throw std::invalid_argument("PeerLink chunk_bytes must be nonzero");
    DeviceGuard guard;
    // Validate both ordinals even for a forced staged or same-device link.
    guard.set(src_dev);
    guard.set(dst_dev);
    impl_ = std::make_unique<Impl>(src_dev, dst_dev, chunk_bytes);
    auto& p = *impl_;
    if (src_dev != dst_dev && !p2p_disabled_by_env()) {
        int forward = 0, reverse = 0;
        VFA_CUDA_CHECK(cudaDeviceCanAccessPeer(&forward, dst_dev, src_dev));
        VFA_CUDA_CHECK(cudaDeviceCanAccessPeer(&reverse, src_dev, dst_dev));
        if (forward && reverse && enable_peer(dst_dev, src_dev, guard) &&
            enable_peer(src_dev, dst_dev, guard)) p.active = Path::Peer;
    }
    if (p.active == Path::Staged) {
        guard.set(dst_dev);
        VFA_CUDA_CHECK(cudaStreamCreateWithFlags(&p.dst_stream, cudaStreamNonBlocking));
        for (int i = 0; i < 2; ++i) {
            VFA_CUDA_CHECK(cudaHostAlloc(&p.buffer[i], chunk_bytes, cudaHostAllocPortable));
            VFA_CUDA_CHECK(cudaEventCreateWithFlags(&p.free[i], cudaEventDisableTiming));
            guard.set(src_dev);
            VFA_CUDA_CHECK(cudaEventCreateWithFlags(&p.ready[i], cudaEventDisableTiming));
            guard.set(dst_dev);
        }
    } else guard.set(src_dev);
    VFA_CUDA_CHECK(cudaEventCreateWithFlags(&p.complete, cudaEventDisableTiming));
    guard.restore();
    std::clog << "PeerLink " << src_dev << "->" << dst_dev
              << " path=" << path_name(p.active) << '\n';
}

PeerLink::~PeerLink() = default;
Path PeerLink::path() const { return impl_->active; }

void PeerLink::copy_async(void* dst, const void* src, std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0) return;
    DeviceGuard guard;
    auto& p = *impl_;
    guard.set(p.src);
    if (p.active == Path::Peer) {
        // Serialize successive uses of this link even across different streams,
        // so the last completion event also covers earlier outstanding copies.
        if (p.complete_recorded) VFA_CUDA_CHECK(cudaStreamWaitEvent(stream, p.complete, 0));
        VFA_CUDA_CHECK(cudaMemcpyPeerAsync(dst, p.dst, src, p.src, bytes, stream));
        VFA_CUDA_CHECK(cudaEventRecord(p.complete, stream));
        p.complete_recorded = true;
    } else {
        std::size_t offset = 0;
        unsigned slot = 0;
        while (offset < bytes) {
            const unsigned i = slot++ % 2;
            const std::size_t count = std::min(p.chunk, bytes - offset);
            // Do not reuse pinned storage until the previous H2D has finished.
            if (p.free_recorded[i]) VFA_CUDA_CHECK(cudaEventSynchronize(p.free[i]));
            guard.set(p.src);
            VFA_CUDA_CHECK(cudaMemcpyAsync(p.buffer[i], static_cast<const char*>(src) + offset,
                                          count, cudaMemcpyDeviceToHost, stream));
            VFA_CUDA_CHECK(cudaEventRecord(p.ready[i], stream));
            p.ready_recorded[i] = true;
            guard.set(p.dst);
            VFA_CUDA_CHECK(cudaStreamWaitEvent(p.dst_stream, p.ready[i], 0));
            VFA_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dst) + offset, p.buffer[i],
                                          count, cudaMemcpyHostToDevice, p.dst_stream));
            VFA_CUDA_CHECK(cudaEventRecord(p.free[i], p.dst_stream));
            p.free_recorded[i] = true;
            offset += count;
        }
        VFA_CUDA_CHECK(cudaEventRecord(p.complete, p.dst_stream));
        p.complete_recorded = true;
        guard.set(p.src);
        VFA_CUDA_CHECK(cudaStreamWaitEvent(stream, p.complete, 0));
    }
    guard.restore();
}
}  // namespace volta_fa::p2p
