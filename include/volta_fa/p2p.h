#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace volta_fa::p2p {
struct PeerInfo { bool can_access = false; int perf_rank = -1; bool native_atomics = false; };
struct Topology {
    int device_count = 0;
    std::vector<std::string> name, pci_bus_id;
    std::vector<std::vector<PeerInfo>> peer;
};
Topology probe();
std::string describe(const Topology& t);

enum class Path { Peer, Staged };
const char* path_name(Path p);
bool p2p_disabled_by_env();

class PeerLink {
public:
    PeerLink(int src_dev, int dst_dev, std::size_t chunk_bytes = 4u << 20);
    ~PeerLink();
    PeerLink(const PeerLink&) = delete;
    PeerLink& operator=(const PeerLink&) = delete;
    Path path() const;
    // stream belongs to src_dev. Prior stream work precedes the copy, and later
    // stream work waits until the destination has received every byte.
    // Host waits occur only when a pinned staging buffer is still in use.
    void copy_async(void* dst, const void* src, std::size_t bytes, cudaStream_t stream);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace volta_fa::p2p
