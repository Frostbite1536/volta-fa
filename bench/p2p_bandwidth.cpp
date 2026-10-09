#include "volta_fa/check.h"
#include "volta_fa/p2p.h"

#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {
using volta_fa::p2p::Path;
using volta_fa::p2p::PeerLink;

std::size_t parse_size(const std::string& text) {
    if (text.empty() || !std::isdigit(static_cast<unsigned char>(text[0])))
        throw std::invalid_argument("invalid size: " + text);
    std::size_t end = 0;
    const auto value = std::stoull(text, &end);
    std::string suffix = text.substr(end);
    for (auto& c : suffix) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    std::size_t multiplier = 1;
    if (suffix == "K" || suffix == "KB" || suffix == "KIB") multiplier = 1u << 10;
    else if (suffix == "M" || suffix == "MB" || suffix == "MIB") multiplier = 1u << 20;
    else if (suffix == "G" || suffix == "GB" || suffix == "GIB") multiplier = 1u << 30;
    else if (!suffix.empty() && suffix != "B") throw std::invalid_argument("invalid size: " + text);
    if (!value || value > std::numeric_limits<std::size_t>::max() / multiplier)
        throw std::invalid_argument("size must be nonzero and fit size_t: " + text);
    return static_cast<std::size_t>(value) * multiplier;
}

struct Options {
    std::vector<std::size_t> sizes{64u << 10, 1u << 20, 16u << 20, 256u << 20};
    std::size_t chunk = 4u << 20;
    int iters = 100;
    bool csv = false;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--csv") { o.csv = true; continue; }
        if (arg != "--sizes" && arg != "--iters" && arg != "--chunk")
            throw std::invalid_argument("unknown option: " + arg);
        if (++i == argc) throw std::invalid_argument("missing value for " + arg);
        const std::string value = argv[i];
        if (arg == "--sizes") {
            o.sizes.clear();
            std::istringstream values(value);
            std::string size;
            while (std::getline(values, size, ',')) o.sizes.push_back(parse_size(size));
            if (o.sizes.empty() || value.back() == ',') throw std::invalid_argument("empty size list entry");
        } else if (arg == "--chunk") o.chunk = parse_size(value);
        else {
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("iters must be a positive integer");
            std::size_t end = 0;
            const auto n = std::stoull(value, &end);
            if (!n || n > static_cast<unsigned>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("iters must fit a positive int");
            o.iters = static_cast<int>(n);
        }
    }
    return o;
}

struct EnvSetting {
    bool present;
    std::string previous;
    explicit EnvSetting(Path path) : present(std::getenv("VFA_DISABLE_P2P") != nullptr),
        previous(present ? std::getenv("VFA_DISABLE_P2P") : "") {
        const int result = path == Path::Staged ? setenv("VFA_DISABLE_P2P", "1", 1) : unsetenv("VFA_DISABLE_P2P");
        if (result != 0) throw std::runtime_error("cannot set VFA_DISABLE_P2P");
    }
    ~EnvSetting() {
        if (present) setenv("VFA_DISABLE_P2P", previous.c_str(), 1);
        else unsetenv("VFA_DISABLE_P2P");
    }
};

struct Buffers {
    int device, original = 0;
    void* src = nullptr;
    void* dst = nullptr;
    cudaStream_t stream{};
    Buffers(int dev, std::size_t bytes) : device(dev) {
        VFA_CUDA_CHECK(cudaGetDevice(&original));
        try {
            VFA_CUDA_CHECK(cudaSetDevice(device));
            VFA_CUDA_CHECK(cudaMalloc(&src, bytes));
            VFA_CUDA_CHECK(cudaMalloc(&dst, bytes));
            VFA_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            VFA_CUDA_CHECK(cudaMemsetAsync(src, 0xAB, bytes, stream));
            VFA_CUDA_CHECK(cudaMemsetAsync(dst, 0, bytes, stream));
            VFA_CUDA_CHECK(cudaStreamSynchronize(stream));
            VFA_CUDA_CHECK(cudaSetDevice(original));
        } catch (...) { release(); throw; }
    }
    ~Buffers() { release(); }
    void release() noexcept {
        auto checked = [](auto f) {
            try { f(); } catch (const std::exception& e) { std::cerr << e.what() << '\n'; }
        };
        checked([&] { VFA_CUDA_CHECK(cudaSetDevice(device)); });
        if (stream) {
            checked([&] { VFA_CUDA_CHECK(cudaStreamSynchronize(stream)); });
            checked([&] { VFA_CUDA_CHECK(cudaStreamDestroy(stream)); });
        }
        if (src) checked([&] { VFA_CUDA_CHECK(cudaFree(src)); });
        if (dst) checked([&] { VFA_CUDA_CHECK(cudaFree(dst)); });
        checked([&] { VFA_CUDA_CHECK(cudaSetDevice(original)); });
    }
    void synchronize() {
        int original_device = 0;
        VFA_CUDA_CHECK(cudaGetDevice(&original_device));
        VFA_CUDA_CHECK(cudaSetDevice(device));
        try { VFA_CUDA_CHECK(cudaStreamSynchronize(stream)); }
        catch (...) { VFA_CUDA_CHECK(cudaSetDevice(original_device)); throw; }
        VFA_CUDA_CHECK(cudaSetDevice(original_device));
    }
};

void measure(const Options& options, int src, int dst, Path requested,
             std::size_t bytes, bool bidirectional) {
    EnvSetting env(requested);
    Buffers a(src, bytes), b(dst, bytes);
    PeerLink forward(src, dst, options.chunk);
    std::unique_ptr<PeerLink> reverse;
    if (bidirectional) reverse = std::make_unique<PeerLink>(dst, src, options.chunk);
    if (forward.path() != requested || (reverse && reverse->path() != requested)) {
        std::cerr << "not run: Peer " << src << "->" << dst
                  << " (peer enable failed; staged fallback available)\n";
        return;
    }
    constexpr int warmup = 3;
    double seconds = 0;
    if (!reverse) {
        for (int i = 0; i < warmup; ++i) forward.copy_async(b.dst, a.src, bytes, a.stream);
        a.synchronize();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < options.iters; ++i) forward.copy_async(b.dst, a.src, bytes, a.stream);
        a.synchronize();
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    } else {
        // Separate enqueue threads keep both directions active even when a
        // staged link must block on reuse of one of its two pinned buffers.
        std::mutex mutex;
        std::condition_variable gate;
        int ready = 0;
        bool start_batch = false;
        std::exception_ptr errors[2];
        auto worker = [&](int direction) {
            auto& link = direction == 0 ? forward : *reverse;
            auto& source = direction == 0 ? a : b;
            auto& destination = direction == 0 ? b : a;
            auto batch = [&](int iters) {
                for (int i = 0; i < iters; ++i)
                    link.copy_async(destination.dst, source.src, bytes, source.stream);
                source.synchronize();
            };
            try { batch(warmup); }
            catch (...) { errors[direction] = std::current_exception(); }
            {
                std::unique_lock<std::mutex> lock(mutex);
                ++ready;
                gate.notify_all();
                gate.wait(lock, [&] { return start_batch; });
            }
            if (!errors[direction]) {
                try { batch(options.iters); }
                catch (...) { errors[direction] = std::current_exception(); }
            }
        };
        std::thread first(worker, 0);
        std::thread second;
        try { second = std::thread(worker, 1); }
        catch (...) {
            { std::lock_guard<std::mutex> lock(mutex); start_batch = true; }
            gate.notify_all();
            first.join();
            throw;
        }
        std::unique_lock<std::mutex> lock(mutex);
        gate.wait(lock, [&] { return ready == 2; });
        const auto start = std::chrono::steady_clock::now();
        start_batch = true;
        lock.unlock();
        gate.notify_all();
        first.join();
        second.join();
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
    }
    const double gbps = static_cast<double>(bytes) * options.iters * (reverse ? 2.0 : 1.0) / seconds / 1e9;
    // For bidirectional mode latency is per concurrent two-copy round.
    const double latency = seconds * 1e6 / options.iters;
    const char* mode = bidirectional ? "bidirectional" : "unidirectional";
    std::cout << std::fixed << std::setprecision(3);
    if (options.csv)
        std::cout << volta_fa::p2p::path_name(forward.path()) << ',' << src << ',' << dst << ',' << mode
                  << ',' << bytes << ',' << options.iters << ',' << gbps << ',' << latency << '\n';
    else
        std::cout << volta_fa::p2p::path_name(forward.path()) << ' ' << src << "->" << dst << ' ' << mode
                  << " bytes=" << bytes << " iters=" << options.iters << " GB/s=" << gbps
                  << " latency_us=" << latency << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        int count = 0;
        try { VFA_CUDA_CHECK(cudaGetDeviceCount(&count)); }
        catch (const std::exception& e) { std::cerr << "not run: " << e.what() << '\n'; return 77; }
        const auto topology = volta_fa::p2p::probe();
        (options.csv ? std::cerr : std::cout) << volta_fa::p2p::describe(topology);
        if (!count) { std::cerr << "not run: cudaGetDeviceCount returned 0\n"; return 77; }
        if (options.csv) std::cout << "path,src,dst,mode,bytes,iters,gbps,latency_us\n";
        else std::cout << "Sizes use binary units; GB/s is decimal. Bidirectional latency is per pair.\n";
        for (int src = 0; src < count; ++src) {
            for (int dst = 0; dst < count; ++dst) {
                const bool peer = src != dst && topology.peer[src][dst].can_access && topology.peer[dst][src].can_access;
                for (Path path : {Path::Peer, Path::Staged}) {
                    if (path == Path::Peer && !peer) continue;
                    for (auto size : options.sizes) {
                        measure(options, src, dst, path, size, false);
                        if (src != dst) measure(options, src, dst, path, size, true);
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\nusage: p2p_bandwidth [--sizes 64K,1M,16M,256M] [--iters N] [--chunk 4M] [--csv]\n";
        return 1;
    }
}
