#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace volta_fa::detail {
inline void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " +
                                 expression + ": " + cudaGetErrorString(status));
    }
}
}  // namespace volta_fa::detail

#define VFA_CUDA_CHECK(expr) \
    ::volta_fa::detail::cuda_check((expr), #expr, __FILE__, __LINE__)
