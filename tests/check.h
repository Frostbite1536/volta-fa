#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

inline void check(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": CHECK(" + expression + ") failed");
    }
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)

// Centralize test thresholds (INV-NUM-003). The FP32 model threshold is
// mandated by this ticket; analytical checks allow only float conversion error.
// No tolerance is prescribed for the FP16 error-budget configurations.
inline constexpr double fp32_abs_tolerance = 1e-5;
inline constexpr double analytic_abs_tolerance = 1e-6;
inline constexpr double analytic_tight_abs_tolerance = 1e-7;

inline bool near(double actual, double expected, double tolerance) {
    return std::isfinite(actual) && std::isfinite(expected) && std::abs(actual - expected) <= tolerance;
}

inline std::uint32_t float_bits(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

using TestCase = std::pair<const char*, void (*)()>;
inline int run_tests(int argc, char** argv, std::initializer_list<TestCase> cases) {
    try {
        bool ran = false;
        for (const auto& test : cases) {
            if (argc == 1 || test.first == std::string(argv[1])) {
                test.second();
                std::cout << "PASS " << test.first << '\n';
                ran = true;
            }
        }
        CHECK(ran);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
