#pragma once

#include <cstddef>

namespace parallel_compute {

/**
 * Standard byte-counting convention for memory benchmarks:
 * - Copy: y = x. Reads x, writes y -> 2 * N * sizeof(T) bytes.
 * - Scale: y = alpha * x. Reads x, writes y -> 2 * N * sizeof(T) bytes.
 * - Triad: a = b + alpha * c. Reads b and c, writes a -> 3 * N * sizeof(T) bytes.
 * - Add: a = b + c. Reads b and c, writes a -> 3 * N * sizeof(T) bytes.
 * - Sequential/Strided read: Reads element -> N * sizeof(T) bytes accessed.
 */
class BandwidthCounter {
public:
    template <typename T>
    [[nodiscard]] static constexpr std::size_t copy_bytes(std::size_t n) noexcept {
        return 2 * n * sizeof(T);
    }

    template <typename T>
    [[nodiscard]] static constexpr std::size_t scale_bytes(std::size_t n) noexcept {
        return 2 * n * sizeof(T);
    }

    template <typename T>
    [[nodiscard]] static constexpr std::size_t triad_bytes(std::size_t n) noexcept {
        return 3 * n * sizeof(T);
    }

    template <typename T>
    [[nodiscard]] static constexpr std::size_t add_bytes(std::size_t n) noexcept {
        return 3 * n * sizeof(T);
    }

    template <typename T>
    [[nodiscard]] static constexpr std::size_t read_bytes(std::size_t n) noexcept {
        return n * sizeof(T);
    }

    [[nodiscard]] static constexpr double gigabytes_per_second(std::size_t bytes, double elapsed_seconds) noexcept {
        if (elapsed_seconds <= 0.0) return 0.0;
        return (static_cast<double>(bytes) / elapsed_seconds) / 1e9;
    }
};

} // namespace parallel_compute
