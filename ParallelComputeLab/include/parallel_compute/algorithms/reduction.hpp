#pragma once

#include "parallel_compute/common/traits.hpp"
#include <cstddef>
#include <optional>

namespace parallel_compute {

// --- Scalar Reference Reductions ---

template <Numeric T>
[[nodiscard]] T reduce_sum_scalar(const T* data, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<T> reduce_min_scalar(const T* data, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<T> reduce_max_scalar(const T* data, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] T dot_product_scalar(const T* a, const T* b, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<double> mean_scalar(const T* data, std::size_t n) noexcept;

// --- Compiler-Optimized Reductions (multiple accumulators to break dependency chains) ---

template <Numeric T>
[[nodiscard]] T reduce_sum_optimized(const T* __restrict__ data, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<T> reduce_min_optimized(const T* __restrict__ data, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<T> reduce_max_optimized(const T* __restrict__ data, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] T dot_product_optimized(const T* __restrict__ a, const T* __restrict__ b, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<double> mean_optimized(const T* __restrict__ data, std::size_t n) noexcept;

} // namespace parallel_compute
