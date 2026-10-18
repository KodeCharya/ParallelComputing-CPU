#pragma once

#include "parallel_compute/common/traits.hpp"
#include <cstddef>
#include <optional>

namespace parallel_compute {

template <Numeric T>
[[nodiscard]] T reduce_sum_simd(const T* data, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<T> reduce_min_simd(const T* data, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<T> reduce_max_simd(const T* data, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] T dot_product_simd(const T* a, const T* b, std::size_t n) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<double> mean_simd(const T* data, std::size_t n) noexcept;

} // namespace parallel_compute
