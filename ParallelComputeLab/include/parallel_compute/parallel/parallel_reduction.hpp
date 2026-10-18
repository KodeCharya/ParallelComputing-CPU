#pragma once

#include "parallel_compute/common/traits.hpp"
#include <cstddef>
#include <optional>

namespace parallel_compute {

template <Numeric T>
[[nodiscard]] T reduce_sum_parallel(const T* data, std::size_t n, std::size_t num_threads = 0) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<T> reduce_min_parallel(const T* data, std::size_t n, std::size_t num_threads = 0) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<T> reduce_max_parallel(const T* data, std::size_t n, std::size_t num_threads = 0) noexcept;

template <Numeric T>
[[nodiscard]] T dot_product_parallel(const T* a, const T* b, std::size_t n, std::size_t num_threads = 0) noexcept;

template <Numeric T>
[[nodiscard]] std::optional<double> mean_parallel(const T* data, std::size_t n, std::size_t num_threads = 0) noexcept;

} // namespace parallel_compute
