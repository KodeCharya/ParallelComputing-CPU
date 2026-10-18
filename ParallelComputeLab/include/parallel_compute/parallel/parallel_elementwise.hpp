#pragma once

#include "parallel_compute/common/traits.hpp"
#include <cstddef>

namespace parallel_compute {

template <Numeric T>
void vector_add_parallel(const T* a, const T* b, T* c, std::size_t n, std::size_t num_threads = 0) noexcept;

template <Numeric T>
void vector_mul_parallel(const T* a, const T* b, T* c, std::size_t n, std::size_t num_threads = 0) noexcept;

template <Numeric T>
void scalar_mul_parallel(const T* a, T alpha, T* c, std::size_t n, std::size_t num_threads = 0) noexcept;

template <Numeric T>
void saxpy_parallel(const T* x, T* y, T alpha, std::size_t n, std::size_t num_threads = 0) noexcept;

template <Numeric T>
void vector_scale_parallel(T* x, T alpha, std::size_t n, std::size_t num_threads = 0) noexcept;

} // namespace parallel_compute
