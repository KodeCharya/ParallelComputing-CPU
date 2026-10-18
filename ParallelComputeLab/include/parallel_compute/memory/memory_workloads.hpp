#pragma once

#include "parallel_compute/common/traits.hpp"
#include <cstddef>

namespace parallel_compute {

template <Numeric T>
void array_copy_serial(T* dst, const T* src, std::size_t n) noexcept;

template <Numeric T>
void array_scale_serial(T* dst, const T* src, T alpha, std::size_t n) noexcept;

template <Numeric T>
void array_triad_serial(T* a, const T* b, const T* c, T alpha, std::size_t n) noexcept;

template <Numeric T>
T sequential_read_serial(const T* src, std::size_t n) noexcept;

template <Numeric T>
T strided_read_serial(const T* src, std::size_t n, std::size_t stride) noexcept;

// OpenMP parallel implementations
template <Numeric T>
void array_copy_parallel(T* dst, const T* src, std::size_t n, std::size_t num_threads = 0) noexcept;

template <Numeric T>
void array_scale_parallel(T* dst, const T* src, T alpha, std::size_t n, std::size_t num_threads = 0) noexcept;

template <Numeric T>
void array_triad_parallel(T* a, const T* b, const T* c, T alpha, std::size_t n, std::size_t num_threads = 0) noexcept;

} // namespace parallel_compute
