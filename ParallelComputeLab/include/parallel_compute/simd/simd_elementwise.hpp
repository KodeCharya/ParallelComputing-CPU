#pragma once

#include "parallel_compute/common/traits.hpp"
#include <cstddef>

namespace parallel_compute {

template <Numeric T>
void vector_add_simd(const T* a, const T* b, T* c, std::size_t n) noexcept;

template <Numeric T>
void vector_mul_simd(const T* a, const T* b, T* c, std::size_t n) noexcept;

template <Numeric T>
void saxpy_simd(const T* x, T* y, T alpha, std::size_t n) noexcept;

template <Numeric T>
void vector_scale_simd(T* x, T alpha, std::size_t n) noexcept;

} // namespace parallel_compute
