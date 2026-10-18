#pragma once

#include "parallel_compute/common/traits.hpp"
#include <cstddef>

namespace parallel_compute {

// --- 1. Scalar Reference Implementations ---
template <Numeric T>
void vector_add_scalar(const T* a, const T* b, T* c, std::size_t n) noexcept;

template <Numeric T>
void vector_mul_scalar(const T* a, const T* b, T* c, std::size_t n) noexcept;

template <Numeric T>
void scalar_mul_scalar(const T* a, T alpha, T* c, std::size_t n) noexcept;

template <Numeric T>
void saxpy_scalar(const T* x, T* y, T alpha, std::size_t n) noexcept;

template <Numeric T>
void vector_scale_scalar(T* x, T alpha, std::size_t n) noexcept;

// --- 2. Compiler-Optimized Implementations (unrolled, restrict, ivdep) ---
template <Numeric T>
void vector_add_optimized(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, std::size_t n) noexcept;

template <Numeric T>
void vector_mul_optimized(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, std::size_t n) noexcept;

template <Numeric T>
void scalar_mul_optimized(const T* __restrict__ a, T alpha, T* __restrict__ c, std::size_t n) noexcept;

template <Numeric T>
void saxpy_optimized(const T* __restrict__ x, T* __restrict__ y, T alpha, std::size_t n) noexcept;

template <Numeric T>
void vector_scale_optimized(T* __restrict__ x, T alpha, std::size_t n) noexcept;

} // namespace parallel_compute
