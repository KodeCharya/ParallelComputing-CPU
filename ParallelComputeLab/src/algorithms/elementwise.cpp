#include "parallel_compute/algorithms/elementwise.hpp"

namespace parallel_compute {

// --- 1. Scalar Reference Implementations ---

template <Numeric T>
void vector_add_scalar(const T* a, const T* b, T* c, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

template <Numeric T>
void vector_mul_scalar(const T* a, const T* b, T* c, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

template <Numeric T>
void scalar_mul_scalar(const T* a, T alpha, T* c, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = alpha * a[i];
    }
}

template <Numeric T>
void saxpy_scalar(const T* x, T* y, T alpha, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

template <Numeric T>
void vector_scale_scalar(T* x, T alpha, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = alpha * x[i];
    }
}

// --- 2. Compiler-Optimized Implementations ---

template <Numeric T>
void vector_add_optimized(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kUnroll = 4;
    const std::size_t limit = (n >= kUnroll) ? (n - kUnroll + 1) : 0;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (; i < limit; i += kUnroll) {
        c[i + 0] = a[i + 0] + b[i + 0];
        c[i + 1] = a[i + 1] + b[i + 1];
        c[i + 2] = a[i + 2] + b[i + 2];
        c[i + 3] = a[i + 3] + b[i + 3];
    }
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

template <Numeric T>
void vector_mul_optimized(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kUnroll = 4;
    const std::size_t limit = (n >= kUnroll) ? (n - kUnroll + 1) : 0;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (; i < limit; i += kUnroll) {
        c[i + 0] = a[i + 0] * b[i + 0];
        c[i + 1] = a[i + 1] * b[i + 1];
        c[i + 2] = a[i + 2] * b[i + 2];
        c[i + 3] = a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

template <Numeric T>
void scalar_mul_optimized(const T* __restrict__ a, T alpha, T* __restrict__ c, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kUnroll = 4;
    const std::size_t limit = (n >= kUnroll) ? (n - kUnroll + 1) : 0;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (; i < limit; i += kUnroll) {
        c[i + 0] = alpha * a[i + 0];
        c[i + 1] = alpha * a[i + 1];
        c[i + 2] = alpha * a[i + 2];
        c[i + 3] = alpha * a[i + 3];
    }
    for (; i < n; ++i) {
        c[i] = alpha * a[i];
    }
}

template <Numeric T>
void saxpy_optimized(const T* __restrict__ x, T* __restrict__ y, T alpha, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kUnroll = 4;
    const std::size_t limit = (n >= kUnroll) ? (n - kUnroll + 1) : 0;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (; i < limit; i += kUnroll) {
        y[i + 0] = alpha * x[i + 0] + y[i + 0];
        y[i + 1] = alpha * x[i + 1] + y[i + 1];
        y[i + 2] = alpha * x[i + 2] + y[i + 2];
        y[i + 3] = alpha * x[i + 3] + y[i + 3];
    }
    for (; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

template <Numeric T>
void vector_scale_optimized(T* __restrict__ x, T alpha, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kUnroll = 4;
    const std::size_t limit = (n >= kUnroll) ? (n - kUnroll + 1) : 0;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (; i < limit; i += kUnroll) {
        x[i + 0] = alpha * x[i + 0];
        x[i + 1] = alpha * x[i + 1];
        x[i + 2] = alpha * x[i + 2];
        x[i + 3] = alpha * x[i + 3];
    }
    for (; i < n; ++i) {
        x[i] = alpha * x[i];
    }
}

// Explicit template instantiations
#define INSTANTIATE_ELEMENTWISE(Type) \
    template void vector_add_scalar<Type>(const Type*, const Type*, Type*, std::size_t) noexcept; \
    template void vector_mul_scalar<Type>(const Type*, const Type*, Type*, std::size_t) noexcept; \
    template void scalar_mul_scalar<Type>(const Type*, Type, Type*, std::size_t) noexcept; \
    template void saxpy_scalar<Type>(const Type*, Type*, Type, std::size_t) noexcept; \
    template void vector_scale_scalar<Type>(Type*, Type, std::size_t) noexcept; \
    template void vector_add_optimized<Type>(const Type*, const Type*, Type*, std::size_t) noexcept; \
    template void vector_mul_optimized<Type>(const Type*, const Type*, Type*, std::size_t) noexcept; \
    template void scalar_mul_optimized<Type>(const Type*, Type, Type*, std::size_t) noexcept; \
    template void saxpy_optimized<Type>(const Type*, Type*, Type, std::size_t) noexcept; \
    template void vector_scale_optimized<Type>(Type*, Type, std::size_t) noexcept;

INSTANTIATE_ELEMENTWISE(float)
INSTANTIATE_ELEMENTWISE(double)
INSTANTIATE_ELEMENTWISE(int32_t)
INSTANTIATE_ELEMENTWISE(int64_t)

} // namespace parallel_compute
