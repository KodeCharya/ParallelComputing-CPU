#include "parallel_compute/algorithms/reduction.hpp"

#include <algorithm>
#include <cmath>

namespace parallel_compute {

// --- Scalar Reference Reductions ---

template <Numeric T>
T reduce_sum_scalar(const T* data, std::size_t n) noexcept {
    T sum = T{0};
    for (std::size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

template <Numeric T>
std::optional<T> reduce_min_scalar(const T* data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;
    T min_val = data[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }
    return min_val;
}

template <Numeric T>
std::optional<T> reduce_max_scalar(const T* data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;
    T max_val = data[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

template <Numeric T>
T dot_product_scalar(const T* a, const T* b, std::size_t n) noexcept {
    T sum = T{0};
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

template <Numeric T>
std::optional<double> mean_scalar(const T* data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(data[i]);
    }
    return sum / static_cast<double>(n);
}

// --- Compiler-Optimized Reductions ---

template <Numeric T>
T reduce_sum_optimized(const T* __restrict__ data, std::size_t n) noexcept {
    if (n == 0) return T{0};

    T acc0 = T{0};
    T acc1 = T{0};
    T acc2 = T{0};
    T acc3 = T{0};

    std::size_t i = 0;
    constexpr std::size_t kUnroll = 4;
    const std::size_t limit = (n >= kUnroll) ? (n - kUnroll + 1) : 0;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (; i < limit; i += kUnroll) {
        acc0 += data[i + 0];
        acc1 += data[i + 1];
        acc2 += data[i + 2];
        acc3 += data[i + 3];
    }

    T total = (acc0 + acc1) + (acc2 + acc3);
    for (; i < n; ++i) {
        total += data[i];
    }
    return total;
}

template <Numeric T>
std::optional<T> reduce_min_optimized(const T* __restrict__ data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;
    if (n < 4) return reduce_min_scalar(data, n);

    T m0 = data[0];
    T m1 = data[1];
    T m2 = data[2];
    T m3 = data[3];

    std::size_t i = 4;
    constexpr std::size_t kUnroll = 4;
    const std::size_t limit = (n >= kUnroll) ? (n - kUnroll + 1) : 0;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (; i < limit; i += kUnroll) {
        m0 = (data[i + 0] < m0) ? data[i + 0] : m0;
        m1 = (data[i + 1] < m1) ? data[i + 1] : m1;
        m2 = (data[i + 2] < m2) ? data[i + 2] : m2;
        m3 = (data[i + 3] < m3) ? data[i + 3] : m3;
    }

    T min_val = std::min({m0, m1, m2, m3});
    for (; i < n; ++i) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }
    return min_val;
}

template <Numeric T>
std::optional<T> reduce_max_optimized(const T* __restrict__ data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;
    if (n < 4) return reduce_max_scalar(data, n);

    T m0 = data[0];
    T m1 = data[1];
    T m2 = data[2];
    T m3 = data[3];

    std::size_t i = 4;
    constexpr std::size_t kUnroll = 4;
    const std::size_t limit = (n >= kUnroll) ? (n - kUnroll + 1) : 0;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (; i < limit; i += kUnroll) {
        m0 = (data[i + 0] > m0) ? data[i + 0] : m0;
        m1 = (data[i + 1] > m1) ? data[i + 1] : m1;
        m2 = (data[i + 2] > m2) ? data[i + 2] : m2;
        m3 = (data[i + 3] > m3) ? data[i + 3] : m3;
    }

    T max_val = std::max({m0, m1, m2, m3});
    for (; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

template <Numeric T>
T dot_product_optimized(const T* __restrict__ a, const T* __restrict__ b, std::size_t n) noexcept {
    if (n == 0) return T{0};

    T acc0 = T{0};
    T acc1 = T{0};
    T acc2 = T{0};
    T acc3 = T{0};

    std::size_t i = 0;
    constexpr std::size_t kUnroll = 4;
    const std::size_t limit = (n >= kUnroll) ? (n - kUnroll + 1) : 0;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (; i < limit; i += kUnroll) {
        acc0 += a[i + 0] * b[i + 0];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
    }

    T total = (acc0 + acc1) + (acc2 + acc3);
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

template <Numeric T>
std::optional<double> mean_optimized(const T* __restrict__ data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;

    double acc0 = 0.0;
    double acc1 = 0.0;
    double acc2 = 0.0;
    double acc3 = 0.0;

    std::size_t i = 0;
    constexpr std::size_t kUnroll = 4;
    const std::size_t limit = (n >= kUnroll) ? (n - kUnroll + 1) : 0;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (; i < limit; i += kUnroll) {
        acc0 += static_cast<double>(data[i + 0]);
        acc1 += static_cast<double>(data[i + 1]);
        acc2 += static_cast<double>(data[i + 2]);
        acc3 += static_cast<double>(data[i + 3]);
    }

    double total = (acc0 + acc1) + (acc2 + acc3);
    for (; i < n; ++i) {
        total += static_cast<double>(data[i]);
    }
    return total / static_cast<double>(n);
}

// Explicit template instantiations
#define INSTANTIATE_REDUCTIONS(Type) \
    template Type reduce_sum_scalar<Type>(const Type*, std::size_t) noexcept; \
    template std::optional<Type> reduce_min_scalar<Type>(const Type*, std::size_t) noexcept; \
    template std::optional<Type> reduce_max_scalar<Type>(const Type*, std::size_t) noexcept; \
    template Type dot_product_scalar<Type>(const Type*, const Type*, std::size_t) noexcept; \
    template std::optional<double> mean_scalar<Type>(const Type*, std::size_t) noexcept; \
    template Type reduce_sum_optimized<Type>(const Type*, std::size_t) noexcept; \
    template std::optional<Type> reduce_min_optimized<Type>(const Type*, std::size_t) noexcept; \
    template std::optional<Type> reduce_max_optimized<Type>(const Type*, std::size_t) noexcept; \
    template Type dot_product_optimized<Type>(const Type*, const Type*, std::size_t) noexcept; \
    template std::optional<double> mean_optimized<Type>(const Type*, std::size_t) noexcept;

INSTANTIATE_REDUCTIONS(float)
INSTANTIATE_REDUCTIONS(double)
INSTANTIATE_REDUCTIONS(int32_t)
INSTANTIATE_REDUCTIONS(int64_t)

} // namespace parallel_compute
