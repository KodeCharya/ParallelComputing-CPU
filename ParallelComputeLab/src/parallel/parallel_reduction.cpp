#include "parallel_compute/parallel/parallel_reduction.hpp"
#include "parallel_compute/algorithms/reduction.hpp"

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
#include <omp.h>
#endif

namespace parallel_compute {

template <Numeric T>
T reduce_sum_parallel(const T* data, std::size_t n, std::size_t num_threads) noexcept {
    if (n == 0) return T{0};

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    T sum = T{0};
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:sum)
    for (std::size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
#else
    (void)num_threads;
    return reduce_sum_optimized(data, n);
#endif
}

template <Numeric T>
std::optional<T> reduce_min_parallel(const T* data, std::size_t n, std::size_t num_threads) noexcept {
    if (n == 0) return std::nullopt;

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    T min_val = data[0];
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(min:min_val)
    for (std::size_t i = 0; i < n; ++i) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }
    return min_val;
#else
    (void)num_threads;
    return reduce_min_optimized(data, n);
#endif
}

template <Numeric T>
std::optional<T> reduce_max_parallel(const T* data, std::size_t n, std::size_t num_threads) noexcept {
    if (n == 0) return std::nullopt;

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    T max_val = data[0];
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(max:max_val)
    for (std::size_t i = 0; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
#else
    (void)num_threads;
    return reduce_max_optimized(data, n);
#endif
}

template <Numeric T>
T dot_product_parallel(const T* a, const T* b, std::size_t n, std::size_t num_threads) noexcept {
    if (n == 0) return T{0};

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    T sum = T{0};
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:sum)
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    (void)num_threads;
    return dot_product_optimized(a, b, n);
#endif
}

template <Numeric T>
std::optional<double> mean_parallel(const T* data, std::size_t n, std::size_t num_threads) noexcept {
    if (n == 0) return std::nullopt;

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    double sum = 0.0;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:sum)
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(data[i]);
    }
    return sum / static_cast<double>(n);
#else
    (void)num_threads;
    return mean_optimized(data, n);
#endif
}

// Explicit template instantiations
#define INSTANTIATE_PARALLEL_REDUCTIONS(Type) \
    template Type reduce_sum_parallel<Type>(const Type*, std::size_t, std::size_t) noexcept; \
    template std::optional<Type> reduce_min_parallel<Type>(const Type*, std::size_t, std::size_t) noexcept; \
    template std::optional<Type> reduce_max_parallel<Type>(const Type*, std::size_t, std::size_t) noexcept; \
    template Type dot_product_parallel<Type>(const Type*, const Type*, std::size_t, std::size_t) noexcept; \
    template std::optional<double> mean_parallel<Type>(const Type*, std::size_t, std::size_t) noexcept;

INSTANTIATE_PARALLEL_REDUCTIONS(float)
INSTANTIATE_PARALLEL_REDUCTIONS(double)
INSTANTIATE_PARALLEL_REDUCTIONS(int32_t)
INSTANTIATE_PARALLEL_REDUCTIONS(int64_t)

} // namespace parallel_compute
