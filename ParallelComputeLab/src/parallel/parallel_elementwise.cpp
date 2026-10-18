#include "parallel_compute/parallel/parallel_elementwise.hpp"
#include "parallel_compute/algorithms/elementwise.hpp"

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
#include <omp.h>
#endif

namespace parallel_compute {

template <Numeric T>
void vector_add_parallel(const T* a, const T* b, T* c, std::size_t n, std::size_t num_threads) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
#else
    (void)num_threads;
    vector_add_optimized(a, b, c, n);
#endif
}

template <Numeric T>
void vector_mul_parallel(const T* a, const T* b, T* c, std::size_t n, std::size_t num_threads) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
#else
    (void)num_threads;
    vector_mul_optimized(a, b, c, n);
#endif
}

template <Numeric T>
void scalar_mul_parallel(const T* a, T alpha, T* c, std::size_t n, std::size_t num_threads) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = alpha * a[i];
    }
#else
    (void)num_threads;
    scalar_mul_optimized(a, alpha, c, n);
#endif
}

template <Numeric T>
void saxpy_parallel(const T* x, T* y, T alpha, std::size_t n, std::size_t num_threads) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
#else
    (void)num_threads;
    saxpy_optimized(x, y, alpha, n);
#endif
}

template <Numeric T>
void vector_scale_parallel(T* x, T alpha, std::size_t n, std::size_t num_threads) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = alpha * x[i];
    }
#else
    (void)num_threads;
    vector_scale_optimized(x, alpha, n);
#endif
}

// Explicit template instantiations
#define INSTANTIATE_PARALLEL_ELEMENTWISE(Type) \
    template void vector_add_parallel<Type>(const Type*, const Type*, Type*, std::size_t, std::size_t) noexcept; \
    template void vector_mul_parallel<Type>(const Type*, const Type*, Type*, std::size_t, std::size_t) noexcept; \
    template void scalar_mul_parallel<Type>(const Type*, Type, Type*, std::size_t, std::size_t) noexcept; \
    template void saxpy_parallel<Type>(const Type*, Type*, Type, std::size_t, std::size_t) noexcept; \
    template void vector_scale_parallel<Type>(Type*, Type, std::size_t, std::size_t) noexcept;

INSTANTIATE_PARALLEL_ELEMENTWISE(float)
INSTANTIATE_PARALLEL_ELEMENTWISE(double)
INSTANTIATE_PARALLEL_ELEMENTWISE(int32_t)
INSTANTIATE_PARALLEL_ELEMENTWISE(int64_t)

} // namespace parallel_compute
