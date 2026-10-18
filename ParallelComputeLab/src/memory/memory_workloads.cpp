#include "parallel_compute/memory/memory_workloads.hpp"
#include <cstdint>

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
#include <omp.h>
#endif

namespace parallel_compute {

template <Numeric T>
void array_copy_serial(T* dst, const T* src, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

template <Numeric T>
void array_scale_serial(T* dst, const T* src, T alpha, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = alpha * src[i];
    }
}

template <Numeric T>
void array_triad_serial(T* a, const T* b, const T* c, T alpha, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = b[i] + alpha * c[i];
    }
}

template <Numeric T>
T sequential_read_serial(const T* src, std::size_t n) noexcept {
    T sum = T{0};
    for (std::size_t i = 0; i < n; ++i) {
        sum += src[i];
    }
    return sum;
}

template <Numeric T>
T strided_read_serial(const T* src, std::size_t n, std::size_t stride) noexcept {
    if (stride == 0) stride = 1;
    T sum = T{0};
    for (std::size_t i = 0; i < n; i += stride) {
        sum += src[i];
    }
    return sum;
}

template <Numeric T>
void array_copy_parallel(T* dst, const T* src, std::size_t n, std::size_t num_threads) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
#else
    (void)num_threads;
    array_copy_serial(dst, src, n);
#endif
}

template <Numeric T>
void array_scale_parallel(T* dst, const T* src, T alpha, std::size_t n, std::size_t num_threads) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = alpha * src[i];
    }
#else
    (void)num_threads;
    array_scale_serial(dst, src, alpha, n);
#endif
}

template <Numeric T>
void array_triad_parallel(T* a, const T* b, const T* c, T alpha, std::size_t n, std::size_t num_threads) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = b[i] + alpha * c[i];
    }
#else
    (void)num_threads;
    array_triad_serial(a, b, c, alpha, n);
#endif
}

// Explicit template instantiations
#define INSTANTIATE_MEMORY_WORKLOADS(Type) \
    template void array_copy_serial<Type>(Type*, const Type*, std::size_t) noexcept; \
    template void array_scale_serial<Type>(Type*, const Type*, Type, std::size_t) noexcept; \
    template void array_triad_serial<Type>(Type*, const Type*, const Type*, Type, std::size_t) noexcept; \
    template Type sequential_read_serial<Type>(const Type*, std::size_t) noexcept; \
    template Type strided_read_serial<Type>(const Type*, std::size_t, std::size_t) noexcept; \
    template void array_copy_parallel<Type>(Type*, const Type*, std::size_t, std::size_t) noexcept; \
    template void array_scale_parallel<Type>(Type*, const Type*, Type, std::size_t, std::size_t) noexcept; \
    template void array_triad_parallel<Type>(Type*, const Type*, const Type*, Type, std::size_t, std::size_t) noexcept;

INSTANTIATE_MEMORY_WORKLOADS(float)
INSTANTIATE_MEMORY_WORKLOADS(double)
INSTANTIATE_MEMORY_WORKLOADS(int32_t)
INSTANTIATE_MEMORY_WORKLOADS(int64_t)

} // namespace parallel_compute
