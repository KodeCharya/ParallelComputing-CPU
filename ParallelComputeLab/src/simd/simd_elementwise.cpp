#include "parallel_compute/simd/simd_elementwise.hpp"
#include "parallel_compute/algorithms/elementwise.hpp"
#include "parallel_compute/simd/simd_detect.hpp"

#if defined(PARALLEL_COMPUTE_HAS_AVX2)
#include <immintrin.h>
#endif

namespace parallel_compute {

#if defined(PARALLEL_COMPUTE_HAS_AVX2)

namespace avx2 {

void vector_add_float(const float* a, const float* b, float* c, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kStep = 8;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;

    for (; i < limit; i += kStep) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void vector_mul_float(const float* a, const float* b, float* c, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kStep = 8;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;

    for (; i < limit; i += kStep) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

void saxpy_float(const float* x, float* y, float alpha, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kStep = 8;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256 valpha = _mm256_set1_ps(alpha);

    for (; i < limit; i += kStep) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 res = _mm256_fmadd_ps(valpha, vx, vy);
        _mm256_storeu_ps(y + i, res);
    }
    for (; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

void vector_scale_float(float* x, float alpha, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kStep = 8;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256 valpha = _mm256_set1_ps(alpha);

    for (; i < limit; i += kStep) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 res = _mm256_mul_ps(valpha, vx);
        _mm256_storeu_ps(x + i, res);
    }
    for (; i < n; ++i) {
        x[i] = alpha * x[i];
    }
}

void vector_add_double(const double* a, const double* b, double* c, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kStep = 4;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;

    for (; i < limit; i += kStep) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void vector_mul_double(const double* a, const double* b, double* c, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kStep = 4;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;

    for (; i < limit; i += kStep) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vc = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(c + i, vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

void saxpy_double(const double* x, double* y, double alpha, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kStep = 4;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256d valpha = _mm256_set1_pd(alpha);

    for (; i < limit; i += kStep) {
        __m256d vx = _mm256_loadu_pd(x + i);
        __m256d vy = _mm256_loadu_pd(y + i);
        __m256d res = _mm256_fmadd_pd(valpha, vx, vy);
        _mm256_storeu_pd(y + i, res);
    }
    for (; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

void vector_scale_double(double* x, double alpha, std::size_t n) noexcept {
    std::size_t i = 0;
    constexpr std::size_t kStep = 4;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256d valpha = _mm256_set1_pd(alpha);

    for (; i < limit; i += kStep) {
        __m256d vx = _mm256_loadu_pd(x + i);
        __m256d res = _mm256_mul_pd(valpha, vx);
        _mm256_storeu_pd(x + i, res);
    }
    for (; i < n; ++i) {
        x[i] = alpha * x[i];
    }
}

} // namespace avx2

#endif // PARALLEL_COMPUTE_HAS_AVX2

template <>
void vector_add_simd<float>(const float* a, const float* b, float* c, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        avx2::vector_add_float(a, b, c, n);
        return;
    }
#endif
    vector_add_optimized(a, b, c, n);
}

template <>
void vector_mul_simd<float>(const float* a, const float* b, float* c, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        avx2::vector_mul_float(a, b, c, n);
        return;
    }
#endif
    vector_mul_optimized(a, b, c, n);
}

template <>
void saxpy_simd<float>(const float* x, float* y, float alpha, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported() && is_fma_supported()) {
        avx2::saxpy_float(x, y, alpha, n);
        return;
    }
#endif
    saxpy_optimized(x, y, alpha, n);
}

template <>
void vector_scale_simd<float>(float* x, float alpha, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        avx2::vector_scale_float(x, alpha, n);
        return;
    }
#endif
    vector_scale_optimized(x, alpha, n);
}

template <>
void vector_add_simd<double>(const double* a, const double* b, double* c, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        avx2::vector_add_double(a, b, c, n);
        return;
    }
#endif
    vector_add_optimized(a, b, c, n);
}

template <>
void vector_mul_simd<double>(const double* a, const double* b, double* c, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        avx2::vector_mul_double(a, b, c, n);
        return;
    }
#endif
    vector_mul_optimized(a, b, c, n);
}

template <>
void saxpy_simd<double>(const double* x, double* y, double alpha, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported() && is_fma_supported()) {
        avx2::saxpy_double(x, y, alpha, n);
        return;
    }
#endif
    saxpy_optimized(x, y, alpha, n);
}

template <>
void vector_scale_simd<double>(double* x, double alpha, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        avx2::vector_scale_double(x, alpha, n);
        return;
    }
#endif
    vector_scale_optimized(x, alpha, n);
}

template <>
void vector_add_simd<int32_t>(const int32_t* a, const int32_t* b, int32_t* c, std::size_t n) noexcept {
    vector_add_optimized(a, b, c, n);
}

template <>
void vector_mul_simd<int32_t>(const int32_t* a, const int32_t* b, int32_t* c, std::size_t n) noexcept {
    vector_mul_optimized(a, b, c, n);
}

template <>
void saxpy_simd<int32_t>(const int32_t* x, int32_t* y, int32_t alpha, std::size_t n) noexcept {
    saxpy_optimized(x, y, alpha, n);
}

template <>
void vector_scale_simd<int32_t>(int32_t* x, int32_t alpha, std::size_t n) noexcept {
    vector_scale_optimized(x, alpha, n);
}

template <>
void vector_add_simd<int64_t>(const int64_t* a, const int64_t* b, int64_t* c, std::size_t n) noexcept {
    vector_add_optimized(a, b, c, n);
}

template <>
void vector_mul_simd<int64_t>(const int64_t* a, const int64_t* b, int64_t* c, std::size_t n) noexcept {
    vector_mul_optimized(a, b, c, n);
}

template <>
void saxpy_simd<int64_t>(const int64_t* x, int64_t* y, int64_t alpha, std::size_t n) noexcept {
    saxpy_optimized(x, y, alpha, n);
}

template <>
void vector_scale_simd<int64_t>(int64_t* x, int64_t alpha, std::size_t n) noexcept {
    vector_scale_optimized(x, alpha, n);
}

} // namespace parallel_compute
