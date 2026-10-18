#include "parallel_compute/simd/simd_reduction.hpp"
#include "parallel_compute/algorithms/reduction.hpp"
#include "parallel_compute/simd/simd_detect.hpp"

#include <algorithm>

#if defined(PARALLEL_COMPUTE_HAS_AVX2)
#include <immintrin.h>
#endif

namespace parallel_compute {

#if defined(PARALLEL_COMPUTE_HAS_AVX2)

namespace avx2 {

inline float hsum_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
    return _mm_cvtss_f32(sum128);
}

inline double hsum_pd(__m256d v) noexcept {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum128 = _mm_add_pd(lo, hi);
    sum128 = _mm_add_sd(sum128, _mm_unpackhi_pd(sum128, sum128));
    return _mm_cvtsd_f64(sum128);
}

inline float hmin_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 min128 = _mm_min_ps(lo, hi);
    min128 = _mm_min_ps(min128, _mm_movehl_ps(min128, min128));
    min128 = _mm_min_ss(min128, _mm_shuffle_ps(min128, min128, 1));
    return _mm_cvtss_f32(min128);
}

inline float hmax_ps(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(lo, hi);
    max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    max128 = _mm_max_ss(max128, _mm_shuffle_ps(max128, max128, 1));
    return _mm_cvtss_f32(max128);
}

inline double hmin_pd(__m256d v) noexcept {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d min128 = _mm_min_pd(lo, hi);
    min128 = _mm_min_sd(min128, _mm_unpackhi_pd(min128, min128));
    return _mm_cvtsd_f64(min128);
}

inline double hmax_pd(__m256d v) noexcept {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d max128 = _mm_max_pd(lo, hi);
    max128 = _mm_max_sd(max128, _mm_unpackhi_pd(max128, max128));
    return _mm_cvtsd_f64(max128);
}

float reduce_sum_float(const float* data, std::size_t n) noexcept {
    if (n == 0) return 0.0f;
    std::size_t i = 0;
    constexpr std::size_t kStep = 8;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256 acc = _mm256_setzero_ps();

    for (; i < limit; i += kStep) {
        __m256 v = _mm256_loadu_ps(data + i);
        acc = _mm256_add_ps(acc, v);
    }
    float total = hsum_ps(acc);
    for (; i < n; ++i) {
        total += data[i];
    }
    return total;
}

double reduce_sum_double(const double* data, std::size_t n) noexcept {
    if (n == 0) return 0.0;
    std::size_t i = 0;
    constexpr std::size_t kStep = 4;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256d acc = _mm256_setzero_pd();

    for (; i < limit; i += kStep) {
        __m256d v = _mm256_loadu_pd(data + i);
        acc = _mm256_add_pd(acc, v);
    }
    double total = hsum_pd(acc);
    for (; i < n; ++i) {
        total += data[i];
    }
    return total;
}

std::optional<float> reduce_min_float(const float* data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;
    if (n < 8) return reduce_min_scalar(data, n);

    std::size_t i = 8;
    constexpr std::size_t kStep = 8;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256 acc = _mm256_loadu_ps(data);

    for (; i < limit; i += kStep) {
        __m256 v = _mm256_loadu_ps(data + i);
        acc = _mm256_min_ps(acc, v);
    }
    float min_val = hmin_ps(acc);
    for (; i < n; ++i) {
        if (data[i] < min_val) min_val = data[i];
    }
    return min_val;
}

std::optional<float> reduce_max_float(const float* data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;
    if (n < 8) return reduce_max_scalar(data, n);

    std::size_t i = 8;
    constexpr std::size_t kStep = 8;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256 acc = _mm256_loadu_ps(data);

    for (; i < limit; i += kStep) {
        __m256 v = _mm256_loadu_ps(data + i);
        acc = _mm256_max_ps(acc, v);
    }
    float max_val = hmax_ps(acc);
    for (; i < n; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }
    return max_val;
}

std::optional<double> reduce_min_double(const double* data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;
    if (n < 4) return reduce_min_scalar(data, n);

    std::size_t i = 4;
    constexpr std::size_t kStep = 4;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256d acc = _mm256_loadu_pd(data);

    for (; i < limit; i += kStep) {
        __m256d v = _mm256_loadu_pd(data + i);
        acc = _mm256_min_pd(acc, v);
    }
    double min_val = hmin_pd(acc);
    for (; i < n; ++i) {
        if (data[i] < min_val) min_val = data[i];
    }
    return min_val;
}

std::optional<double> reduce_max_double(const double* data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;
    if (n < 4) return reduce_max_scalar(data, n);

    std::size_t i = 4;
    constexpr std::size_t kStep = 4;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256d acc = _mm256_loadu_pd(data);

    for (; i < limit; i += kStep) {
        __m256d v = _mm256_loadu_pd(data + i);
        acc = _mm256_max_pd(acc, v);
    }
    double max_val = hmax_pd(acc);
    for (; i < n; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }
    return max_val;
}

float dot_product_float(const float* a, const float* b, std::size_t n) noexcept {
    if (n == 0) return 0.0f;
    std::size_t i = 0;
    constexpr std::size_t kStep = 8;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256 acc = _mm256_setzero_ps();

    for (; i < limit; i += kStep) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float total = hsum_ps(acc);
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

double dot_product_double(const double* a, const double* b, std::size_t n) noexcept {
    if (n == 0) return 0.0;
    std::size_t i = 0;
    constexpr std::size_t kStep = 4;
    const std::size_t limit = (n >= kStep) ? (n - kStep + 1) : 0;
    __m256d acc = _mm256_setzero_pd();

    for (; i < limit; i += kStep) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        acc = _mm256_fmadd_pd(va, vb, acc);
    }
    double total = hsum_pd(acc);
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

} // namespace avx2

#endif // PARALLEL_COMPUTE_HAS_AVX2

template <>
float reduce_sum_simd<float>(const float* data, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        return avx2::reduce_sum_float(data, n);
    }
#endif
    return reduce_sum_optimized(data, n);
}

template <>
double reduce_sum_simd<double>(const double* data, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        return avx2::reduce_sum_double(data, n);
    }
#endif
    return reduce_sum_optimized(data, n);
}

template <>
std::optional<float> reduce_min_simd<float>(const float* data, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        return avx2::reduce_min_float(data, n);
    }
#endif
    return reduce_min_optimized(data, n);
}

template <>
std::optional<float> reduce_max_simd<float>(const float* data, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        return avx2::reduce_max_float(data, n);
    }
#endif
    return reduce_max_optimized(data, n);
}

template <>
std::optional<double> reduce_min_simd<double>(const double* data, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        return avx2::reduce_min_double(data, n);
    }
#endif
    return reduce_min_optimized(data, n);
}

template <>
std::optional<double> reduce_max_simd<double>(const double* data, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported()) {
        return avx2::reduce_max_double(data, n);
    }
#endif
    return reduce_max_optimized(data, n);
}

template <>
float dot_product_simd<float>(const float* a, const float* b, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported() && is_fma_supported()) {
        return avx2::dot_product_float(a, b, n);
    }
#endif
    return dot_product_optimized(a, b, n);
}

template <>
double dot_product_simd<double>(const double* a, const double* b, std::size_t n) noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported() && is_fma_supported()) {
        return avx2::dot_product_double(a, b, n);
    }
#endif
    return dot_product_optimized(a, b, n);
}

template <>
std::optional<double> mean_simd<float>(const float* data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;
    double sum = static_cast<double>(reduce_sum_simd(data, n));
    return sum / static_cast<double>(n);
}

template <>
std::optional<double> mean_simd<double>(const double* data, std::size_t n) noexcept {
    if (n == 0) return std::nullopt;
    double sum = reduce_sum_simd(data, n);
    return sum / static_cast<double>(n);
}

template <>
int32_t reduce_sum_simd<int32_t>(const int32_t* data, std::size_t n) noexcept {
    return reduce_sum_optimized(data, n);
}

template <>
std::optional<int32_t> reduce_min_simd<int32_t>(const int32_t* data, std::size_t n) noexcept {
    return reduce_min_optimized(data, n);
}

template <>
std::optional<int32_t> reduce_max_simd<int32_t>(const int32_t* data, std::size_t n) noexcept {
    return reduce_max_optimized(data, n);
}

template <>
int32_t dot_product_simd<int32_t>(const int32_t* a, const int32_t* b, std::size_t n) noexcept {
    return dot_product_optimized(a, b, n);
}

template <>
std::optional<double> mean_simd<int32_t>(const int32_t* data, std::size_t n) noexcept {
    return mean_optimized(data, n);
}

template <>
int64_t reduce_sum_simd<int64_t>(const int64_t* data, std::size_t n) noexcept {
    return reduce_sum_optimized(data, n);
}

template <>
std::optional<int64_t> reduce_min_simd<int64_t>(const int64_t* data, std::size_t n) noexcept {
    return reduce_min_optimized(data, n);
}

template <>
std::optional<int64_t> reduce_max_simd<int64_t>(const int64_t* data, std::size_t n) noexcept {
    return reduce_max_optimized(data, n);
}

template <>
int64_t dot_product_simd<int64_t>(const int64_t* a, const int64_t* b, std::size_t n) noexcept {
    return dot_product_optimized(a, b, n);
}

template <>
std::optional<double> mean_simd<int64_t>(const int64_t* data, std::size_t n) noexcept {
    return mean_optimized(data, n);
}

} // namespace parallel_compute
