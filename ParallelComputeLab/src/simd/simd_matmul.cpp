#include "parallel_compute/simd/simd_matmul.hpp"
#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/simd/simd_detect.hpp"

#include <algorithm>

#if defined(PARALLEL_COMPUTE_HAS_AVX2)
#include <immintrin.h>
#endif

namespace parallel_compute {

#if defined(PARALLEL_COMPUTE_HAS_AVX2)

namespace avx2 {

void matmul_float_avx2(const MatrixView<const float>& a,
                       const MatrixView<const float>& b,
                       MatrixView<float> c,
                       std::size_t block_m,
                       std::size_t block_n,
                       std::size_t block_k) {
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();

    // Zero output
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            c(i, j) = 0.0f;
        }
    }

    for (std::size_t bi = 0; bi < M; bi += block_m) {
        const std::size_t i_max = std::min(bi + block_m, M);
        for (std::size_t bk = 0; bk < K; bk += block_k) {
            const std::size_t k_max = std::min(bk + block_k, K);
            for (std::size_t bj = 0; bj < N; bj += block_n) {
                const std::size_t j_max = std::min(bj + block_n, N);

                for (std::size_t i = bi; i < i_max; ++i) {
                    for (std::size_t k = bk; k < k_max; ++k) {
                        const float a_val = a(i, k);
                        const __m256 va = _mm256_set1_ps(a_val);
                        const float* b_row = b.row_ptr(k);
                        float* c_row = c.row_ptr(i);

                        std::size_t j = bj;
                        constexpr std::size_t kStep = 8;
                        const std::size_t j_simd_limit = (j_max >= kStep) ? (j_max - kStep + 1) : 0;

                        for (; j < j_simd_limit; j += kStep) {
                            __m256 vb = _mm256_loadu_ps(b_row + j);
                            __m256 vc = _mm256_loadu_ps(c_row + j);
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            _mm256_storeu_ps(c_row + j, vc);
                        }
                        for (; j < j_max; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

void matmul_double_avx2(const MatrixView<const double>& a,
                        const MatrixView<const double>& b,
                        MatrixView<double> c,
                        std::size_t block_m,
                        std::size_t block_n,
                        std::size_t block_k) {
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();

    // Zero output
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            c(i, j) = 0.0;
        }
    }

    for (std::size_t bi = 0; bi < M; bi += block_m) {
        const std::size_t i_max = std::min(bi + block_m, M);
        for (std::size_t bk = 0; bk < K; bk += block_k) {
            const std::size_t k_max = std::min(bk + block_k, K);
            for (std::size_t bj = 0; bj < N; bj += block_n) {
                const std::size_t j_max = std::min(bj + block_n, N);

                for (std::size_t i = bi; i < i_max; ++i) {
                    for (std::size_t k = bk; k < k_max; ++k) {
                        const double a_val = a(i, k);
                        const __m256d va = _mm256_set1_pd(a_val);
                        const double* b_row = b.row_ptr(k);
                        double* c_row = c.row_ptr(i);

                        std::size_t j = bj;
                        constexpr std::size_t kStep = 4;
                        const std::size_t j_simd_limit = (j_max >= kStep) ? (j_max - kStep + 1) : 0;

                        for (; j < j_simd_limit; j += kStep) {
                            __m256d vb = _mm256_loadu_pd(b_row + j);
                            __m256d vc = _mm256_loadu_pd(c_row + j);
                            vc = _mm256_fmadd_pd(va, vb, vc);
                            _mm256_storeu_pd(c_row + j, vc);
                        }
                        for (; j < j_max; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

} // namespace avx2

#endif // PARALLEL_COMPUTE_HAS_AVX2

template <>
void matmul_simd<float>(const MatrixView<const float>& a,
                        const MatrixView<const float>& b,
                        MatrixView<float> c,
                        std::size_t block_m,
                        std::size_t block_n,
                        std::size_t block_k) {
    validate_matmul_dimensions<float, float, float>(a.rows(), a.cols(), b.rows(), b.cols(), c.rows(), c.cols());
    if (block_m == 0) block_m = 64;
    if (block_n == 0) block_n = 64;
    if (block_k == 0) block_k = 64;

#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported() && is_fma_supported()) {
        avx2::matmul_float_avx2(a, b, c, block_m, block_n, block_k);
        return;
    }
#endif
    matmul_blocked(a, b, c, block_m, block_n, block_k);
}

template <>
void matmul_simd<double>(const MatrixView<const double>& a,
                         const MatrixView<const double>& b,
                         MatrixView<double> c,
                         std::size_t block_m,
                         std::size_t block_n,
                         std::size_t block_k) {
    validate_matmul_dimensions<double, double, double>(a.rows(), a.cols(), b.rows(), b.cols(), c.rows(), c.cols());
    if (block_m == 0) block_m = 64;
    if (block_n == 0) block_n = 64;
    if (block_k == 0) block_k = 64;

#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    if (is_avx2_supported() && is_fma_supported()) {
        avx2::matmul_double_avx2(a, b, c, block_m, block_n, block_k);
        return;
    }
#endif
    matmul_blocked(a, b, c, block_m, block_n, block_k);
}

template <>
void matmul_simd<int32_t>(const MatrixView<const int32_t>& a,
                          const MatrixView<const int32_t>& b,
                          MatrixView<int32_t> c,
                          std::size_t block_m,
                          std::size_t block_n,
                          std::size_t block_k) {
    matmul_blocked(a, b, c, block_m, block_n, block_k);
}

template <>
void matmul_simd<int64_t>(const MatrixView<const int64_t>& a,
                          const MatrixView<const int64_t>& b,
                          MatrixView<int64_t> c,
                          std::size_t block_m,
                          std::size_t block_n,
                          std::size_t block_k) {
    matmul_blocked(a, b, c, block_m, block_n, block_k);
}

} // namespace parallel_compute
