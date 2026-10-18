#include "parallel_compute/algorithms/matmul.hpp"

#include <algorithm>
#include <cstring>

#if defined(PARALLEL_COMPUTE_HAS_BLAS)
#include <cblas.h>
#endif

namespace parallel_compute {

template <Numeric T>
void matmul_reference(const MatrixView<const T>& a,
                      const MatrixView<const T>& b,
                      MatrixView<T> c) {
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();

    validate_matmul_dimensions<T, T, T>(M, K, b.rows(), N, c.rows(), c.cols());

    // Initialize C to zero
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            c(i, j) = T{0};
        }
    }

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            T sum = T{0};
            for (std::size_t k = 0; k < K; ++k) {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }
}

template <Numeric T>
void matmul_blocked(const MatrixView<const T>& a,
                    const MatrixView<const T>& b,
                    MatrixView<T> c,
                    std::size_t block_m,
                    std::size_t block_n,
                    std::size_t block_k) {
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();

    validate_matmul_dimensions<T, T, T>(M, K, b.rows(), N, c.rows(), c.cols());

    if (block_m == 0) block_m = 64;
    if (block_n == 0) block_n = 64;
    if (block_k == 0) block_k = 64;

    // Initialize C to zero
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            c(i, j) = T{0};
        }
    }

    // Tiled execution with i-k-j loop order for continuous cache lines
    for (std::size_t bi = 0; bi < M; bi += block_m) {
        const std::size_t i_max = std::min(bi + block_m, M);
        for (std::size_t bk = 0; bk < K; bk += block_k) {
            const std::size_t k_max = std::min(bk + block_k, K);
            for (std::size_t bj = 0; bj < N; bj += block_n) {
                const std::size_t j_max = std::min(bj + block_n, N);

                for (std::size_t i = bi; i < i_max; ++i) {
                    for (std::size_t k = bk; k < k_max; ++k) {
                        const T a_ik = a(i, k);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
                        for (std::size_t j = bj; j < j_max; ++j) {
                            c(i, j) += a_ik * b(k, j);
                        }
                    }
                }
            }
        }
    }
}

#if defined(PARALLEL_COMPUTE_HAS_BLAS)
template <>
void matmul_blas<float>(const MatrixView<const float>& a,
                        const MatrixView<const float>& b,
                        MatrixView<float> c) {
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();
    validate_matmul_dimensions<float, float, float>(M, K, b.rows(), N, c.rows(), c.cols());

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0f, a.data(), static_cast<int>(a.lda()),
                b.data(), static_cast<int>(b.lda()),
                0.0f, c.data(), static_cast<int>(c.lda()));
}

template <>
void matmul_blas<double>(const MatrixView<const double>& a,
                         const MatrixView<const double>& b,
                         MatrixView<double> c) {
    const std::size_t M = a.rows();
    const std::size_t K = a.cols();
    const std::size_t N = b.cols();
    validate_matmul_dimensions<double, double, double>(M, K, b.rows(), N, c.rows(), c.cols());

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0, a.data(), static_cast<int>(a.lda()),
                b.data(), static_cast<int>(b.lda()),
                0.0, c.data(), static_cast<int>(c.lda()));
}
#endif

// Explicit template instantiations
#define INSTANTIATE_MATMUL(Type) \
    template void matmul_reference<Type>(const MatrixView<const Type>&, const MatrixView<const Type>&, MatrixView<Type>); \
    template void matmul_blocked<Type>(const MatrixView<const Type>&, const MatrixView<const Type>&, MatrixView<Type>, std::size_t, std::size_t, std::size_t);

INSTANTIATE_MATMUL(float)
INSTANTIATE_MATMUL(double)
INSTANTIATE_MATMUL(int32_t)
INSTANTIATE_MATMUL(int64_t)

} // namespace parallel_compute
