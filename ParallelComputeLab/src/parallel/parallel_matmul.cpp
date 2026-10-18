#include "parallel_compute/parallel/parallel_matmul.hpp"
#include "parallel_compute/algorithms/matmul.hpp"

#include <algorithm>

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
#include <omp.h>
#endif

namespace parallel_compute {

template <Numeric T>
void matmul_parallel(const MatrixView<const T>& a,
                     const MatrixView<const T>& b,
                     MatrixView<T> c,
                     std::size_t num_threads,
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

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();

    // Parallelize over row blocks
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t bi = 0; bi < M; bi += block_m) {
        const std::size_t i_max = std::min(bi + block_m, M);

        // Zero output block rows owned by this task
        for (std::size_t i = bi; i < i_max; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                c(i, j) = T{0};
            }
        }

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
#else
    (void)num_threads;
    matmul_blocked(a, b, c, block_m, block_n, block_k);
#endif
}

// Explicit template instantiations
#define INSTANTIATE_PARALLEL_MATMUL(Type) \
    template void matmul_parallel<Type>(const MatrixView<const Type>&, const MatrixView<const Type>&, MatrixView<Type>, std::size_t, std::size_t, std::size_t, std::size_t);

INSTANTIATE_PARALLEL_MATMUL(float)
INSTANTIATE_PARALLEL_MATMUL(double)
INSTANTIATE_PARALLEL_MATMUL(int32_t)
INSTANTIATE_PARALLEL_MATMUL(int64_t)

} // namespace parallel_compute
