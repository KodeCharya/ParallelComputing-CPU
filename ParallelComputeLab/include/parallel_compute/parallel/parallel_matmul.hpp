#pragma once

#include "parallel_compute/common/traits.hpp"
#include "parallel_compute/common/types.hpp"
#include <cstddef>

namespace parallel_compute {

template <Numeric T>
void matmul_parallel(const MatrixView<const T>& a,
                     const MatrixView<const T>& b,
                     MatrixView<T> c,
                     std::size_t num_threads = 0,
                     std::size_t block_m = 64,
                     std::size_t block_n = 64,
                     std::size_t block_k = 64);

template <Numeric T>
void matmul_parallel(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c,
                     std::size_t num_threads = 0,
                     std::size_t block_m = 64, std::size_t block_n = 64, std::size_t block_k = 64) {
    matmul_parallel(a.view(), b.view(), c.view(), num_threads, block_m, block_n, block_k);
}

} // namespace parallel_compute
