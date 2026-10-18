#pragma once

#include "parallel_compute/common/error.hpp"
#include "parallel_compute/common/traits.hpp"
#include "parallel_compute/common/types.hpp"
#include <cstddef>

namespace parallel_compute {

// --- 1. Reference Implementation (Correctness baseline) ---
template <Numeric T>
void matmul_reference(const MatrixView<const T>& a,
                      const MatrixView<const T>& b,
                      MatrixView<T> c);

// Overload for Matrix<T>
template <Numeric T>
void matmul_reference(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c) {
    matmul_reference(a.view(), b.view(), c.view());
}

// --- 2. Cache-Blocked Tiled Implementation ---
template <Numeric T>
void matmul_blocked(const MatrixView<const T>& a,
                    const MatrixView<const T>& b,
                    MatrixView<T> c,
                    std::size_t block_m = 64,
                    std::size_t block_n = 64,
                    std::size_t block_k = 64);

template <Numeric T>
void matmul_blocked(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c,
                    std::size_t block_m = 64, std::size_t block_n = 64, std::size_t block_k = 64) {
    matmul_blocked(a.view(), b.view(), c.view(), block_m, block_n, block_k);
}

// --- 3. Optional External BLAS Comparison ---
#if defined(PARALLEL_COMPUTE_HAS_BLAS)
template <FloatingPoint T>
void matmul_blas(const MatrixView<const T>& a,
                 const MatrixView<const T>& b,
                 MatrixView<T> c);

template <FloatingPoint T>
void matmul_blas(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c) {
    matmul_blas(a.view(), b.view(), c.view());
}
#endif

// Dimension validation helper
template <Numeric T1, Numeric T2, Numeric T3>
void validate_matmul_dimensions(std::size_t a_rows, std::size_t a_cols,
                                std::size_t b_rows, std::size_t b_cols,
                                std::size_t c_rows, std::size_t c_cols) {
    if (a_cols != b_rows) {
        throw DimensionMismatchError(
            "Matrix multiplication inner dimensions mismatch: A cols (" +
            std::to_string(a_cols) + ") != B rows (" + std::to_string(b_rows) + ")");
    }
    if (a_rows != c_rows || b_cols != c_cols) {
        throw DimensionMismatchError(
            "Output matrix C dimensions mismatch: expected (" +
            std::to_string(a_rows) + "x" + std::to_string(b_cols) + ") but got (" +
            std::to_string(c_rows) + "x" + std::to_string(c_cols) + ")");
    }
}

} // namespace parallel_compute
