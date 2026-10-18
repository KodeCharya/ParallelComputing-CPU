#include <gtest/gtest.h>
#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/parallel/parallel_matmul.hpp"
#include "parallel_compute/simd/simd_matmul.hpp"
#include "parallel_compute/common/types.hpp"

#include <cmath>

using namespace parallel_compute;

struct MatmulDims {
    std::size_t M, K, N;
};

class MatmulNumericalTest : public ::testing::TestWithParam<MatmulDims> {};

TEST_P(MatmulNumericalTest, BlockedAndParallelAndSIMDMatchReferenceFloat) {
    const auto [M, K, N] = GetParam();

    Matrix<float> A(M, K);
    Matrix<float> B(K, N);
    Matrix<float> C_ref(M, N);
    Matrix<float> C_blk(M, N);
    Matrix<float> C_par(M, N);
    Matrix<float> C_simd(M, N);

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            A(i, k) = static_cast<float>((i + k) % 17) * 0.1f;
        }
    }
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t j = 0; j < N; ++j) {
            B(k, j) = static_cast<float>((k * 2 + j) % 19) * 0.1f;
        }
    }

    matmul_reference(A, B, C_ref);
    matmul_blocked(A, B, C_blk, 16, 16, 16);
    matmul_parallel(A, B, C_par, 4, 16, 16, 16);
    matmul_simd(A, B, C_simd, 16, 16, 16);

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            const float ref = C_ref(i, j);
            EXPECT_NEAR(C_blk(i, j), ref, 1e-4f) << "Blocked mismatch at (" << i << ", " << j << ")";
            EXPECT_NEAR(C_par(i, j), ref, 1e-4f) << "Parallel mismatch at (" << i << ", " << j << ")";
            EXPECT_NEAR(C_simd(i, j), ref, 1e-4f) << "SIMD mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST_P(MatmulNumericalTest, BlockedAndParallelAndSIMDMatchReferenceDouble) {
    const auto [M, K, N] = GetParam();

    Matrix<double> A(M, K);
    Matrix<double> B(K, N);
    Matrix<double> C_ref(M, N);
    Matrix<double> C_blk(M, N);
    Matrix<double> C_par(M, N);
    Matrix<double> C_simd(M, N);

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            A(i, k) = static_cast<double>(i + k) * 0.05;
        }
    }
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t j = 0; j < N; ++j) {
            B(k, j) = static_cast<double>(k + j * 2) * 0.05;
        }
    }

    matmul_reference(A, B, C_ref);
    matmul_blocked(A, B, C_blk, 16, 16, 16);
    matmul_parallel(A, B, C_par, 4, 16, 16, 16);
    matmul_simd(A, B, C_simd, 16, 16, 16);

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            const double ref = C_ref(i, j);
            EXPECT_NEAR(C_blk(i, j), ref, 1e-9) << "Blocked mismatch at (" << i << ", " << j << ")";
            EXPECT_NEAR(C_par(i, j), ref, 1e-9) << "Parallel mismatch at (" << i << ", " << j << ")";
            EXPECT_NEAR(C_simd(i, j), ref, 1e-9) << "SIMD mismatch at (" << i << ", " << j << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    RectangularDimensions,
    MatmulNumericalTest,
    ::testing::Values(
        MatmulDims{1, 1, 1},
        MatmulDims{7, 13, 11},
        MatmulDims{25, 33, 19},
        MatmulDims{64, 64, 64},
        MatmulDims{1, 65, 1},
        MatmulDims{48, 17, 35}
    )
);
