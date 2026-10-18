#include <gtest/gtest.h>
#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/parallel/parallel_matmul.hpp"
#include "parallel_compute/simd/simd_matmul.hpp"
#include "parallel_compute/common/types.hpp"
#include "parallel_compute/common/error.hpp"

using namespace parallel_compute;

TEST(DimensionValidationTest, ThrowsOnInnerDimensionMismatch) {
    Matrix<float> A(10, 5);
    Matrix<float> B(6, 8); // B rows (6) != A cols (5)
    Matrix<float> C(10, 8);

    EXPECT_THROW(matmul_reference(A, B, C), DimensionMismatchError);
    EXPECT_THROW(matmul_blocked(A, B, C), DimensionMismatchError);
    EXPECT_THROW(matmul_parallel(A, B, C), DimensionMismatchError);
    EXPECT_THROW(matmul_simd(A, B, C), DimensionMismatchError);
}

TEST(DimensionValidationTest, ThrowsOnOutputDimensionMismatch) {
    Matrix<float> A(10, 5);
    Matrix<float> B(5, 8);
    Matrix<float> C_wrong_rows(9, 8);
    Matrix<float> C_wrong_cols(10, 7);

    EXPECT_THROW(matmul_reference(A, B, C_wrong_rows), DimensionMismatchError);
    EXPECT_THROW(matmul_blocked(A, B, C_wrong_rows), DimensionMismatchError);
    EXPECT_THROW(matmul_parallel(A, B, C_wrong_rows), DimensionMismatchError);
    EXPECT_THROW(matmul_simd(A, B, C_wrong_rows), DimensionMismatchError);

    EXPECT_THROW(matmul_reference(A, B, C_wrong_cols), DimensionMismatchError);
    EXPECT_THROW(matmul_blocked(A, B, C_wrong_cols), DimensionMismatchError);
    EXPECT_THROW(matmul_parallel(A, B, C_wrong_cols), DimensionMismatchError);
    EXPECT_THROW(matmul_simd(A, B, C_wrong_cols), DimensionMismatchError);
}

TEST(DimensionValidationTest, ThrowsOnMatrixOutOfBoundsAccess) {
    Matrix<float> M(4, 4);
    EXPECT_NO_THROW((void)M(3, 3));
    EXPECT_THROW((void)M(4, 0), DimensionMismatchError);
    EXPECT_THROW((void)M(0, 4), DimensionMismatchError);
    EXPECT_THROW((void)M(10, 10), DimensionMismatchError);
}
