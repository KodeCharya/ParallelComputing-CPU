#include <gtest/gtest.h>
#include "parallel_compute/algorithms/elementwise.hpp"
#include "parallel_compute/algorithms/reduction.hpp"
#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/simd/simd_elementwise.hpp"
#include "parallel_compute/simd/simd_reduction.hpp"
#include "parallel_compute/simd/simd_matmul.hpp"
#include "parallel_compute/common/types.hpp"

using namespace parallel_compute;

TEST(RegressionTest, RepeatedExecutionDeterminism) {
    constexpr std::size_t n = 1024;
    AlignedVector<float> a(n, 2.5f);
    AlignedVector<float> b(n, 4.0f);
    AlignedVector<float> c(n);

    float first_sum = 0.0f;
    for (int rep = 0; rep < 10; ++rep) {
        vector_add_simd(a.data(), b.data(), c.data(), n);
        float current_sum = reduce_sum_simd(c.data(), n);
        if (rep == 0) {
            first_sum = current_sum;
        } else {
            EXPECT_FLOAT_EQ(current_sum, first_sum);
        }
    }
}

TEST(RegressionTest, ExhaustiveVectorTailLengthsFrom1To33) {
    // Check every vector length from 1 to 33 to guarantee all SIMD remainder branches work
    for (std::size_t len = 1; len <= 33; ++len) {
        AlignedVector<float> x(len, 1.5f);
        AlignedVector<float> y(len, 2.5f);
        AlignedVector<float> z_simd(len);
        AlignedVector<float> z_ref(len);

        vector_add_scalar(x.data(), y.data(), z_ref.data(), len);
        vector_add_simd(x.data(), y.data(), z_simd.data(), len);

        for (std::size_t i = 0; i < len; ++i) {
            EXPECT_FLOAT_EQ(z_simd[i], z_ref[i]) << "Mismatch at len=" << len << ", i=" << i;
        }

        float dot_ref = dot_product_scalar(x.data(), y.data(), len);
        float dot_simd = dot_product_simd(x.data(), y.data(), len);
        EXPECT_NEAR(dot_simd, dot_ref, 1e-4f) << "Dot mismatch at len=" << len;
    }
}

TEST(RegressionTest, TallSkinnyAndShortWideMatmul) {
    // Tall and skinny
    {
        Matrix<float> A(64, 2, 1.0f);
        Matrix<float> B(2, 64, 2.0f);
        Matrix<float> C_ref(64, 64);
        Matrix<float> C_simd(64, 64);

        matmul_reference(A, B, C_ref);
        matmul_simd(A, B, C_simd, 16, 16, 16);

        for (std::size_t i = 0; i < 64; ++i) {
            for (std::size_t j = 0; j < 64; ++j) {
                EXPECT_FLOAT_EQ(C_simd(i, j), C_ref(i, j));
            }
        }
    }
    // Short and wide
    {
        Matrix<float> A(2, 64, 1.0f);
        Matrix<float> B(64, 2, 2.0f);
        Matrix<float> C_ref(2, 2);
        Matrix<float> C_simd(2, 2);

        matmul_reference(A, B, C_ref);
        matmul_simd(A, B, C_simd, 16, 16, 16);

        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                EXPECT_FLOAT_EQ(C_simd(i, j), C_ref(i, j));
            }
        }
    }
}
