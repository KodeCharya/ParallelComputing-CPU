#include <gtest/gtest.h>
#include "parallel_compute/algorithms/elementwise.hpp"
#include "parallel_compute/parallel/parallel_elementwise.hpp"
#include "parallel_compute/simd/simd_elementwise.hpp"
#include "parallel_compute/common/types.hpp"

using namespace parallel_compute;

class ElementwiseTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(ElementwiseTest, VectorAddMatchesAcrossAllVariants) {
    const std::size_t n = GetParam();
    AlignedVector<float> a(n);
    AlignedVector<float> b(n);
    AlignedVector<float> c_ref(n);
    AlignedVector<float> c_opt(n);
    AlignedVector<float> c_par(n);
    AlignedVector<float> c_simd(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = static_cast<float>(i + 1) * 1.5f;
    }

    vector_add_scalar(a.data(), b.data(), c_ref.data(), n);
    vector_add_optimized(a.data(), b.data(), c_opt.data(), n);
    vector_add_parallel(a.data(), b.data(), c_par.data(), n, 2);
    vector_add_simd(a.data(), b.data(), c_simd.data(), n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(c_opt[i], c_ref[i]) << "Mismatch at index " << i;
        EXPECT_FLOAT_EQ(c_par[i], c_ref[i]) << "Mismatch at index " << i;
        EXPECT_FLOAT_EQ(c_simd[i], c_ref[i]) << "Mismatch at index " << i;
    }
}

TEST_P(ElementwiseTest, VectorMulMatchesAcrossAllVariants) {
    const std::size_t n = GetParam();
    AlignedVector<float> a(n);
    AlignedVector<float> b(n);
    AlignedVector<float> c_ref(n);
    AlignedVector<float> c_opt(n);
    AlignedVector<float> c_par(n);
    AlignedVector<float> c_simd(n);

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i + 1) * 0.1f;
        b[i] = static_cast<float>(i + 2) * 0.2f;
    }

    vector_mul_scalar(a.data(), b.data(), c_ref.data(), n);
    vector_mul_optimized(a.data(), b.data(), c_opt.data(), n);
    vector_mul_parallel(a.data(), b.data(), c_par.data(), n, 4);
    vector_mul_simd(a.data(), b.data(), c_simd.data(), n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(c_opt[i], c_ref[i]) << "Mismatch at index " << i;
        EXPECT_FLOAT_EQ(c_par[i], c_ref[i]) << "Mismatch at index " << i;
        EXPECT_FLOAT_EQ(c_simd[i], c_ref[i]) << "Mismatch at index " << i;
    }
}

TEST_P(ElementwiseTest, SAXPYMatchesAcrossAllVariants) {
    const std::size_t n = GetParam();
    AlignedVector<float> x(n);
    AlignedVector<float> y_ref(n);
    AlignedVector<float> y_opt(n);
    AlignedVector<float> y_par(n);
    AlignedVector<float> y_simd(n);
    const float alpha = 3.14159f;

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i) * 0.25f;
        y_ref[i] = y_opt[i] = y_par[i] = y_simd[i] = static_cast<float>(i) * 1.75f;
    }

    saxpy_scalar(x.data(), y_ref.data(), alpha, n);
    saxpy_optimized(x.data(), y_opt.data(), alpha, n);
    saxpy_parallel(x.data(), y_par.data(), alpha, n, 4);
    saxpy_simd(x.data(), y_simd.data(), alpha, n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(y_opt[i], y_ref[i]) << "Mismatch at index " << i;
        EXPECT_FLOAT_EQ(y_par[i], y_ref[i]) << "Mismatch at index " << i;
        EXPECT_FLOAT_EQ(y_simd[i], y_ref[i]) << "Mismatch at index " << i;
    }
}

TEST_P(ElementwiseTest, VectorScaleMatchesAcrossAllVariants) {
    const std::size_t n = GetParam();
    AlignedVector<double> x_ref(n);
    AlignedVector<double> x_opt(n);
    AlignedVector<double> x_par(n);
    AlignedVector<double> x_simd(n);
    const double alpha = 2.718281828;

    for (std::size_t i = 0; i < n; ++i) {
        x_ref[i] = x_opt[i] = x_par[i] = x_simd[i] = static_cast<double>(i + 1);
    }

    vector_scale_scalar(x_ref.data(), alpha, n);
    vector_scale_optimized(x_opt.data(), alpha, n);
    vector_scale_parallel(x_par.data(), alpha, n, 4);
    vector_scale_simd(x_simd.data(), alpha, n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(x_opt[i], x_ref[i]) << "Mismatch at index " << i;
        EXPECT_DOUBLE_EQ(x_par[i], x_ref[i]) << "Mismatch at index " << i;
        EXPECT_DOUBLE_EQ(x_simd[i], x_ref[i]) << "Mismatch at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    VaryingSizes,
    ElementwiseTest,
    ::testing::Values(1u, 7u, 8u, 15u, 16u, 33u, 128u, 1025u)
);
