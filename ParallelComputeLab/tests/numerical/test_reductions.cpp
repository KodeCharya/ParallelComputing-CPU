#include <gtest/gtest.h>
#include "parallel_compute/algorithms/reduction.hpp"
#include "parallel_compute/parallel/parallel_reduction.hpp"
#include "parallel_compute/simd/simd_reduction.hpp"
#include "parallel_compute/common/types.hpp"

#include <cmath>

using namespace parallel_compute;

TEST(ReductionsTest, SumMatchesWithinTolerance) {
    constexpr std::size_t n = 10000;
    AlignedVector<float> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i % 100) * 0.01f;
    }

    float sum_ref = reduce_sum_scalar(data.data(), n);
    float sum_opt = reduce_sum_optimized(data.data(), n);
    float sum_par = reduce_sum_parallel(data.data(), n, 4);
    float sum_simd = reduce_sum_simd(data.data(), n);

    // In single precision float, summing 10,000 numbers accumulates order-of-summation
    // differences between sequential and parallel reductions. Bound within 1e-2f (~0.0002% relative error).
    EXPECT_NEAR(sum_opt, sum_ref, 1e-2f);
    EXPECT_NEAR(sum_par, sum_ref, 1e-2f);
    EXPECT_NEAR(sum_simd, sum_ref, 1e-2f);
}

TEST(ReductionsTest, MinAndMaxAreExact) {
    constexpr std::size_t n = 2049;
    AlignedVector<double> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = 1000.0 - static_cast<double>(i);
    }
    data[123] = -9999.0; // explicit min
    data[777] = 8888.0;  // explicit max

    auto min_ref = reduce_min_scalar(data.data(), n);
    auto min_opt = reduce_min_optimized(data.data(), n);
    auto min_par = reduce_min_parallel(data.data(), n, 4);
    auto min_simd = reduce_min_simd(data.data(), n);

    ASSERT_TRUE(min_ref.has_value());
    ASSERT_TRUE(min_opt.has_value());
    ASSERT_TRUE(min_par.has_value());
    ASSERT_TRUE(min_simd.has_value());

    EXPECT_DOUBLE_EQ(*min_opt, *min_ref);
    EXPECT_DOUBLE_EQ(*min_par, *min_ref);
    EXPECT_DOUBLE_EQ(*min_simd, *min_ref);
    EXPECT_DOUBLE_EQ(*min_ref, -9999.0);

    auto max_ref = reduce_max_scalar(data.data(), n);
    auto max_opt = reduce_max_optimized(data.data(), n);
    auto max_par = reduce_max_parallel(data.data(), n, 4);
    auto max_simd = reduce_max_simd(data.data(), n);

    ASSERT_TRUE(max_ref.has_value());
    ASSERT_TRUE(max_opt.has_value());
    ASSERT_TRUE(max_par.has_value());
    ASSERT_TRUE(max_simd.has_value());

    EXPECT_DOUBLE_EQ(*max_opt, *max_ref);
    EXPECT_DOUBLE_EQ(*max_par, *max_ref);
    EXPECT_DOUBLE_EQ(*max_simd, *max_ref);
    EXPECT_DOUBLE_EQ(*max_ref, 8888.0);
}

TEST(ReductionsTest, DotProductMatchesWithinTolerance) {
    constexpr std::size_t n = 5000;
    AlignedVector<float> a(n);
    AlignedVector<float> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.001f;
        b[i] = static_cast<float>(n - i) * 0.001f;
    }

    float dot_ref = dot_product_scalar(a.data(), b.data(), n);
    float dot_opt = dot_product_optimized(a.data(), b.data(), n);
    float dot_par = dot_product_parallel(a.data(), b.data(), n, 4);
    float dot_simd = dot_product_simd(a.data(), b.data(), n);

    EXPECT_NEAR(dot_opt, dot_ref, 1e-2f);
    EXPECT_NEAR(dot_par, dot_ref, 1e-2f);
    EXPECT_NEAR(dot_simd, dot_ref, 1e-2f);
}

TEST(ReductionsTest, MeanMatchesAccurately) {
    constexpr std::size_t n = 1000;
    AlignedVector<double> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<double>(i + 1);
    }
    // Expected mean of 1..1000 is 500.5
    auto mean_ref = mean_scalar(data.data(), n);
    auto mean_opt = mean_optimized(data.data(), n);
    auto mean_par = mean_parallel(data.data(), n, 4);
    auto mean_simd_val = mean_simd(data.data(), n);

    ASSERT_TRUE(mean_ref.has_value());
    ASSERT_TRUE(mean_opt.has_value());
    ASSERT_TRUE(mean_par.has_value());
    ASSERT_TRUE(mean_simd_val.has_value());

    EXPECT_NEAR(*mean_ref, 500.5, 1e-9);
    EXPECT_NEAR(*mean_opt, 500.5, 1e-9);
    EXPECT_NEAR(*mean_par, 500.5, 1e-9);
    EXPECT_NEAR(*mean_simd_val, 500.5, 1e-9);
}
