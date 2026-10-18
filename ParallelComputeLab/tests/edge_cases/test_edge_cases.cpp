#include <gtest/gtest.h>
#include "parallel_compute/algorithms/elementwise.hpp"
#include "parallel_compute/parallel/parallel_elementwise.hpp"
#include "parallel_compute/simd/simd_elementwise.hpp"
#include "parallel_compute/algorithms/reduction.hpp"
#include "parallel_compute/parallel/parallel_reduction.hpp"
#include "parallel_compute/simd/simd_reduction.hpp"
#include "parallel_compute/common/types.hpp"

#include <cmath>
#include <limits>

using namespace parallel_compute;

TEST(EdgeCasesTest, EmptyVectorDoesNotCrash) {
    float dummy = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    vector_add_scalar(&d1, &d2, &d3, 0);
    vector_add_optimized(&d1, &d2, &d3, 0);
    vector_add_parallel(&d1, &d2, &d3, 0, 2);
    vector_add_simd(&d1, &d2, &d3, 0);

    EXPECT_FLOAT_EQ(reduce_sum_scalar(&dummy, 0), 0.0f);
    EXPECT_FLOAT_EQ(reduce_sum_optimized(&dummy, 0), 0.0f);
    EXPECT_FLOAT_EQ(reduce_sum_parallel(&dummy, 0, 2), 0.0f);
    EXPECT_FLOAT_EQ(reduce_sum_simd(&dummy, 0), 0.0f);

    EXPECT_FALSE(reduce_min_scalar(&dummy, 0).has_value());
    EXPECT_FALSE(reduce_min_optimized(&dummy, 0).has_value());
    EXPECT_FALSE(reduce_min_parallel(&dummy, 0, 2).has_value());
    EXPECT_FALSE(reduce_min_simd(&dummy, 0).has_value());

    EXPECT_FALSE(reduce_max_scalar(&dummy, 0).has_value());
    EXPECT_FALSE(reduce_max_optimized(&dummy, 0).has_value());
    EXPECT_FALSE(reduce_max_parallel(&dummy, 0, 2).has_value());
    EXPECT_FALSE(reduce_max_simd(&dummy, 0).has_value());

    EXPECT_FLOAT_EQ(dot_product_scalar(&dummy, &dummy, 0), 0.0f);
    EXPECT_FLOAT_EQ(dot_product_optimized(&dummy, &dummy, 0), 0.0f);
    EXPECT_FLOAT_EQ(dot_product_parallel(&dummy, &dummy, 0, 2), 0.0f);
    EXPECT_FLOAT_EQ(dot_product_simd(&dummy, &dummy, 0), 0.0f);

    EXPECT_FALSE(mean_scalar(&dummy, 0).has_value());
    EXPECT_FALSE(mean_optimized(&dummy, 0).has_value());
    EXPECT_FALSE(mean_parallel(&dummy, 0, 2).has_value());
    EXPECT_FALSE(mean_simd(&dummy, 0).has_value());
}

TEST(EdgeCasesTest, SingleElementInputs) {
    float a = 3.0f;
    float b = 7.0f;
    float c = 0.0f;

    vector_add_simd(&a, &b, &c, 1);
    EXPECT_FLOAT_EQ(c, 10.0f);

    vector_mul_simd(&a, &b, &c, 1);
    EXPECT_FLOAT_EQ(c, 21.0f);

    EXPECT_FLOAT_EQ(reduce_sum_simd(&a, 1), 3.0f);
    EXPECT_FLOAT_EQ(*reduce_min_simd(&a, 1), 3.0f);
    EXPECT_FLOAT_EQ(*reduce_max_simd(&a, 1), 3.0f);
    EXPECT_DOUBLE_EQ(*mean_simd(&a, 1), 3.0);
}

TEST(EdgeCasesTest, HandlesNaNAndInfinity) {
    constexpr std::size_t n = 16;
    AlignedVector<float> vec(n, 1.0f);
    vec[5] = std::numeric_limits<float>::quiet_NaN();

    float sum = reduce_sum_scalar(vec.data(), n);
    EXPECT_TRUE(std::isnan(sum));

    vec[5] = std::numeric_limits<float>::infinity();
    float sum_inf = reduce_sum_scalar(vec.data(), n);
    EXPECT_TRUE(std::isinf(sum_inf));
}
