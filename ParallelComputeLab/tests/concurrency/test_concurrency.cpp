#include <gtest/gtest.h>
#include "parallel_compute/parallel/parallel_elementwise.hpp"
#include "parallel_compute/parallel/parallel_reduction.hpp"
#include "parallel_compute/parallel/parallel_matmul.hpp"
#include "parallel_compute/parallel/schedule_experiment.hpp"
#include "parallel_compute/common/types.hpp"

using namespace parallel_compute;

TEST(ConcurrencyTest, ThreadScalingReductionsYieldIdenticalResults) {
    constexpr std::size_t n = 20000;
    AlignedVector<double> vec(n);
    for (std::size_t i = 0; i < n; ++i) {
        vec[i] = static_cast<double>(i) * 0.05;
    }

    double res1 = reduce_sum_parallel(vec.data(), n, 1);
    double res2 = reduce_sum_parallel(vec.data(), n, 2);
    double res4 = reduce_sum_parallel(vec.data(), n, 4);
    double res8 = reduce_sum_parallel(vec.data(), n, 8);

    // Sum reduction can have minor floating point reassociation differences, bound within 1e-7
    EXPECT_NEAR(res2, res1, 1e-7);
    EXPECT_NEAR(res4, res1, 1e-7);
    EXPECT_NEAR(res8, res1, 1e-7);

    // Min and max must be identical regardless of thread count
    auto min1 = reduce_min_parallel(vec.data(), n, 1);
    auto min4 = reduce_min_parallel(vec.data(), n, 4);
    auto min8 = reduce_min_parallel(vec.data(), n, 8);
    EXPECT_DOUBLE_EQ(*min4, *min1);
    EXPECT_DOUBLE_EQ(*min8, *min1);
}

TEST(ConcurrencyTest, ThreadScalingMatmulYieldsIdenticalResults) {
    constexpr std::size_t M = 32, K = 32, N = 32;
    Matrix<float> A(M, K, 1.5f);
    Matrix<float> B(K, N, 2.0f);
    Matrix<float> C1(M, N);
    Matrix<float> C2(M, N);
    Matrix<float> C4(M, N);

    matmul_parallel(A, B, C1, 1, 16, 16, 16);
    matmul_parallel(A, B, C2, 2, 16, 16, 16);
    matmul_parallel(A, B, C4, 4, 16, 16, 16);

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            EXPECT_FLOAT_EQ(C2(i, j), C1(i, j));
            EXPECT_FLOAT_EQ(C4(i, j), C1(i, j));
        }
    }
}

TEST(ConcurrencyTest, SchedulingPoliciesProduceConsistentResults) {
    constexpr std::size_t n = 2048;

    double res_serial = execute_schedule_workload(SchedulePolicy::Serial, WorkloadType::Uniform, n, 1, 1);
    double res_static = execute_schedule_workload(SchedulePolicy::Static, WorkloadType::Uniform, n, 16, 4);
    double res_dyn    = execute_schedule_workload(SchedulePolicy::Dynamic, WorkloadType::Uniform, n, 16, 4);
    double res_guided = execute_schedule_workload(SchedulePolicy::Guided, WorkloadType::Uniform, n, 16, 4);

    EXPECT_NEAR(res_static, res_serial, 1e-7);
    EXPECT_NEAR(res_dyn,    res_serial, 1e-7);
    EXPECT_NEAR(res_guided, res_serial, 1e-7);

    // Test imbalanced workload across policies
    double imb_serial = execute_schedule_workload(SchedulePolicy::Serial, WorkloadType::Imbalanced, n, 1, 1);
    double imb_static = execute_schedule_workload(SchedulePolicy::Static, WorkloadType::Imbalanced, n, 8, 4);
    double imb_dyn    = execute_schedule_workload(SchedulePolicy::Dynamic, WorkloadType::Imbalanced, n, 8, 4);
    double imb_guided = execute_schedule_workload(SchedulePolicy::Guided, WorkloadType::Imbalanced, n, 8, 4);

    EXPECT_NEAR(imb_static, imb_serial, 1e-6);
    EXPECT_NEAR(imb_dyn,    imb_serial, 1e-6);
    EXPECT_NEAR(imb_guided, imb_serial, 1e-6);
}
