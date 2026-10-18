#include <gtest/gtest.h>
#include "parallel_compute/memory/alignment.hpp"
#include "parallel_compute/memory/bandwidth_counter.hpp"
#include "parallel_compute/memory/memory_workloads.hpp"
#include "parallel_compute/common/types.hpp"

using namespace parallel_compute;

TEST(AlignmentTest, ChecksAlignment) {
    alignas(64) float aligned_arr[16];
    EXPECT_TRUE(is_aligned(aligned_arr, 64));
    EXPECT_TRUE(is_aligned(aligned_arr, 32));
    EXPECT_TRUE(is_aligned(aligned_arr, 16));

    const float* unaligned = aligned_arr + 1;
    EXPECT_FALSE(is_aligned(unaligned, 64));
}

TEST(AlignedVectorTest, AllocatesAlignedMemory) {
    AlignedVector<float, 64> vec(1024, 1.0f);
    EXPECT_EQ(vec.size(), 1024u);
    EXPECT_TRUE(is_aligned(vec.data(), 64));

    for (std::size_t i = 0; i < vec.size(); ++i) {
        EXPECT_FLOAT_EQ(vec[i], 1.0f);
    }
}

TEST(MatrixTest, ManagesRowMajorStorage) {
    Matrix<float> mat(4, 5, 2.5f);
    EXPECT_EQ(mat.rows(), 4u);
    EXPECT_EQ(mat.cols(), 5u);
    EXPECT_EQ(mat.size(), 20u);

    EXPECT_FLOAT_EQ(mat(0, 0), 2.5f);
    EXPECT_FLOAT_EQ(mat(3, 4), 2.5f);

    mat(2, 3) = 42.0f;
    EXPECT_FLOAT_EQ(mat(2, 3), 42.0f);

    auto view = mat.view();
    EXPECT_FLOAT_EQ(view(2, 3), 42.0f);
    EXPECT_THROW((void)mat(4, 0), DimensionMismatchError);
}

TEST(BandwidthCounterTest, CalculatesStandardSTREAMBytes) {
    constexpr std::size_t n = 1000;
    // float is 4 bytes
    EXPECT_EQ(BandwidthCounter::copy_bytes<float>(n), 2 * 1000 * sizeof(float));
    EXPECT_EQ(BandwidthCounter::scale_bytes<float>(n), 2 * 1000 * sizeof(float));
    EXPECT_EQ(BandwidthCounter::triad_bytes<float>(n), 3 * 1000 * sizeof(float));
    EXPECT_EQ(BandwidthCounter::add_bytes<float>(n), 3 * 1000 * sizeof(float));

    // 10 GB transferred in 1 second -> 10.0 GB/s
    double bw = BandwidthCounter::gigabytes_per_second(10'000'000'000ULL, 1.0);
    EXPECT_DOUBLE_EQ(bw, 10.0);
}

TEST(MemoryWorkloadsTest, VerifiesWorkloadsCorrectness) {
    constexpr std::size_t n = 256;
    AlignedVector<float> a(n, 1.0f);
    AlignedVector<float> b(n, 2.0f);
    AlignedVector<float> c(n, 3.0f);
    AlignedVector<float> dst(n, 0.0f);

    array_copy_serial(dst.data(), a.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], 1.0f);
    }

    array_scale_serial(dst.data(), a.data(), 5.0f, n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], 5.0f);
    }

    array_triad_serial(dst.data(), b.data(), c.data(), 4.0f, n);
    for (std::size_t i = 0; i < n; ++i) {
        // b + 4 * c = 2.0 + 4 * 3.0 = 14.0
        EXPECT_FLOAT_EQ(dst[i], 14.0f);
    }

    float sum = sequential_read_serial(a.data(), n);
    EXPECT_FLOAT_EQ(sum, 256.0f);

    float strided_sum = strided_read_serial(a.data(), n, 2);
    EXPECT_FLOAT_EQ(strided_sum, 128.0f);
}
