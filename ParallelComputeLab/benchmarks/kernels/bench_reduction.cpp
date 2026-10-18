#include <benchmark/benchmark.h>
#include "parallel_compute/algorithms/reduction.hpp"
#include "parallel_compute/parallel/parallel_reduction.hpp"
#include "parallel_compute/simd/simd_reduction.hpp"
#include "parallel_compute/common/types.hpp"
#include "parallel_compute/memory/bandwidth_counter.hpp"

using namespace parallel_compute;

// --- Sum Reduction Benchmarks ---

static void BM_ReduceSum_Scalar(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> data(n, 1.0f);

    for (auto _ : state) {
        float sum = reduce_sum_scalar(data.data(), n);
        benchmark::DoNotOptimize(sum);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::read_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_ReduceSum_Optimized(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> data(n, 1.0f);

    for (auto _ : state) {
        float sum = reduce_sum_optimized(data.data(), n);
        benchmark::DoNotOptimize(sum);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::read_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_ReduceSum_Parallel(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto threads = static_cast<std::size_t>(state.range(1));
    AlignedVector<float> data(n, 1.0f);

    for (auto _ : state) {
        float sum = reduce_sum_parallel(data.data(), n, threads);
        benchmark::DoNotOptimize(sum);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::read_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_ReduceSum_SIMD(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> data(n, 1.0f);

    for (auto _ : state) {
        float sum = reduce_sum_simd(data.data(), n);
        benchmark::DoNotOptimize(sum);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::read_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

// --- Dot Product Benchmarks ---

static void BM_DotProduct_Scalar(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> a(n, 1.0f);
    AlignedVector<float> b(n, 2.0f);

    for (auto _ : state) {
        float dot = dot_product_scalar(a.data(), b.data(), n);
        benchmark::DoNotOptimize(dot);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(2 * n * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_DotProduct_Optimized(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> a(n, 1.0f);
    AlignedVector<float> b(n, 2.0f);

    for (auto _ : state) {
        float dot = dot_product_optimized(a.data(), b.data(), n);
        benchmark::DoNotOptimize(dot);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(2 * n * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_DotProduct_Parallel(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto threads = static_cast<std::size_t>(state.range(1));
    AlignedVector<float> a(n, 1.0f);
    AlignedVector<float> b(n, 2.0f);

    for (auto _ : state) {
        float dot = dot_product_parallel(a.data(), b.data(), n, threads);
        benchmark::DoNotOptimize(dot);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(2 * n * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_DotProduct_SIMD(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> a(n, 1.0f);
    AlignedVector<float> b(n, 2.0f);

    for (auto _ : state) {
        float dot = dot_product_simd(a.data(), b.data(), n);
        benchmark::DoNotOptimize(dot);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(2 * n * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

BENCHMARK(BM_ReduceSum_Scalar)->Arg(1024)->Arg(65536)->Arg(1048576);
BENCHMARK(BM_ReduceSum_Optimized)->Arg(1024)->Arg(65536)->Arg(1048576);
BENCHMARK(BM_ReduceSum_SIMD)->Arg(1024)->Arg(65536)->Arg(1048576);

BENCHMARK(BM_ReduceSum_Parallel)->Args({65536, 1})->Args({65536, 2})->Args({65536, 4})->Args({65536, 8})
                               ->Args({1048576, 1})->Args({1048576, 2})->Args({1048576, 4})->Args({1048576, 8});

BENCHMARK(BM_DotProduct_Scalar)->Arg(1024)->Arg(65536)->Arg(1048576);
BENCHMARK(BM_DotProduct_Optimized)->Arg(1024)->Arg(65536)->Arg(1048576);
BENCHMARK(BM_DotProduct_SIMD)->Arg(1024)->Arg(65536)->Arg(1048576);

BENCHMARK(BM_DotProduct_Parallel)->Args({65536, 1})->Args({65536, 2})->Args({65536, 4})->Args({65536, 8})
                                 ->Args({1048576, 1})->Args({1048576, 2})->Args({1048576, 4})->Args({1048576, 8});
