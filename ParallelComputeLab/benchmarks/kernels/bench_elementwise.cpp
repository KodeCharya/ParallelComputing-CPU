#include <benchmark/benchmark.h>
#include "parallel_compute/algorithms/elementwise.hpp"
#include "parallel_compute/parallel/parallel_elementwise.hpp"
#include "parallel_compute/simd/simd_elementwise.hpp"
#include "parallel_compute/common/types.hpp"
#include "parallel_compute/memory/bandwidth_counter.hpp"

using namespace parallel_compute;

// --- Vector Add Benchmarks ---

static void BM_VectorAdd_Scalar(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> a(n, 1.5f);
    AlignedVector<float> b(n, 2.5f);
    AlignedVector<float> c(n, 0.0f);

    for (auto _ : state) {
        vector_add_scalar(a.data(), b.data(), c.data(), n);
        benchmark::DoNotOptimize(c.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::add_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_VectorAdd_Optimized(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> a(n, 1.5f);
    AlignedVector<float> b(n, 2.5f);
    AlignedVector<float> c(n, 0.0f);

    for (auto _ : state) {
        vector_add_optimized(a.data(), b.data(), c.data(), n);
        benchmark::DoNotOptimize(c.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::add_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_VectorAdd_Parallel(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto threads = static_cast<std::size_t>(state.range(1));
    AlignedVector<float> a(n, 1.5f);
    AlignedVector<float> b(n, 2.5f);
    AlignedVector<float> c(n, 0.0f);

    for (auto _ : state) {
        vector_add_parallel(a.data(), b.data(), c.data(), n, threads);
        benchmark::DoNotOptimize(c.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::add_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_VectorAdd_SIMD(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> a(n, 1.5f);
    AlignedVector<float> b(n, 2.5f);
    AlignedVector<float> c(n, 0.0f);

    for (auto _ : state) {
        vector_add_simd(a.data(), b.data(), c.data(), n);
        benchmark::DoNotOptimize(c.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::add_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

// --- SAXPY Benchmarks ---

static void BM_SAXPY_Scalar(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> x(n, 1.25f);
    AlignedVector<float> y(n, 2.75f);
    const float alpha = 3.14159f;

    for (auto _ : state) {
        saxpy_scalar(x.data(), y.data(), alpha, n);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::triad_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_SAXPY_Optimized(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> x(n, 1.25f);
    AlignedVector<float> y(n, 2.75f);
    const float alpha = 3.14159f;

    for (auto _ : state) {
        saxpy_optimized(x.data(), y.data(), alpha, n);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::triad_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_SAXPY_Parallel(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto threads = static_cast<std::size_t>(state.range(1));
    AlignedVector<float> x(n, 1.25f);
    AlignedVector<float> y(n, 2.75f);
    const float alpha = 3.14159f;

    for (auto _ : state) {
        saxpy_parallel(x.data(), y.data(), alpha, n, threads);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::triad_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_SAXPY_SIMD(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> x(n, 1.25f);
    AlignedVector<float> y(n, 2.75f);
    const float alpha = 3.14159f;

    for (auto _ : state) {
        saxpy_simd(x.data(), y.data(), alpha, n);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::triad_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

// Register benchmarks across sizes: 4 KiB (L1), 64 KiB (L2), 1 MiB (L3), 16 MiB (RAM)
BENCHMARK(BM_VectorAdd_Scalar)->Arg(1024)->Arg(16384)->Arg(262144)->Arg(4194304);
BENCHMARK(BM_VectorAdd_Optimized)->Arg(1024)->Arg(16384)->Arg(262144)->Arg(4194304);
BENCHMARK(BM_VectorAdd_SIMD)->Arg(1024)->Arg(16384)->Arg(262144)->Arg(4194304);

BENCHMARK(BM_VectorAdd_Parallel)->Args({1024, 1})->Args({1024, 2})->Args({1024, 4})->Args({1024, 8})
                                ->Args({262144, 1})->Args({262144, 2})->Args({262144, 4})->Args({262144, 8})
                                ->Args({4194304, 1})->Args({4194304, 2})->Args({4194304, 4})->Args({4194304, 8});

BENCHMARK(BM_SAXPY_Scalar)->Arg(1024)->Arg(16384)->Arg(262144)->Arg(4194304);
BENCHMARK(BM_SAXPY_Optimized)->Arg(1024)->Arg(16384)->Arg(262144)->Arg(4194304);
BENCHMARK(BM_SAXPY_SIMD)->Arg(1024)->Arg(16384)->Arg(262144)->Arg(4194304);

BENCHMARK(BM_SAXPY_Parallel)->Args({262144, 1})->Args({262144, 2})->Args({262144, 4})->Args({262144, 8})
                            ->Args({4194304, 1})->Args({4194304, 2})->Args({4194304, 4})->Args({4194304, 8});
