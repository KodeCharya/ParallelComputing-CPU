#include <benchmark/benchmark.h>
#include "parallel_compute/memory/memory_workloads.hpp"
#include "parallel_compute/memory/bandwidth_counter.hpp"
#include "parallel_compute/common/types.hpp"

using namespace parallel_compute;

static void BM_Memory_ArrayCopy(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> src(n, 1.0f);
    AlignedVector<float> dst(n, 0.0f);

    for (auto _ : state) {
        array_copy_serial(dst.data(), src.data(), n);
        benchmark::DoNotOptimize(dst.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::copy_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Memory_ArrayTriad(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> a(n, 0.0f);
    AlignedVector<float> b(n, 1.0f);
    AlignedVector<float> c(n, 2.0f);
    const float alpha = 3.0f;

    for (auto _ : state) {
        array_triad_serial(a.data(), b.data(), c.data(), alpha, n);
        benchmark::DoNotOptimize(a.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::triad_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Memory_SequentialRead(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    AlignedVector<float> data(n, 1.0f);

    for (auto _ : state) {
        float sum = sequential_read_serial(data.data(), n);
        benchmark::DoNotOptimize(sum);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BandwidthCounter::read_bytes<float>(n)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Memory_StridedRead(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto stride = static_cast<std::size_t>(state.range(1));
    AlignedVector<float> data(n, 1.0f);

    for (auto _ : state) {
        float sum = strided_read_serial(data.data(), n, stride);
        benchmark::DoNotOptimize(sum);
    }
    const std::size_t items = (n + stride - 1) / stride;
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(items * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(items));
}

// 4 KiB (L1), 64 KiB (L2), 1 MiB (L3), 16 MiB (RAM)
BENCHMARK(BM_Memory_ArrayCopy)->Arg(1024)->Arg(16384)->Arg(262144)->Arg(4194304);
BENCHMARK(BM_Memory_ArrayTriad)->Arg(1024)->Arg(16384)->Arg(262144)->Arg(4194304);
BENCHMARK(BM_Memory_SequentialRead)->Arg(1024)->Arg(16384)->Arg(262144)->Arg(4194304);

// Strided read across strides: 1, 2, 4, 8, 16, 32, 64 on 4M elements (16 MiB)
BENCHMARK(BM_Memory_StridedRead)
    ->Args({4194304, 1})
    ->Args({4194304, 2})
    ->Args({4194304, 4})
    ->Args({4194304, 8})
    ->Args({4194304, 16})
    ->Args({4194304, 32})
    ->Args({4194304, 64});
