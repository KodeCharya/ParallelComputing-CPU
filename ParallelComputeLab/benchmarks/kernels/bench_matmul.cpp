#include <benchmark/benchmark.h>
#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/parallel/parallel_matmul.hpp"
#include "parallel_compute/simd/simd_matmul.hpp"
#include "parallel_compute/common/types.hpp"

using namespace parallel_compute;

static void BM_MatMul_Reference(benchmark::State& state) {
    const auto N = static_cast<std::size_t>(state.range(0));
    Matrix<float> A(N, N, 1.0f);
    Matrix<float> B(N, N, 2.0f);
    Matrix<float> C(N, N, 0.0f);

    for (auto _ : state) {
        matmul_reference(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    const int64_t flops = 2 * static_cast<int64_t>(N) * static_cast<int64_t>(N) * static_cast<int64_t>(N);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

static void BM_MatMul_Blocked(benchmark::State& state) {
    const auto N = static_cast<std::size_t>(state.range(0));
    Matrix<float> A(N, N, 1.0f);
    Matrix<float> B(N, N, 2.0f);
    Matrix<float> C(N, N, 0.0f);

    for (auto _ : state) {
        matmul_blocked(A, B, C, 64, 64, 64);
        benchmark::DoNotOptimize(C.data());
    }
    const int64_t flops = 2 * static_cast<int64_t>(N) * static_cast<int64_t>(N) * static_cast<int64_t>(N);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

static void BM_MatMul_SIMD(benchmark::State& state) {
    const auto N = static_cast<std::size_t>(state.range(0));
    Matrix<float> A(N, N, 1.0f);
    Matrix<float> B(N, N, 2.0f);
    Matrix<float> C(N, N, 0.0f);

    for (auto _ : state) {
        matmul_simd(A, B, C, 64, 64, 64);
        benchmark::DoNotOptimize(C.data());
    }
    const int64_t flops = 2 * static_cast<int64_t>(N) * static_cast<int64_t>(N) * static_cast<int64_t>(N);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

static void BM_MatMul_Parallel(benchmark::State& state) {
    const auto N = static_cast<std::size_t>(state.range(0));
    const auto threads = static_cast<std::size_t>(state.range(1));
    Matrix<float> A(N, N, 1.0f);
    Matrix<float> B(N, N, 2.0f);
    Matrix<float> C(N, N, 0.0f);

    for (auto _ : state) {
        matmul_parallel(A, B, C, threads, 64, 64, 64);
        benchmark::DoNotOptimize(C.data());
    }
    const int64_t flops = 2 * static_cast<int64_t>(N) * static_cast<int64_t>(N) * static_cast<int64_t>(N);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

// 64x64, 128x128, 256x256
BENCHMARK(BM_MatMul_Reference)->Arg(64)->Arg(128)->Arg(256);
BENCHMARK(BM_MatMul_Blocked)->Arg(64)->Arg(128)->Arg(256)->Arg(512);
BENCHMARK(BM_MatMul_SIMD)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK(BM_MatMul_Parallel)->Args({128, 1})->Args({128, 2})->Args({128, 4})->Args({128, 8})
                             ->Args({256, 1})->Args({256, 2})->Args({256, 4})->Args({256, 8})
                             ->Args({512, 1})->Args({512, 2})->Args({512, 4})->Args({512, 8});
