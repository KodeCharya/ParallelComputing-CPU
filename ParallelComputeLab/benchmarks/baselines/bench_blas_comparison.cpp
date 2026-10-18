#include <benchmark/benchmark.h>
#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/common/types.hpp"

using namespace parallel_compute;

#if defined(PARALLEL_COMPUTE_HAS_BLAS)

static void BM_MatMul_BLAS(benchmark::State& state) {
    const auto N = static_cast<std::size_t>(state.range(0));
    Matrix<float> A(N, N, 1.0f);
    Matrix<float> B(N, N, 2.0f);
    Matrix<float> C(N, N, 0.0f);

    for (auto _ : state) {
        matmul_blas(A, B, C);
        benchmark::DoNotOptimize(C.data());
    }
    const int64_t flops = 2 * static_cast<int64_t>(N) * static_cast<int64_t>(N) * static_cast<int64_t>(N);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

BENCHMARK(BM_MatMul_BLAS)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

#else

static void BM_BLAS_Unavailable(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(0);
    }
}
BENCHMARK(BM_BLAS_Unavailable);

#endif
