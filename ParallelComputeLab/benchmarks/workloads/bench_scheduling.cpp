#include <benchmark/benchmark.h>
#include "parallel_compute/parallel/schedule_experiment.hpp"

using namespace parallel_compute;

static void BM_Schedule_Uniform(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto policy_idx = static_cast<int>(state.range(1));
    const auto chunk = static_cast<std::size_t>(state.range(2));
    const auto threads = static_cast<std::size_t>(state.range(3));

    const SchedulePolicy policy = static_cast<SchedulePolicy>(policy_idx);

    for (auto _ : state) {
        double res = execute_schedule_workload(policy, WorkloadType::Uniform, n, chunk, threads);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Schedule_Imbalanced(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto policy_idx = static_cast<int>(state.range(1));
    const auto chunk = static_cast<std::size_t>(state.range(2));
    const auto threads = static_cast<std::size_t>(state.range(3));

    const SchedulePolicy policy = static_cast<SchedulePolicy>(policy_idx);

    for (auto _ : state) {
        double res = execute_schedule_workload(policy, WorkloadType::Imbalanced, n, chunk, threads);
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

// Policies: 0: Serial, 1: Static, 2: Dynamic, 3: Guided
// Compare on 32768 iterations
BENCHMARK(BM_Schedule_Uniform)
    ->Args({32768, 0, 1, 1})    // Serial
    ->Args({32768, 1, 1, 4})    // Static chunk 1
    ->Args({32768, 1, 32, 4})   // Static chunk 32
    ->Args({32768, 2, 1, 4})    // Dynamic chunk 1
    ->Args({32768, 2, 32, 4})   // Dynamic chunk 32
    ->Args({32768, 3, 32, 4});  // Guided chunk 32

BENCHMARK(BM_Schedule_Imbalanced)
    ->Args({32768, 0, 1, 1})    // Serial
    ->Args({32768, 1, 1, 4})    // Static chunk 1
    ->Args({32768, 1, 32, 4})   // Static chunk 32
    ->Args({32768, 2, 1, 4})    // Dynamic chunk 1
    ->Args({32768, 2, 32, 4})   // Dynamic chunk 32
    ->Args({32768, 3, 32, 4});  // Guided chunk 32
