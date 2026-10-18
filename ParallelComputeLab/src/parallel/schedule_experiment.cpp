#include "parallel_compute/parallel/schedule_experiment.hpp"

#include <cmath>

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
#include <omp.h>
#endif

namespace parallel_compute {

namespace {

inline double compute_uniform(std::size_t i) noexcept {
    double x = static_cast<double>(i);
    // Standard floating point computation block
    return std::sin(x) * std::cos(x) + std::sqrt(x + 1.0);
}

inline double compute_imbalanced(std::size_t i) noexcept {
    double acc = 0.0;
    // Imbalanced iterations: iteration cost varies strongly with index
    const std::size_t iterations = (i % 256) + 1;
    for (std::size_t k = 0; k < iterations; ++k) {
        double v = static_cast<double>(i + k);
        acc += std::sin(v) * std::cos(v);
    }
    return acc;
}

} // namespace

double execute_schedule_workload(SchedulePolicy policy,
                                 WorkloadType workload,
                                 std::size_t n,
                                 std::size_t chunk_size,
                                 std::size_t num_threads) noexcept {
    double total = 0.0;
    if (chunk_size == 0) chunk_size = 1;

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    int threads = (num_threads > 0) ? static_cast<int>(num_threads) : omp_get_max_threads();
    int chunk = static_cast<int>(chunk_size);

    switch (policy) {
        case SchedulePolicy::Serial: {
            for (std::size_t i = 0; i < n; ++i) {
                total += (workload == WorkloadType::Uniform) ? compute_uniform(i) : compute_imbalanced(i);
            }
            break;
        }
        case SchedulePolicy::Static: {
            #pragma omp parallel for num_threads(threads) schedule(static, chunk) reduction(+:total)
            for (std::size_t i = 0; i < n; ++i) {
                total += (workload == WorkloadType::Uniform) ? compute_uniform(i) : compute_imbalanced(i);
            }
            break;
        }
        case SchedulePolicy::Dynamic: {
            #pragma omp parallel for num_threads(threads) schedule(dynamic, chunk) reduction(+:total)
            for (std::size_t i = 0; i < n; ++i) {
                total += (workload == WorkloadType::Uniform) ? compute_uniform(i) : compute_imbalanced(i);
            }
            break;
        }
        case SchedulePolicy::Guided: {
            #pragma omp parallel for num_threads(threads) schedule(guided, chunk) reduction(+:total)
            for (std::size_t i = 0; i < n; ++i) {
                total += (workload == WorkloadType::Uniform) ? compute_uniform(i) : compute_imbalanced(i);
            }
            break;
        }
    }
#else
    (void)num_threads;
    (void)chunk_size;
    (void)policy;
    for (std::size_t i = 0; i < n; ++i) {
        total += (workload == WorkloadType::Uniform) ? compute_uniform(i) : compute_imbalanced(i);
    }
#endif

    return total;
}

} // namespace parallel_compute
