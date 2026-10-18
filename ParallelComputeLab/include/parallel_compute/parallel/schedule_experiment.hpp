#pragma once

#include <cstddef>
#include <string_view>

namespace parallel_compute {

enum class SchedulePolicy {
    Serial,
    Static,
    Dynamic,
    Guided
};

enum class WorkloadType {
    Uniform,
    Imbalanced
};

[[nodiscard]] constexpr std::string_view schedule_policy_to_string(SchedulePolicy policy) noexcept {
    switch (policy) {
        case SchedulePolicy::Serial:  return "serial";
        case SchedulePolicy::Static:  return "static";
        case SchedulePolicy::Dynamic: return "dynamic";
        case SchedulePolicy::Guided:  return "guided";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view workload_type_to_string(WorkloadType type) noexcept {
    switch (type) {
        case WorkloadType::Uniform:    return "uniform";
        case WorkloadType::Imbalanced: return "imbalanced";
    }
    return "unknown";
}

double execute_schedule_workload(SchedulePolicy policy,
                                 WorkloadType workload,
                                 std::size_t n,
                                 std::size_t chunk_size = 1,
                                 std::size_t num_threads = 0) noexcept;

} // namespace parallel_compute
