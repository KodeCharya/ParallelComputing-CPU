#pragma once

#include "parallel_compute/system/cpu_features.hpp"

#include <cstddef>
#include <string>

namespace parallel_compute {

struct CacheInfo {
    std::size_t l1d_bytes{0};
    std::size_t l1i_bytes{0};
    std::size_t l2_bytes{0};
    std::size_t l3_bytes{0};
};

struct SystemInfo {
    std::string cpu_model{};
    std::string cpu_vendor{};
    std::string os_name{};
    std::string architecture{};
    std::size_t logical_cores{0};
    std::size_t physical_cores{0};
    std::size_t openmp_max_threads{0};
    CacheInfo cache{};
    CpuFeatures features{};

    static const SystemInfo& get() noexcept;
    [[nodiscard]] std::string to_json() const;
    [[nodiscard]] std::string to_string() const;
};

} // namespace parallel_compute
