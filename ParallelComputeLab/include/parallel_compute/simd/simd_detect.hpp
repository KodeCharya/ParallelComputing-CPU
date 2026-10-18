#pragma once

#include "parallel_compute/system/cpu_features.hpp"

namespace parallel_compute {

[[nodiscard]] inline bool is_avx2_supported() noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    return CpuFeatures::get().has_avx2;
#else
    return false;
#endif
}

[[nodiscard]] inline bool is_fma_supported() noexcept {
#if defined(PARALLEL_COMPUTE_HAS_AVX2)
    return CpuFeatures::get().has_fma;
#else
    return false;
#endif
}

} // namespace parallel_compute
