#pragma once

#include <string>

namespace parallel_compute {

struct CpuFeatures {
    bool has_sse{false};
    bool has_sse2{false};
    bool has_sse3{false};
    bool has_ssse3{false};
    bool has_sse4_1{false};
    bool has_sse4_2{false};
    bool has_avx{false};
    bool has_avx2{false};
    bool has_fma{false};
    bool has_avx512f{false};
    bool has_avx512dq{false};
    bool has_avx512bw{false};
    bool has_avx512vl{false};
    bool has_neon{false};

    std::string vendor{};
    std::string brand{};

    static const CpuFeatures& get() noexcept;
    [[nodiscard]] std::string to_string() const;
};

} // namespace parallel_compute
