#include "parallel_compute/system/cpu_features.hpp"

#include <array>
#include <cstring>
#include <sstream>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#define PARALLEL_COMPUTE_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define PARALLEL_COMPUTE_ARCH_ARM64 1
#endif

namespace parallel_compute {

namespace {

#if defined(PARALLEL_COMPUTE_ARCH_X86)
void run_cpuid(unsigned int leaf, unsigned int subleaf, unsigned int regs[4]) {
#if defined(_MSC_VER)
    __cpuidex(reinterpret_cast<int*>(regs), static_cast<int>(leaf), static_cast<int>(subleaf));
#else
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#endif
}
#endif

CpuFeatures detect_cpu_features() {
    CpuFeatures f;

#if defined(PARALLEL_COMPUTE_ARCH_X86)
    unsigned int regs[4] = {0};

    // Get max leaf and vendor string
    run_cpuid(0, 0, regs);
    unsigned int max_leaf = regs[0];
    char vendor_buf[13] = {0};
    std::memcpy(vendor_buf, &regs[1], 4);
    std::memcpy(vendor_buf + 4, &regs[3], 4);
    std::memcpy(vendor_buf + 8, &regs[2], 4);
    f.vendor = vendor_buf;

    if (max_leaf >= 1) {
        run_cpuid(1, 0, regs);
        // edx
        f.has_sse   = (regs[3] & (1 << 25)) != 0;
        f.has_sse2  = (regs[3] & (1 << 26)) != 0;
        // ecx
        f.has_sse3   = (regs[2] & (1 << 0)) != 0;
        f.has_ssse3  = (regs[2] & (1 << 9)) != 0;
        f.has_fma    = (regs[2] & (1 << 12)) != 0;
        f.has_sse4_1 = (regs[2] & (1 << 19)) != 0;
        f.has_sse4_2 = (regs[2] & (1 << 20)) != 0;
        f.has_avx    = (regs[2] & (1 << 28)) != 0;
    }

    if (max_leaf >= 7) {
        run_cpuid(7, 0, regs);
        // ebx
        f.has_avx2     = (regs[1] & (1 << 5)) != 0;
        f.has_avx512f  = (regs[1] & (1 << 16)) != 0;
        f.has_avx512dq = (regs[1] & (1 << 17)) != 0;
        f.has_avx512bw = (regs[1] & (1 << 30)) != 0;
        f.has_avx512vl = (regs[1] & (1 << 31)) != 0;
    }

    // Extended function for CPU brand string
    run_cpuid(0x80000000, 0, regs);
    unsigned int max_ext_leaf = regs[0];
    if (max_ext_leaf >= 0x80000004) {
        char brand_buf[49] = {0};
        for (unsigned int i = 0; i < 3; ++i) {
            run_cpuid(0x80000002 + i, 0, regs);
            std::memcpy(brand_buf + i * 16, regs, 16);
        }
        f.brand = brand_buf;
        // Trim leading spaces if any
        size_t first = f.brand.find_first_not_of(' ');
        if (first != std::string::npos) {
            f.brand = f.brand.substr(first);
        }
    }
#elif defined(PARALLEL_COMPUTE_ARCH_ARM64)
    f.vendor = "ARM";
    f.brand = "AArch64 Compatible";
    f.has_neon = true;
#else
    f.vendor = "Generic";
    f.brand = "Generic CPU";
#endif

    return f;
}

} // namespace

const CpuFeatures& CpuFeatures::get() noexcept {
    static const CpuFeatures instance = detect_cpu_features();
    return instance;
}

std::string CpuFeatures::to_string() const {
    std::ostringstream oss;
    oss << "Vendor: " << vendor << "\n"
        << "Brand: " << brand << "\n"
        << "SSE: " << (has_sse ? "yes" : "no") << "\n"
        << "SSE2: " << (has_sse2 ? "yes" : "no") << "\n"
        << "SSE3: " << (has_sse3 ? "yes" : "no") << "\n"
        << "SSSE3: " << (has_ssse3 ? "yes" : "no") << "\n"
        << "SSE4.1: " << (has_sse4_1 ? "yes" : "no") << "\n"
        << "SSE4.2: " << (has_sse4_2 ? "yes" : "no") << "\n"
        << "AVX: " << (has_avx ? "yes" : "no") << "\n"
        << "AVX2: " << (has_avx2 ? "yes" : "no") << "\n"
        << "FMA: " << (has_fma ? "yes" : "no") << "\n"
        << "AVX512F: " << (has_avx512f ? "yes" : "no") << "\n"
        << "AVX512DQ: " << (has_avx512dq ? "yes" : "no") << "\n"
        << "AVX512BW: " << (has_avx512bw ? "yes" : "no") << "\n"
        << "AVX512VL: " << (has_avx512vl ? "yes" : "no") << "\n"
        << "NEON: " << (has_neon ? "yes" : "no");
    return oss.str();
}

} // namespace parallel_compute
