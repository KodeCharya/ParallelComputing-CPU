#include "parallel_compute/system/system_info.hpp"
#include "parallel_compute/common/version.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>

#if defined(__linux__)
#include <sys/utsname.h>
#endif

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
#include <omp.h>
#endif

namespace parallel_compute {

namespace {

std::size_t parse_cache_size(const std::string& str) {
    if (str.empty()) return 0;
    std::size_t num = 0;
    char unit = 0;
    std::istringstream iss(str);
    iss >> num >> unit;
    if (unit == 'K' || unit == 'k') {
        return num * 1024;
    } else if (unit == 'M' || unit == 'm') {
        return num * 1024 * 1024;
    } else if (unit == 'G' || unit == 'g') {
        return num * 1024 * 1024 * 1024;
    }
    return num;
}

CacheInfo detect_cache_topology() {
    CacheInfo info;
#if defined(__linux__)
    for (int i = 0; i < 6; ++i) {
        std::string base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(i) + "/";
        std::ifstream level_file(base + "level");
        std::ifstream type_file(base + "type");
        std::ifstream size_file(base + "size");

        if (!level_file.is_open() || !size_file.is_open()) {
            continue;
        }

        int level = 0;
        level_file >> level;

        std::string type;
        if (type_file.is_open()) {
            type_file >> type;
        }

        std::string size_str;
        size_file >> size_str;
        std::size_t bytes = parse_cache_size(size_str);

        if (level == 1) {
            if (type == "Data") {
                info.l1d_bytes = bytes;
            } else if (type == "Instruction") {
                info.l1i_bytes = bytes;
            }
        } else if (level == 2) {
            info.l2_bytes = bytes;
        } else if (level == 3) {
            info.l3_bytes = bytes;
        }
    }
#endif
    // Sensible fallbacks if sysfs is unavailable
    if (info.l1d_bytes == 0) info.l1d_bytes = 32 * 1024;
    if (info.l1i_bytes == 0) info.l1i_bytes = 32 * 1024;
    if (info.l2_bytes == 0)  info.l2_bytes = 256 * 1024;
    if (info.l3_bytes == 0)  info.l3_bytes = 6 * 1024 * 1024;

    return info;
}

std::size_t detect_physical_cores(std::size_t logical) {
#if defined(__linux__)
    std::set<std::pair<int, int>> core_socket_pairs;
    for (std::size_t cpu = 0; cpu < 256; ++cpu) {
        std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::ifstream core_file(base + "core_id");
        std::ifstream socket_file(base + "physical_package_id");
        if (core_file.is_open()) {
            int core = 0;
            int socket = 0;
            core_file >> core;
            if (socket_file.is_open()) {
                socket_file >> socket;
            }
            core_socket_pairs.insert({socket, core});
        } else {
            break;
        }
    }
    if (!core_socket_pairs.empty()) {
        return core_socket_pairs.size();
    }
#endif
    return std::max<std::size_t>(1, logical / 2);
}

SystemInfo detect_system_info() {
    SystemInfo info;
    info.features = CpuFeatures::get();
    info.cpu_model = info.features.brand.empty() ? "x86_64 CPU" : info.features.brand;
    info.cpu_vendor = info.features.vendor;

    info.logical_cores = std::thread::hardware_concurrency();
    if (info.logical_cores == 0) {
        info.logical_cores = 1;
    }
    info.physical_cores = detect_physical_cores(info.logical_cores);

#if defined(PARALLEL_COMPUTE_HAS_OPENMP)
    info.openmp_max_threads = static_cast<std::size_t>(omp_get_max_threads());
#else
    info.openmp_max_threads = 1;
#endif

    info.cache = detect_cache_topology();

#if defined(__linux__)
    struct utsname uts;
    if (uname(&uts) == 0) {
        info.os_name = std::string(uts.sysname) + " " + uts.release;
        info.architecture = uts.machine;
    } else {
        info.os_name = "Linux";
        info.architecture = "x86_64";
    }
#else
    info.os_name = "Generic OS";
    info.architecture = "Unknown";
#endif

    return info;
}

} // namespace

const SystemInfo& SystemInfo::get() noexcept {
    static const SystemInfo instance = detect_system_info();
    return instance;
}

std::string SystemInfo::to_json() const {
    std::ostringstream ss;
    ss << "{\n"
       << "  \"cpu_model\": \"" << cpu_model << "\",\n"
       << "  \"cpu_vendor\": \"" << cpu_vendor << "\",\n"
       << "  \"os_name\": \"" << os_name << "\",\n"
       << "  \"architecture\": \"" << architecture << "\",\n"
       << "  \"logical_cores\": " << logical_cores << ",\n"
       << "  \"physical_cores\": " << physical_cores << ",\n"
       << "  \"openmp_max_threads\": " << openmp_max_threads << ",\n"
       << "  \"cache\": {\n"
       << "    \"l1d_bytes\": " << cache.l1d_bytes << ",\n"
       << "    \"l1i_bytes\": " << cache.l1i_bytes << ",\n"
       << "    \"l2_bytes\": " << cache.l2_bytes << ",\n"
       << "    \"l3_bytes\": " << cache.l3_bytes << "\n"
       << "  },\n"
       << "  \"features\": {\n"
       << "    \"sse4_2\": " << (features.has_sse4_2 ? "true" : "false") << ",\n"
       << "    \"avx\": " << (features.has_avx ? "true" : "false") << ",\n"
       << "    \"avx2\": " << (features.has_avx2 ? "true" : "false") << ",\n"
       << "    \"fma\": " << (features.has_fma ? "true" : "false") << ",\n"
       << "    \"avx512f\": " << (features.has_avx512f ? "true" : "false") << ",\n"
       << "    \"neon\": " << (features.has_neon ? "true" : "false") << "\n"
       << "  },\n"
       << "  \"build\": {\n"
       << "    \"compiler_id\": \"" << kCompilerId << "\",\n"
       << "    \"compiler_version\": \"" << kCompilerVersion << "\",\n"
       << "    \"build_type\": \"" << kBuildType << "\"\n"
       << "  }\n"
       << "}";
    return ss.str();
}

std::string SystemInfo::to_string() const {
    std::ostringstream ss;
    ss << "CPU: " << cpu_model << " (" << cpu_vendor << ")\n"
       << "OS: " << os_name << " (" << architecture << ")\n"
       << "Cores: " << physical_cores << " physical, " << logical_cores << " logical\n"
       << "OpenMP Max Threads: " << openmp_max_threads << "\n"
       << "Caches:\n"
       << "  L1d: " << (cache.l1d_bytes / 1024) << " KiB\n"
       << "  L1i: " << (cache.l1i_bytes / 1024) << " KiB\n"
       << "  L2:  " << (cache.l2_bytes / 1024) << " KiB\n"
       << "  L3:  " << (cache.l3_bytes / (1024 * 1024)) << " MiB\n"
       << "Features: "
       << (features.has_avx2 ? "AVX2 " : "")
       << (features.has_fma ? "FMA " : "")
       << (features.has_sse4_2 ? "SSE4.2 " : "")
       << (features.has_avx512f ? "AVX-512 " : "")
       << (features.has_neon ? "NEON " : "");
    return ss.str();
}

} // namespace parallel_compute
