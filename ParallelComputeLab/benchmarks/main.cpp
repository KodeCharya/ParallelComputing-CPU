#include <benchmark/benchmark.h>
#include "parallel_compute/system/system_info.hpp"
#include "parallel_compute/common/version.hpp"
#include <iostream>

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);

    const auto& sys = parallel_compute::SystemInfo::get();

    // Register custom context metadata for JSON export
    ::benchmark::AddCustomContext("cpu_model", sys.cpu_model);
    ::benchmark::AddCustomContext("cpu_vendor", sys.cpu_vendor);
    ::benchmark::AddCustomContext("os_name", sys.os_name);
    ::benchmark::AddCustomContext("architecture", sys.architecture);
    ::benchmark::AddCustomContext("logical_cores", std::to_string(sys.logical_cores));
    ::benchmark::AddCustomContext("physical_cores", std::to_string(sys.physical_cores));
    ::benchmark::AddCustomContext("openmp_max_threads", std::to_string(sys.openmp_max_threads));
    ::benchmark::AddCustomContext("l1d_cache_bytes", std::to_string(sys.cache.l1d_bytes));
    ::benchmark::AddCustomContext("l2_cache_bytes", std::to_string(sys.cache.l2_bytes));
    ::benchmark::AddCustomContext("l3_cache_bytes", std::to_string(sys.cache.l3_bytes));
    ::benchmark::AddCustomContext("compiler_id", std::string(parallel_compute::kCompilerId));
    ::benchmark::AddCustomContext("compiler_version", std::string(parallel_compute::kCompilerVersion));
    ::benchmark::AddCustomContext("build_type", std::string(parallel_compute::kBuildType));
    ::benchmark::AddCustomContext("has_avx2", sys.features.has_avx2 ? "true" : "false");
    ::benchmark::AddCustomContext("has_fma", sys.features.has_fma ? "true" : "false");

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    std::cout << "========================================================\n"
              << "ParallelComputeLab Benchmark Suite\n"
              << "--------------------------------------------------------\n"
              << sys.to_string() << "\n"
              << "Compiler: " << parallel_compute::kCompilerId << " "
              << parallel_compute::kCompilerVersion << " (" << parallel_compute::kBuildType << ")\n"
              << "========================================================\n\n";

    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
