cmake_minimum_required(VERSION 3.20)

project(
    ParallelComputeLab
    VERSION 0.1.0
    DESCRIPTION "CPU Parallel Computing Optimization and Benchmarking Framework"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_TESTS "Build correctness and unit tests" ON)
option(BUILD_BENCHMARKS "Build Google Benchmark performance suite" ON)
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(ENABLE_AVX2 "Enable AVX2/FMA vectorization on supported x86_64 targets" ON)
option(ENABLE_BLAS "Enable optional external BLAS comparison" ON)
option(ENABLE_ASAN_UBSAN "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Build with ThreadSanitizer" OFF)
option(ENABLE_PROFILING_FLAGS "Enable frame pointers and debugging symbols for profiling" OFF)

# Sanity checks for sanitizers
if(ENABLE_ASAN_UBSAN AND ENABLE_TSAN)
    message(FATAL_ERROR "ASan/UBSan and TSan cannot be enabled simultaneously.")
endif()

# Compiler warning flags
set(PROJECT_WARNINGS)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND PROJECT_WARNINGS
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wformat=2
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND PROJECT_WARNINGS -Wlogical-op)
    endif()
endif()

# Sanitizers
set(SANITIZER_FLAGS)
if(ENABLE_ASAN_UBSAN)
    list(APPEND SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -g)
endif()
if(ENABLE_TSAN)
    list(APPEND SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer -g)
endif()
if(ENABLE_PROFILING_FLAGS)
    list(APPEND SANITIZER_FLAGS -fno-omit-frame-pointer -g)
endif()

# OpenMP Configuration
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP CXX found: ${OpenMP_CXX_VERSION}")
        add_compile_definitions(PARALLEL_COMPUTE_HAS_OPENMP=1)
    endif()
endif()

# Optional BLAS Configuration
set(BLAS_LIBRARIES_TO_LINK)
if(ENABLE_BLAS)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        # Check if cblas header can be included
        include(CheckIncludeFileCXX)
        check_include_file_cxx("cblas.h" HAS_CBLAS_H)
        if(HAS_CBLAS_H)
            message(STATUS "CBLAS header and BLAS library found: ${BLAS_LIBRARIES}")
            add_compile_definitions(PARALLEL_COMPUTE_HAS_BLAS=1)
            set(BLAS_LIBRARIES_TO_LINK ${BLAS_LIBRARIES})
        else()
            message(STATUS "BLAS library found but cblas.h missing; BLAS comparisons disabled.")
        endif()
    else()
        message(STATUS "External BLAS not found; baseline comparisons will be skipped gracefully.")
    endif()
endif()

# Core Library Sources
set(CORE_SOURCES
    src/common/error.cpp
    src/system/cpu_features.cpp
    src/system/system_info.cpp
    src/memory/memory_workloads.cpp
    src/algorithms/elementwise.cpp
    src/algorithms/reduction.cpp
    src/algorithms/matmul.cpp
    src/parallel/parallel_elementwise.cpp
    src/parallel/parallel_reduction.cpp
    src/parallel/parallel_matmul.cpp
    src/parallel/schedule_experiment.cpp
    src/simd/simd_elementwise.cpp
    src/simd/simd_reduction.cpp
    src/simd/simd_matmul.cpp
)

add_library(parallel_compute_core STATIC ${CORE_SOURCES})

target_include_directories(parallel_compute_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(parallel_compute_core PUBLIC cxx_std_20)
target_compile_options(parallel_compute_core PRIVATE ${PROJECT_WARNINGS} ${SANITIZER_FLAGS})
target_link_options(parallel_compute_core PUBLIC ${SANITIZER_FLAGS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(parallel_compute_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(BLAS_LIBRARIES_TO_LINK)
    target_link_libraries(parallel_compute_core PRIVATE ${BLAS_LIBRARIES_TO_LINK})
endif()

# SIMD specialized compiler flags on x86_64
if(ENABLE_AVX2 AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"))
    message(STATUS "Configuring AVX2/FMA for SIMD translation units")
    set_source_files_properties(
        src/simd/simd_elementwise.cpp
        src/simd/simd_reduction.cpp
        src/simd/simd_matmul.cpp
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )
    target_compile_definitions(parallel_compute_core PUBLIC PARALLEL_COMPUTE_HAS_AVX2=1)
endif()

# Record build metadata definitions
target_compile_definitions(parallel_compute_core PUBLIC
    PARALLEL_COMPUTE_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    PARALLEL_COMPUTE_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    PARALLEL_COMPUTE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
)

# System info CLI tool
add_executable(system_info_cli src/system/main.cpp)
target_link_libraries(system_info_cli PRIVATE parallel_compute_core)
target_compile_options(system_info_cli PRIVATE ${PROJECT_WARNINGS} ${SANITIZER_FLAGS})
target_link_options(system_info_cli PRIVATE ${SANITIZER_FLAGS})

# GoogleTest and Testing Target
if(BUILD_TESTS)
    include(CTest)
    include(FetchContent)
    message(STATUS "Fetching GoogleTest v1.15.2...")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    set(TEST_SOURCES
        tests/unit/test_cpu_features.cpp
        tests/unit/test_memory.cpp
        tests/unit/test_elementwise.cpp
        tests/numerical/test_reductions.cpp
        tests/numerical/test_matmul.cpp
        tests/edge_cases/test_edge_cases.cpp
        tests/edge_cases/test_dimension_validation.cpp
        tests/concurrency/test_concurrency.cpp
        tests/regression/test_regression.cpp
    )

    add_executable(parallel_compute_tests ${TEST_SOURCES})
    target_link_libraries(parallel_compute_tests
        PRIVATE
            parallel_compute_core
            GTest::gtest_main
    )
    target_compile_options(parallel_compute_tests PRIVATE ${PROJECT_WARNINGS} ${SANITIZER_FLAGS})
    target_link_options(parallel_compute_tests PRIVATE ${SANITIZER_FLAGS})

    include(GoogleTest)
    gtest_discover_tests(parallel_compute_tests)
endif()

# Google Benchmark Target
if(BUILD_BENCHMARKS)
    include(FetchContent)
    message(STATUS "Fetching Google Benchmark v1.9.0...")
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.0
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)

    set(BENCHMARK_SOURCES
        benchmarks/main.cpp
        benchmarks/kernels/bench_elementwise.cpp
        benchmarks/kernels/bench_reduction.cpp
        benchmarks/kernels/bench_matmul.cpp
        benchmarks/workloads/bench_memory.cpp
        benchmarks/workloads/bench_scheduling.cpp
        benchmarks/baselines/bench_blas_comparison.cpp
    )

    add_executable(parallel_compute_benchmarks ${BENCHMARK_SOURCES})
    target_link_libraries(parallel_compute_benchmarks
        PRIVATE
            parallel_compute_core
            benchmark::benchmark
    )
    target_compile_options(parallel_compute_benchmarks PRIVATE ${PROJECT_WARNINGS} ${SANITIZER_FLAGS})
    target_link_options(parallel_compute_benchmarks PRIVATE ${SANITIZER_FLAGS})
endif()
