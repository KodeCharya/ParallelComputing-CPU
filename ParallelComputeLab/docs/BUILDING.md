# Building ParallelComputeLab

This document details the build prerequisites, CMake presets, toolchain requirements, and compile configurations for `ParallelComputeLab`.

---

## 1. Prerequisites and Toolchain

- **C++ Compiler**: GCC 11+ or Clang 13+ with full C++20 support (`std::span`, concepts, `std::allocator_traits`).
- **Build System**: CMake 3.20 or newer, Make or Ninja.
- **OpenMP**: `libgomp` (GCC) or `libomp` (Clang) for parallel loop execution.
- **Python (Offline Analysis)**: Python 3.9+ with optional `numpy` and `matplotlib` for generating charts.
- **GoogleTest & Google Benchmark**: Automatically acquired via CMake `FetchContent` (pinned tags: `v1.15.2` and `v1.9.0` respectively).

### Debian / Ubuntu Installation

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ libomp-dev python3 python3-pip python3-numpy python3-matplotlib
```

---

## 2. CMake Presets

`ParallelComputeLab` provides predefined build presets via `CMakePresets.json`:

| Preset | Build Type | Features Enabled | Primary Use Case |
|---|---|---|---|
| `release` | `Release` | `-O3`, OpenMP, AVX2/FMA, Tests, Benchmarks | Maximum performance benchmarking |
| `debug` | `Debug` | `-O0 -g`, Debug assertions, Tests | Debugging kernel behavior and step-through |
| `asan-ubsan` | `Debug` | `-fsanitize=address,undefined -g` | Memory safety, leak detection, UB checking |
| `tsan` | `RelWithDebInfo` | `-fsanitize=thread -g` | Concurrency race detection |
| `profiling` | `RelWithDebInfo` | `-fno-omit-frame-pointer -g -O3` | Linux `perf`, flamegraphs, and VTune |

---

## 3. Step-by-Step Build Commands

### A. Release Build (Recommended for Benchmarking)

```bash
# Configure
cmake -B build/release -S ParallelComputeLab -DCMAKE_BUILD_TYPE=Release

# Compile all targets using available CPU cores
cmake --build build/release -j$(nproc)
```

Generated binaries in `build/release/`:
- `parallel_compute_core.a` (Static library)
- `parallel_compute_tests` (Correctness test executable)
- `parallel_compute_benchmarks` (Google Benchmark executable)
- `system_info_cli` (Hardware and CPU diagnostics tool)

### B. AddressSanitizer and UndefinedBehaviorSanitizer Build

```bash
cmake -B build/asan -S ParallelComputeLab \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_ASAN_UBSAN=ON \
  -DBUILD_BENCHMARKS=OFF

cmake --build build/asan -j$(nproc)
./build/asan/parallel_compute_tests
```

### C. ThreadSanitizer Build

```bash
cmake -B build/tsan -S ParallelComputeLab \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DENABLE_TSAN=ON \
  -DBUILD_BENCHMARKS=OFF

cmake --build build/tsan -j$(nproc)
```

### D. Portable Fallback (Without Architecture-Specific SIMD)

```bash
cmake -B build/portable -S ParallelComputeLab \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_AVX2=OFF

cmake --build build/portable -j$(nproc)
ctest --test-dir build/portable --output-on-failure
```

---

## 4. Compiler Vectorization Diagnostics

To inspect compiler auto-vectorization decisions directly:

```bash
g++ -O3 -std=c++20 -IParallelComputeLab/include -fopt-info-vec-optimized \
  -c ParallelComputeLab/src/algorithms/elementwise.cpp -o /dev/null
```

To view generated AVX2 and FMA assembly instructions:

```bash
g++ -O3 -mavx2 -mfma -std=c++20 -IParallelComputeLab/include -DPARALLEL_COMPUTE_HAS_AVX2=1 \
  -S ParallelComputeLab/src/simd/simd_elementwise.cpp -o - | grep -E "vaddps|vmulps|vfmadd"
```
