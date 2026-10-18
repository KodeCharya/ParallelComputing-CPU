# ParallelComputeLab

A research-oriented C++20 framework for **CPU parallel computing, low-level optimization, SIMD vectorization, memory efficiency, and reproducible performance benchmarking**.

---

## 1. Project Overview and Philosophy

`ParallelComputeLab` provides an experimentally reproducible laboratory for implementing, testing, measuring, and profiling CPU algorithms across multiple optimization tiers:

1. **Level 1 (Algorithmic & Compiler)**: Portable scalar reference kernels, loop unrolling, pointer aliasing hints (`__restrict__`), vectorization directives (`#pragma GCC ivdep`), multi-accumulator dependency chain elimination.
2. **Level 2 (Memory Subsystem)**: Cache blocking ($M \times K \times N$ tiles), 64-byte cache-line aligned allocations (`AlignedVector`), STREAM memory workloads, strided access latency profiling.
3. **Level 3 (Thread-Level Parallelism)**: OpenMP multi-threaded parallel loops, reduction clauses, thread scaling experiments ($p \in \{1, 2, 4, 8\}$), scheduling policy comparisons (`static`, `dynamic`, `guided`).
4. **Level 4 (SIMD Vectorization)**: Explicit 256-bit AVX2 and FMA fused multiply-add intrinsics with automatic CPUID feature detection and safe portable fallbacks for non-AVX2 architectures.

---

## 2. Directory Layout

```text
ParallelComputeLab/
├── CMakeLists.txt                 # Main CMake build configuration
├── CMakePresets.json              # Presets for Release, Debug, ASan, TSan, Profiling
├── LICENSE                        # MIT License
├── README.md                      # Project documentation
├── include/
│   └── parallel_compute/
│       ├── common/                # AlignedVector, MatrixView, Matrix, Error, Version
│       ├── system/                # CPUID feature detection and cache topology
│       ├── memory/                # Alignment utilities, BandwidthCounter, MemoryWorkloads
│       ├── algorithms/            # Elementwise, Reduction, MatMul (Reference & Optimized)
│       ├── simd/                  # Explicit AVX2/FMA SIMD implementations
│       └── parallel/              # OpenMP parallel elementwise, reductions, and matmul
├── src/                           # Core C++ implementation sources
├── tests/
│   ├── unit/                      # Unit tests for system, memory, and elementwise ops
│   ├── numerical/                 # Numerical tolerance validation for reductions and matmul
│   ├── edge_cases/                # Empty arrays, single elements, NaNs/Infs, dimension errors
│   ├── concurrency/               # Thread scaling and OpenMP scheduling consistency
│   └── regression/                # Repeated execution determinism and vector tail checks
├── benchmarks/
│   ├── kernels/                   # Google Benchmark kernels (Elementwise, Reductions, MatMul)
│   ├── workloads/                 # STREAM memory workloads and OpenMP scheduling
│   └── baselines/                 # Optional BLAS baseline comparison
├── scripts/
│   ├── collect_system_info.py     # Dumps hardware, CPU, and compiler metadata
│   ├── run_benchmarks.py          # Drives benchmark suite and verifies output
│   ├── validate_results.py        # Schema and numeric integrity validator
│   └── analyze_results.py         # Statistical analysis, speedup/efficiency, and plots
├── configs/
│   └── benchmark_defaults.json    # Workload sizes and benchmark execution defaults
├── docs/
│   ├── BUILDING.md                # Build instructions and toolchain requirements
│   ├── TESTING.md                 # Test suite documentation and numerical tolerances
│   ├── BENCHMARKING.md            # Benchmark execution guide and CLI parameters
│   ├── PERFORMANCE_METHODOLOGY.md # Mathematical definitions and experimental findings
│   └── LIMITATIONS.md             # Platform constraints, perf status, and known limitations
└── results/
    ├── system_metadata.json       # Host CPU and environment metadata
    ├── raw_benchmark_results.json # Full benchmark measurements
    ├── summary_metrics.json       # Aggregated median, IQR, and speedups
    ├── summary_table.md           # Formatted performance tables
    └── *.png                      # Publication-quality charts (scaling, memory, workloads)
```

---

## 3. Quick Start: Build, Test, and Benchmark

### A. Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ libomp-dev python3 python3-numpy python3-matplotlib
```

### B. Configure and Build

```bash
# Configure Release build
cmake -B build/release -S ParallelComputeLab -DCMAKE_BUILD_TYPE=Release

# Compile all targets
cmake --build build/release -j$(nproc)
```

### C. Run Correctness Tests

```bash
# Run all 67 unit, numerical, edge-case, and concurrency tests
ctest --test-dir build/release --output-on-failure
```

### D. Execute Benchmarks and Generate Reports

```bash
# Run benchmarks and record hardware metadata
python3 ParallelComputeLab/scripts/run_benchmarks.py \
  --binary build/release/parallel_compute_benchmarks \
  --output-dir ParallelComputeLab/results \
  --repetitions 3 \
  --min-time 0.2

# Generate statistics, markdown summary, and scaling plots
python3 ParallelComputeLab/scripts/analyze_results.py \
  ParallelComputeLab/results/raw_benchmark_results.json \
  ParallelComputeLab/results
```

---

## 4. Key Experimental Findings on Host Hardware

Tested on: **Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz (4 physical cores, 8 threads)**:

1. **Parallel Overhead Threshold**: On small vectors ($N=1,024$, 4 KiB), OpenMP thread dispatch and synchronization causes a **$0.39\times$ slowdown** with 8 threads compared to single-thread scalar execution. Multithreading is only beneficial when per-thread work exceeds dispatch latency.
2. **Memory Bus Saturation**: SAXPY on 4,194,304 floats (16 MiB DRAM) saturates the memory bus (~15-20 GB/s) on 1 thread, achieving negligible scaling ($1.04\times$ on 4 threads, $0.94\times$ on 8 threads). Adding threads cannot overcome physical DRAM bandwidth limits.
3. **Compute Scaling Efficiency**: Floating-point `ReduceSum` on 1,048,576 floats achieves **$3.90\times$ speedup on 4 physical cores (97.6% efficiency)** and **$7.13\times$ speedup on 8 threads (89.1% efficiency)** due to high compute-to-memory ratio.
4. **Load Balancing via Scheduling**: On imbalanced loops, OpenMP `guided` and `dynamic` scheduling outperform `static` scheduling by **29%**, effectively distributing irregular chunks to idle cores.

---

## 5. Verification Status

| Component | Target / Suite | Verification Command | Status |
|---|---|---|---|
| **C++20 Core Library** | `parallel_compute_core` | `cmake --build build/release` | **PASS** |
| **Correctness Test Suite** | 67 tests in 14 test suites | `ctest --test-dir build/release` | **PASS** (67/67 passed) |
| **Address & UB Sanitizers** | `parallel_compute_tests` | `./build/asan/parallel_compute_tests` | **PASS** (0 leaks, 0 errors) |
| **ThreadSanitizer** | `parallel_compute_tests` | `./build/tsan/parallel_compute_tests` | **PASS** (Logic race-free; documented `libgomp` notes) |
| **Benchmark Suite** | 931 measurements | `run_benchmarks.py` | **PASS** |
| **Schema Validation** | Result JSON validation | `validate_results.py` | **PASS** |
| **Plot Generation** | Matplotlib analysis | `analyze_results.py` | **PASS** (4 plots generated) |
| **Portable Fallback** | Non-AVX2 configuration | `cmake -B build/portable -DENABLE_AVX2=OFF` | **PASS** |
| **Continuous Integration** | GitHub Actions | `.github/workflows/ci.yml` | **PASS** |

---

## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
