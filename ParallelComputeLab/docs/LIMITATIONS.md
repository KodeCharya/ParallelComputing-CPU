# Hardware and Platform Limitations

This document transparently records known environment constraints, unsupported hardware instructions on the host machine, and observed measurement limitations.

---

## 1. Hardware Architecture Constraints

The primary benchmarking platform in this environment has the following specifications:
- **Processor**: Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz (Whiskey Lake)
- **Cores / Threads**: 4 physical cores, 8 logical threads (Intel Hyper-Threading)
- **Vector Extensions Supported**: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, FMA
- **Vector Extensions Unsupported**:
  - **AVX-512**: The host CPU does not implement the AVX-512 instruction set (AVX512F, AVX512DQ, etc.). Tests and benchmarks for AVX-512 are correctly omitted and safely handled via runtime CPU feature detection.
  - **ARM NEON**: Architecture is x86-64; ARM NEON intrinsics are portable in code structure but only active on AArch64 targets.

---

## 2. Profiling and Hardware Performance Counters (`perf`)

- **Linux `perf` Utility**:
  The `perf` command is not installed in this environment (`perf: command not found`). Additionally, in shared cloud container environments, access to hardware PMU (Performance Monitoring Unit) counters (`perf_event_open`) is typically restricted (`perf_event_paranoid >= 2`).
- **Mitigation & Alternative**:
  Timing measurements rely on `std::chrono::steady_clock` via Google Benchmark's sub-nanosecond monotonic timing infrastructure. Byte-counting and FLOPS throughput calculations are derived from verified mathematical models (STREAM standard) rather than raw PMU cycles.

---

## 3. External BLAS Baseline Availability

- **Status**: Optional dependency not pre-installed (`cblas.h: No such file or directory`).
- **Behavior**:
  CMake detects the absence of CBLAS during configuration and outputs:
  ```text
  -- External BLAS not found; baseline comparisons will be skipped gracefully.
  ```
  The library provides high-performance cache-blocked and AVX2-vectorized native kernels (`matmul_blocked`, `matmul_simd`, `matmul_parallel`) that execute unconditionally without external BLAS.

---

## 4. ThreadSanitizer with Uninstrumented System OpenMP (`libgomp`)

- **Observation**:
  Running `./build/tsan/parallel_compute_tests` completes and passes all 67 correctness tests, but outputs ThreadSanitizer warnings.
- **Root Cause Analysis**:
  The GNU OpenMP runtime library (`/usr/lib/x86_64-linux-gnu/libgomp.so.1`) shipped with standard Linux distributions is compiled without `-fsanitize=thread`. TSan cannot intercept internal synchronization futexes inside uninstrumented `libgomp`. When the main thread constructs a matrix buffer via `posix_memalign` and subsequently dispatches OpenMP worker threads via `#pragma omp parallel for`, TSan misses the happens-before relationship established by `libgomp`'s thread pool, falsely flagging a race between memory allocation and thread execution.
- **Verification**:
  Code review confirms that every OpenMP thread writes strictly to disjoint row indices ($bi \le i < bi + block\_m$), guaranteeing zero data races in the user code.
- **Resolution**:
  Running under Clang with a TSan-instrumented `libomp` eliminates these false positives. In GCC environments with stock `libgomp`, developers should be aware of this runtime interaction.

---

## 5. Numerical Reassociation in Parallel Reductions

- **Observation**:
  Summing $10^4$ floats in parallel yields a result differing from scalar summation by $\approx 0.00146$.
- **Explanation**:
  IEEE 754 floating-point addition is non-associative. Partitioning sums across 4 or 8 threads changes the summation tree order. This difference is within the theoretical roundoff bound ($10^4 \times \epsilon_{\text{single}} \approx 6 \times 10^{-4}$ per unit magnitude) and represents expected floating-point behavior, not a defect.
