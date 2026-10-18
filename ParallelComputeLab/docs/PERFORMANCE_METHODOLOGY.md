# Performance and Optimization Methodology

This document outlines the scientific benchmarking controls, metrics definitions, and optimization principles adopted throughout `ParallelComputeLab`.

---

## 1. Metrics and Mathematical Definitions

### A. Speedup
Parallel speedup $S(p)$ measures the execution rate improvement when using $p$ active threads relative to a single thread executing the exact same algorithm:

$$S(p) = \frac{T(1)}{T(p)}$$

Where:
- $T(1)$ is the median execution time of the parallel algorithm constrained to 1 participating thread.
- $T(p)$ is the median execution time of the algorithm running with $p$ threads.

### B. Parallel Efficiency
Parallel efficiency $E(p)$ assesses the fraction of linear scaling realized per thread:

$$E(p) = \frac{S(p)}{p} = \frac{T(1)}{p \cdot T(p)}$$

- $E(p) = 1.0$ (100%): Perfect linear scaling.
- $E(p) < 1.0$: Scaling loss due to Amdahl's serial fraction, memory bus saturation, thread synchronization, or load imbalance.
- $E(p) > 1.0$: Superlinear scaling (typically occurring when partitioned workloads fit into private per-core CPU caches).

### C. Memory Bandwidth Byte-Counting Convention
Bandwidth calculations strictly follow the standard STREAM benchmark conventions:

| Workload | Kernel Formula | Memory Accesses per Element | Total Bytes Transferred |
|---|---|---|---|
| **Array Copy** | $y_i = x_i$ | 1 Read ($x$), 1 Write ($y$) | $2 \cdot N \cdot \text{sizeof}(T)$ |
| **Array Scale** | $y_i = \alpha \cdot x_i$ | 1 Read ($x$), 1 Write ($y$) | $2 \cdot N \cdot \text{sizeof}(T)$ |
| **Array Triad** | $a_i = b_i + \alpha \cdot c_i$ | 2 Reads ($b, c$), 1 Write ($a$) | $3 \cdot N \cdot \text{sizeof}(T)$ |
| **Vector Add** | $c_i = a_i + b_i$ | 2 Reads ($a, b$), 1 Write ($c$) | $3 \cdot N \cdot \text{sizeof}(T)$ |
| **Sequential Read** | $\sum x_i$ | 1 Read ($x$) | $1 \cdot N \cdot \text{sizeof}(T)$ |

$$\text{Effective Bandwidth (GB/s)} = \frac{\text{Total Bytes Transferred}}{\text{Elapsed Time (seconds)} \times 10^9}$$

---

## 2. Experimental Controls

To eliminate measurement artifacts and preserve experimental validity:

1. **Decoupled Setup and Allocation**:
   Memory allocations (`AlignedVector`, `Matrix`) and data initialization occur strictly outside Google Benchmark timing loops. Measured time corresponds exclusively to kernel execution.
2. **Cache Warming**:
   Each benchmark executes dedicated warmup iterations before measurement data is collected.
3. **Median and Dispersion Metrics**:
   Because modern operating systems experience context switching, CPU frequency governors, and background interrupts, performance is reported using the **median** and **Interquartile Range (IQR)** rather than the minimum runtime.
4. **Compiler Optimization Barrier**:
   Output pointers are wrapped in `benchmark::DoNotOptimize(...)` to prevent the compiler from dead-code-eliminating computations.

---

## 3. Findings from Empirical Hardware Profiling

Experiments conducted on the target hardware (Intel Core i5-8265U, 4 cores / 8 threads):

### A. Parallel Overhead on Small Workloads
On small working sets (e.g., $N=1,024$ floats, 4 KiB):
- 1 thread runtime: **770 ns**
- 4 threads runtime: **1,167 ns** ($S(4) = 0.66\times$, efficiency = $16.5\%$)
- 8 threads runtime: **1,997 ns** ($S(8) = 0.39\times$)

**Finding**: Thread dispatch, barrier synchronization, and OpenMP thread pool wake-up costs exceed the entire execution duration of small vectors. Multithreading is counter-productive below cache-sized thresholds.

### B. Memory Bandwidth Saturation (DRAM Bottleneck)
On large out-of-cache workloads (e.g. SAXPY on 4,194,304 floats, 16 MiB):
- 1 thread runtime: **3.87 ms** (Speedup: $1.00\times$)
- 4 threads runtime: **3.72 ms** (Speedup: $1.04\times$, Efficiency: $26.0\%$)
- 8 threads runtime: **4.12 ms** (Speedup: $0.94\times$, Efficiency: $11.7\%$)

**Finding**: The memory bus saturates at approximately 15–20 GB/s on dual-channel DDR4. Adding more threads past the memory bus bandwidth ceiling cannot accelerate memory-bound kernels.

### C. Compute-Bound vs Memory-Bound Scaling
In contrast to SAXPY, floating-point `ReduceSum` on 1,048,576 floats scales cleanly:
- 1 thread: **1.13 ms** ($1.00\times$)
- 2 threads: **0.57 ms** ($1.98\times$, $99.1\%$ efficiency)
- 4 threads: **0.29 ms** ($3.90\times$, $97.6\%$ efficiency)
- 8 threads: **0.16 ms** ($7.13\times$, $89.1\%$ efficiency)

**Finding**: Accumulating partial sums into per-thread registers reduces DRAM traffic and enables near-linear CPU scaling across all 4 physical cores and hyperthreads.

### D. OpenMP Scheduling under Load Imbalance
Under synthetic imbalanced workloads where iteration cost scales with iteration index $i$:
- **Serial**: 66.3 ms
- **Static(32)**: 27.1 ms ($2.45\times$ speedup)
- **Dynamic(32)**: 22.8 ms ($2.91\times$ speedup)
- **Guided(32)**: 20.9 ms ($3.18\times$ speedup)

**Finding**: Guided and Dynamic scheduling outperform Static scheduling by 29% under non-uniform workloads, dynamically balancing iterations across idle cores.
