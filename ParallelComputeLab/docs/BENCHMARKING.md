# Benchmarking Guide and Experimental Automation

`ParallelComputeLab` provides an end-to-end benchmarking infrastructure using **Google Benchmark**, combined with Python automation scripts for metadata logging, result validation, statistical aggregation, and chart generation.

---

## 1. Quick Start

Run the automated benchmark suite:

```bash
python3 ParallelComputeLab/scripts/run_benchmarks.py \
  --binary build/release/parallel_compute_benchmarks \
  --output-dir ParallelComputeLab/results \
  --repetitions 3 \
  --min-time 0.2
```

This command automatically:
1. Gathers hardware topology, cache hierarchy, and compiler metadata into `results/system_metadata.json`.
2. Executes Google Benchmark with specified repetition and timing settings.
3. Saves raw JSON measurements into `results/raw_benchmark_results.json` and CSV into `results/raw_benchmark_results.csv`.
4. Runs `validate_results.py` to ensure schema conformance and measurement integrity.

---

## 2. Statistical Analysis and Graph Generation

Process the raw results to produce summary metrics, speedup/efficiency calculations, markdown tables, and plots:

```bash
python3 ParallelComputeLab/scripts/analyze_results.py \
  ParallelComputeLab/results/raw_benchmark_results.json \
  ParallelComputeLab/results
```

Generated Output Artifacts:
- `results/summary_metrics.json`: Median, mean, stddev, IQR, bandwidth, and speedup for each benchmark.
- `results/summary_table.md`: Formatted Markdown table comparing variants, scaling factors, and efficiency.
- `results/scaling_threads.png`: Strong scaling speedup $S(p)$ and parallel efficiency $E(p)$ plots.
- `results/workload_comparison.png`: Comparison of Scalar, Optimized, SIMD, and Parallel variants across working sets.
- `results/memory_bandwidth.png`: Strided memory access latency showing cache eviction penalties.
- `results/scheduling_comparison.png`: Runtime comparison of OpenMP static, dynamic, and guided scheduling policies.

---

## 3. Benchmark Binary CLI Options

The `parallel_compute_benchmarks` binary accepts standard Google Benchmark flags:

```bash
# Filter benchmarks by name regex
./build/release/parallel_compute_benchmarks --benchmark_filter="BM_MatMul.*"

# Run 5 repetitions per configuration
./build/release/parallel_compute_benchmarks --benchmark_repetitions=5

# Set minimum time per repetition
./build/release/parallel_compute_benchmarks --benchmark_min_time=0.5s

# Output to JSON
./build/release/parallel_compute_benchmarks --benchmark_out=results/custom.json --benchmark_out_format=json
```

---

## 4. Working Set Sizes Evaluated

To differentiate cache-resident computation from memory-bandwidth bottlenecks, benchmarks are evaluated across four representative memory levels:

| Working Set Size | Element Count (Float) | Memory Footprint | Primary Memory Subsystem |
|---|---|---|---|
| **Small** | 1,024 elements | 4 KiB | L1 Data Cache (32 KiB) |
| **Medium** | 16,384 elements | 64 KiB | L2 Cache (256 KiB) |
| **Large** | 262,144 elements | 1 MiB | L3 Shared Cache (6 MiB) |
| **DRAM** | 4,194,304 elements | 16 MiB | Main Memory (DRAM) |

---

## 5. Result Schema

The raw benchmark output follows the Google Benchmark JSON schema version 1, enriched with custom context:

```json
{
  "context": {
    "cpu_model": "Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz",
    "cpu_vendor": "GenuineIntel",
    "os_name": "Linux 6.12.113+deb13-amd64",
    "architecture": "x86_64",
    "logical_cores": "8",
    "physical_cores": "4",
    "openmp_max_threads": "8",
    "l1d_cache_bytes": "32768",
    "l2_cache_bytes": "262144",
    "l3_cache_bytes": "6291456",
    "compiler_id": "GNU",
    "compiler_version": "14.2.0",
    "build_type": "Release",
    "has_avx2": "true",
    "has_fma": "true"
  },
  "benchmarks": [
    {
      "name": "BM_VectorAdd_SIMD/4194304",
      "iterations": 83,
      "real_time": 8234125.12,
      "cpu_time": 8233400.05,
      "time_unit": "ns",
      "bytes_per_second": 6112349000.0,
      "items_per_second": 509362416.0
    }
  ]
}
```
