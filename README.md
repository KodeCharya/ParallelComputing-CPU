# ParallelComputeLab

**ParallelComputeLab** is a CPU parallel computing, low-level optimization, SIMD vectorization, memory efficiency, and reproducible performance benchmarking framework in C++20.

The complete project implementation, test suite, benchmark suite, analysis scripts, and documentation reside in the [`ParallelComputeLab/`](ParallelComputeLab/) directory.

## Quick Links

- [Project Overview and Architecture](ParallelComputeLab/README.md)
- [Building and Compilation Guide](ParallelComputeLab/docs/BUILDING.md)
- [Testing and Verification Guide](ParallelComputeLab/docs/TESTING.md)
- [Benchmarking Guide](ParallelComputeLab/docs/BENCHMARKING.md)
- [Performance Methodology](ParallelComputeLab/docs/PERFORMANCE_METHODOLOGY.md)
- [Known Limitations](ParallelComputeLab/docs/LIMITATIONS.md)
- [Experimental Results](ParallelComputeLab/results/README.md)

## Quick Start

```bash
# Build
cmake -B build/release -S ParallelComputeLab -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j$(nproc)

# Test
ctest --test-dir build/release --output-on-failure

# Benchmark
python3 ParallelComputeLab/scripts/run_benchmarks.py \
  --binary build/release/parallel_compute_benchmarks \
  --output-dir ParallelComputeLab/results

# Analyze
python3 ParallelComputeLab/scripts/analyze_results.py \
  ParallelComputeLab/results/raw_benchmark_results.json \
  ParallelComputeLab/results
```
