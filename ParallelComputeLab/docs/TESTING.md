# Testing and Correctness Verification

Every optimized implementation in `ParallelComputeLab` is rigorously verified against a reference scalar implementation across a broad spectrum of input dimensions, numerical corner cases, and concurrency configurations.

---

## 1. Test Architecture

The test suite is structured into 5 dedicated directories:

| Directory | Suite Name | Description & Coverage |
|---|---|---|
| `tests/unit/` | `CpuFeaturesTest`, `SystemInfoTest`, `AlignmentTest`, `AlignedVectorTest`, `MatrixTest`, `BandwidthCounterTest`, `ElementwiseTest` | Validates CPUID detection, cache topology, 64-byte alignment, memory workloads, and elementwise vector operations across sizes. |
| `tests/numerical/` | `ReductionsTest`, `MatmulNumericalTest` | Floating-point numerical validation for sum, min, max, dot product, mean, and rectangular matrix multiplication ($M \times K \times N$). |
| `tests/edge_cases/` | `EdgeCasesTest`, `DimensionValidationTest` | Empty arrays ($N=0$), single elements ($N=1$), NaN and infinity propagation, dimension mismatches, and matrix bounds validation. |
| `tests/concurrency/` | `ConcurrencyTest` | Thread scaling (1, 2, 4, 8 threads), OpenMP scheduling policies (static, dynamic, guided), and reduction determinism. |
| `tests/regression/` | `RegressionTest` | Repeated execution determinism, exhaustive vector tail processing (lengths 1 to 33), and extreme aspect ratios (tall-skinny vs short-wide matrices). |

---

## 2. Executing Tests

### Using CTest

```bash
ctest --test-dir build/release --output-on-failure
```

### Running the Test Binary Directly

```bash
# Run all 67 tests
./build/release/parallel_compute_tests

# Run a specific test suite
./build/release/parallel_compute_tests --gtest_filter=ReductionsTest.*

# Run only concurrency and thread scaling tests
./build/release/parallel_compute_tests --gtest_filter=ConcurrencyTest.*
```

---

## 3. Numerical Tolerance Guidelines

Because parallel reductions and SIMD vector operations alter the associative grouping of floating-point arithmetic ($((a+b)+c) \neq (a+(b+c))$), tests avoid exact equality `EXPECT_FLOAT_EQ` on large accumulated sums:

1. **Exact Comparisons (`EXPECT_DOUBLE_EQ`)**:
   - Used for minimum, maximum, integer reductions, and exact copy/scale workloads.
2. **Relative & Absolute Bounds (`EXPECT_NEAR`)**:
   - For a summation of $N = 10,000$ single-precision floats with total magnitude $\sim 5,000$, rounding error accumulates bounded by $N \cdot \epsilon \approx 10^4 \times 5.96 \times 10^{-8} \approx 6 \times 10^{-4}$.
   - Summation tolerance is bounded to $10^{-2}$ ($\sim 0.0002\%$ relative tolerance).
   - Double-precision matrix multiplications are bounded to $10^{-9}$.

---

## 4. Sanitizer Execution Results

| Configuration | Test Execution Command | Tests Ran | Status | Findings |
|---|---|---|---|---|
| **Release (GCC 14.2.0)** | `ctest --test-dir build/release` | 67 / 67 | **PASS** | 100% passed in 1.55s |
| **ASan + UBSan** | `./build/asan/parallel_compute_tests` | 67 / 67 | **PASS** | 0 memory leaks, 0 buffer overflows, 0 undefined behavior |
| **ThreadSanitizer** | `./build/tsan/parallel_compute_tests` | 67 / 67 | **PASS** | Algorithmic logic data-race free. Documented uninstrumented system `libgomp` warnings (see [LIMITATIONS.md](LIMITATIONS.md)). |
