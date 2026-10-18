# ParallelComputeLab Experimental Results Summary

## 1. Environment Provenance
- **CPU Model**: Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz
- **Logical Cores**: 8
- **Physical Cores**: 4
- **Caches**: L1d: 32 KiB, L2: 256 KiB, L3: 6 MiB
- **Compiler**: Unknown Unknown (Unknown)
- **SIMD Features**: AVX2: true, FMA: true

## 2. Kernel Scaling, Speedup, and Parallel Efficiency

| Kernel | Workload Size | Threads | Median Time (ns) | Speedup S(p) | Efficiency E(p) |
|---|---|---|---|---|---|
| `BM_DotProduct_Parallel` | 65536 | 1 | 69852.96 | **1.00x** | 100.0% |
| `BM_DotProduct_Parallel` | 65536 | 2 | 36395.43 | **1.92x** | 96.0% |
| `BM_DotProduct_Parallel` | 65536 | 4 | 18869.82 | **3.70x** | 92.5% |
| `BM_DotProduct_Parallel` | 65536 | 8 | 11379.78 | **6.14x** | 76.7% |
| `BM_DotProduct_Parallel` | 1048576 | 1 | 1194493.76 | **1.00x** | 100.0% |
| `BM_DotProduct_Parallel` | 1048576 | 2 | 631180.25 | **1.89x** | 94.6% |
| `BM_DotProduct_Parallel` | 1048576 | 4 | 335605.87 | **3.56x** | 89.0% |
| `BM_DotProduct_Parallel` | 1048576 | 8 | 288611.27 | **4.14x** | 51.7% |
| `BM_MatMul_Parallel` | 128 | 1 | 2832830.19 | **1.00x** | 100.0% |
| `BM_MatMul_Parallel` | 128 | 2 | 1433182.22 | **1.98x** | 98.8% |
| `BM_MatMul_Parallel` | 128 | 4 | 1561105.87 | **1.81x** | 45.4% |
| `BM_MatMul_Parallel` | 128 | 8 | 2198921.31 | **1.29x** | 16.1% |
| `BM_MatMul_Parallel` | 256 | 1 | 22917717.40 | **1.00x** | 100.0% |
| `BM_MatMul_Parallel` | 256 | 2 | 11603682.50 | **1.98x** | 98.8% |
| `BM_MatMul_Parallel` | 256 | 4 | 6986025.67 | **3.28x** | 82.0% |
| `BM_MatMul_Parallel` | 256 | 8 | 10021530.00 | **2.29x** | 28.6% |
| `BM_MatMul_Parallel` | 512 | 1 | 182720960.00 | **1.00x** | 100.0% |
| `BM_MatMul_Parallel` | 512 | 2 | 93096522.00 | **1.96x** | 98.1% |
| `BM_MatMul_Parallel` | 512 | 4 | 56343840.00 | **3.24x** | 81.1% |
| `BM_MatMul_Parallel` | 512 | 8 | 57321565.00 | **3.19x** | 39.8% |
| `BM_ReduceSum_Parallel` | 65536 | 1 | 69103.22 | **1.00x** | 100.0% |
| `BM_ReduceSum_Parallel` | 65536 | 2 | 36080.57 | **1.92x** | 95.8% |
| `BM_ReduceSum_Parallel` | 65536 | 4 | 19733.64 | **3.50x** | 87.5% |
| `BM_ReduceSum_Parallel` | 65536 | 8 | 10783.21 | **6.41x** | 80.1% |
| `BM_ReduceSum_Parallel` | 1048576 | 1 | 1126693.30 | **1.00x** | 100.0% |
| `BM_ReduceSum_Parallel` | 1048576 | 2 | 568617.73 | **1.98x** | 99.1% |
| `BM_ReduceSum_Parallel` | 1048576 | 4 | 288628.73 | **3.90x** | 97.6% |
| `BM_ReduceSum_Parallel` | 1048576 | 8 | 158030.65 | **7.13x** | 89.1% |
| `BM_SAXPY_Parallel` | 262144 | 1 | 64094.48 | **1.00x** | 100.0% |
| `BM_SAXPY_Parallel` | 262144 | 2 | 33037.12 | **1.94x** | 97.0% |
| `BM_SAXPY_Parallel` | 262144 | 4 | 33995.65 | **1.89x** | 47.1% |
| `BM_SAXPY_Parallel` | 262144 | 8 | 18958.11 | **3.38x** | 42.3% |
| `BM_SAXPY_Parallel` | 4194304 | 1 | 3869699.46 | **1.00x** | 100.0% |
| `BM_SAXPY_Parallel` | 4194304 | 2 | 3876567.48 | **1.00x** | 49.9% |
| `BM_SAXPY_Parallel` | 4194304 | 4 | 3723541.18 | **1.04x** | 26.0% |
| `BM_SAXPY_Parallel` | 4194304 | 8 | 4124826.97 | **0.94x** | 11.7% |
| `BM_VectorAdd_Parallel` | 1024 | 1 | 770.93 | **1.00x** | 100.0% |
| `BM_VectorAdd_Parallel` | 1024 | 2 | 1213.27 | **0.64x** | 31.8% |
| `BM_VectorAdd_Parallel` | 1024 | 4 | 1167.10 | **0.66x** | 16.5% |
| `BM_VectorAdd_Parallel` | 1024 | 8 | 1997.29 | **0.39x** | 4.8% |
| `BM_VectorAdd_Parallel` | 262144 | 1 | 77530.10 | **1.00x** | 100.0% |
| `BM_VectorAdd_Parallel` | 262144 | 2 | 47595.07 | **1.63x** | 81.4% |
| `BM_VectorAdd_Parallel` | 262144 | 4 | 26193.50 | **2.96x** | 74.0% |
| `BM_VectorAdd_Parallel` | 262144 | 8 | 32132.62 | **2.41x** | 30.2% |
| `BM_VectorAdd_Parallel` | 4194304 | 1 | 6936224.47 | **1.00x** | 100.0% |
| `BM_VectorAdd_Parallel` | 4194304 | 2 | 6521549.50 | **1.06x** | 53.2% |
| `BM_VectorAdd_Parallel` | 4194304 | 4 | 6698213.42 | **1.04x** | 25.9% |
| `BM_VectorAdd_Parallel` | 4194304 | 8 | 8104803.86 | **0.86x** | 10.7% |

## 3. Implementation Variant Comparison (Scalar vs Optimized vs SIMD)

| Workload / Size | Variant | Median Latency (ns) | IQR (ns) | Bandwidth / Throughput |
|---|---|---|---|---|
| `BM_BLAS_Unavailable` (-) | `Unavailable` | 0.13 | 0.00 | - |
| `BM_DotProduct_Optimized` (1024) | `Optimized` | 256.51 | 5.49 | 31.94 GB/s |
| `BM_DotProduct_Optimized` (1048576) | `Optimized` | 516601.63 | 13184.63 | 16.24 GB/s |
| `BM_DotProduct_Optimized` (65536) | `Optimized` | 17308.42 | 607.79 | 30.29 GB/s |
| `BM_DotProduct_SIMD` (1024) | `SIMD` | 130.01 | 3.69 | 63.01 GB/s |
| `BM_DotProduct_SIMD` (1048576) | `SIMD` | 474393.59 | 26703.21 | 17.68 GB/s |
| `BM_DotProduct_SIMD` (65536) | `SIMD` | 8781.59 | 108.83 | 59.71 GB/s |
| `BM_DotProduct_Scalar` (1024) | `Scalar` | 1044.38 | 30.85 | 7.85 GB/s |
| `BM_DotProduct_Scalar` (1048576) | `Scalar` | 1217664.08 | 14290.66 | 6.89 GB/s |
| `BM_DotProduct_Scalar` (65536) | `Scalar` | 69550.22 | 336.02 | 7.54 GB/s |
| `BM_MatMul_Blocked` (128) | `Blocked` | 3034517.58 | 333962.00 | - |
| `BM_MatMul_Blocked` (256) | `Blocked` | 24127382.20 | 828365.00 | - |
| `BM_MatMul_Blocked` (512) | `Blocked` | 193105250.00 | 18245921.00 | - |
| `BM_MatMul_Blocked` (64) | `Blocked` | 375920.05 | 7786.17 | - |
| `BM_MatMul_Reference` (128) | `Reference` | 3728053.31 | 29579.34 | - |
| `BM_MatMul_Reference` (256) | `Reference` | 32399893.75 | 334521.75 | - |
| `BM_MatMul_Reference` (64) | `Reference` | 462225.88 | 13306.42 | - |
| `BM_MatMul_SIMD` (128) | `SIMD` | 231653.44 | 13858.43 | - |
| `BM_MatMul_SIMD` (256) | `SIMD` | 2221290.36 | 128472.87 | - |
| `BM_MatMul_SIMD` (512) | `SIMD` | 17414398.14 | 1596961.43 | - |
| `BM_MatMul_SIMD` (64) | `SIMD` | 28572.08 | 586.64 | - |
| `BM_Memory_ArrayCopy` (1024) | `ArrayCopy` | 159.56 | 2.28 | 51.35 GB/s |
| `BM_Memory_ArrayCopy` (16384) | `ArrayCopy` | 1695.27 | 50.29 | 77.32 GB/s |
| `BM_Memory_ArrayCopy` (262144) | `ArrayCopy` | 58753.28 | 526.00 | 35.70 GB/s |
| `BM_Memory_ArrayCopy` (4194304) | `ArrayCopy` | 3775948.46 | 124035.00 | 8.89 GB/s |
| `BM_Memory_ArrayTriad` (1024) | `ArrayTriad` | 152.84 | 1.36 | 80.41 GB/s |
| `BM_Memory_ArrayTriad` (16384) | `ArrayTriad` | 2863.40 | 11.54 | 68.67 GB/s |
| `BM_Memory_ArrayTriad` (262144) | `ArrayTriad` | 60692.36 | 2655.77 | 51.84 GB/s |
| `BM_Memory_ArrayTriad` (4194304) | `ArrayTriad` | 5243008.37 | 428802.59 | 9.60 GB/s |
| `BM_Memory_SequentialRead` (1024) | `SequentialRead` | 1029.24 | 10.25 | 3.98 GB/s |
| `BM_Memory_SequentialRead` (16384) | `SequentialRead` | 17206.20 | 183.10 | 3.81 GB/s |
| `BM_Memory_SequentialRead` (262144) | `SequentialRead` | 281279.35 | 5558.86 | 3.73 GB/s |
| `BM_Memory_SequentialRead` (4194304) | `SequentialRead` | 4649481.92 | 86233.62 | 3.61 GB/s |
| `BM_Memory_StridedRead` (4194304) | `StridedRead` | 4621683.77 | 580176.35 | 3.63 GB/s |
| `BM_Memory_StridedRead` (4194304) | `StridedRead` | 1252147.25 | 114499.21 | 0.84 GB/s |
| `BM_Memory_StridedRead` (4194304) | `StridedRead` | 2424054.20 | 30229.94 | 3.46 GB/s |
| `BM_Memory_StridedRead` (4194304) | `StridedRead` | 1189162.23 | 16756.61 | 0.44 GB/s |
| `BM_Memory_StridedRead` (4194304) | `StridedRead` | 1743570.40 | 305869.34 | 2.41 GB/s |
| `BM_Memory_StridedRead` (4194304) | `StridedRead` | 1024621.11 | 150209.96 | 0.26 GB/s |
| `BM_Memory_StridedRead` (4194304) | `StridedRead` | 1168857.41 | 164366.46 | 1.79 GB/s |
| `BM_ReduceSum_Optimized` (1024) | `Optimized` | 245.97 | 3.87 | 16.69 GB/s |
| `BM_ReduceSum_Optimized` (1048576) | `Optimized` | 282584.02 | 10408.72 | 14.85 GB/s |
| `BM_ReduceSum_Optimized` (65536) | `Optimized` | 17421.09 | 247.22 | 15.05 GB/s |
| `BM_ReduceSum_SIMD` (1024) | `SIMD` | 110.01 | 2.61 | 37.24 GB/s |
| `BM_ReduceSum_SIMD` (1048576) | `SIMD` | 141271.15 | 4195.16 | 29.69 GB/s |
| `BM_ReduceSum_SIMD` (65536) | `SIMD` | 8600.65 | 105.59 | 30.48 GB/s |
| `BM_ReduceSum_Scalar` (1024) | `Scalar` | 1030.11 | 1.99 | 3.98 GB/s |
| `BM_ReduceSum_Scalar` (1048576) | `Scalar` | 1123263.62 | 100149.13 | 3.74 GB/s |
| `BM_ReduceSum_Scalar` (65536) | `Scalar` | 69381.48 | 1032.20 | 3.79 GB/s |
| `BM_SAXPY_Optimized` (1024) | `Optimized` | 269.62 | 44.74 | 45.62 GB/s |
| `BM_SAXPY_Optimized` (16384) | `Optimized` | 2665.70 | 240.10 | 73.76 GB/s |
| `BM_SAXPY_Optimized` (262144) | `Optimized` | 65455.71 | 10321.26 | 48.07 GB/s |
| `BM_SAXPY_Optimized` (4194304) | `Optimized` | 5343956.50 | 437491.15 | 9.42 GB/s |
| `BM_SAXPY_SIMD` (1024) | `SIMD` | 192.28 | 6.22 | 63.91 GB/s |
| `BM_SAXPY_SIMD` (16384) | `SIMD` | 1707.36 | 79.13 | 115.16 GB/s |
| `BM_SAXPY_SIMD` (262144) | `SIMD` | 48339.86 | 3134.68 | 65.07 GB/s |
| `BM_SAXPY_SIMD` (4194304) | `SIMD` | 4952655.41 | 444960.62 | 10.16 GB/s |
| `BM_SAXPY_Scalar` (1024) | `Scalar` | 245.06 | 1.61 | 50.15 GB/s |
| `BM_SAXPY_Scalar` (16384) | `Scalar` | 2526.76 | 63.87 | 77.81 GB/s |
| `BM_SAXPY_Scalar` (262144) | `Scalar` | 59736.83 | 4553.82 | 52.66 GB/s |
| `BM_SAXPY_Scalar` (4194304) | `Scalar` | 4997119.05 | 2045459.19 | 10.08 GB/s |
| `BM_Schedule_Imbalanced` (32768) | `Imbalanced` | 66318406.50 | 1245779.00 | - |
| `BM_Schedule_Imbalanced` (32768) | `Imbalanced` | 21675850.57 | 1219767.86 | - |
| `BM_Schedule_Imbalanced` (32768) | `Imbalanced` | 27091650.00 | 633570.00 | - |
| `BM_Schedule_Imbalanced` (32768) | `Imbalanced` | 21094485.67 | 938809.33 | - |
| `BM_Schedule_Imbalanced` (32768) | `Imbalanced` | 22759744.17 | 4013464.67 | - |
| `BM_Schedule_Imbalanced` (32768) | `Imbalanced` | 20887942.20 | 808776.80 | - |
| `BM_Schedule_Uniform` (32768) | `Uniform` | 583543.76 | 11011.85 | - |
| `BM_Schedule_Uniform` (32768) | `Uniform` | 178714.37 | 54291.91 | - |
| `BM_Schedule_Uniform` (32768) | `Uniform` | 174186.80 | 6304.74 | - |
| `BM_Schedule_Uniform` (32768) | `Uniform` | 795661.94 | 18866.54 | - |
| `BM_Schedule_Uniform` (32768) | `Uniform` | 180828.15 | 16779.80 | - |
| `BM_Schedule_Uniform` (32768) | `Uniform` | 175833.68 | 8483.65 | - |
| `BM_VectorAdd_Optimized` (1024) | `Optimized` | 217.23 | 1.84 | 56.57 GB/s |
| `BM_VectorAdd_Optimized` (16384) | `Optimized` | 3472.13 | 91.49 | 56.62 GB/s |
| `BM_VectorAdd_Optimized` (262144) | `Optimized` | 77564.87 | 11842.39 | 40.56 GB/s |
| `BM_VectorAdd_Optimized` (4194304) | `Optimized` | 6983938.07 | 71180.73 | 7.21 GB/s |
| `BM_VectorAdd_SIMD` (1024) | `SIMD` | 186.46 | 4.79 | 65.90 GB/s |
| `BM_VectorAdd_SIMD` (16384) | `SIMD` | 2445.66 | 267.65 | 80.42 GB/s |
| `BM_VectorAdd_SIMD` (262144) | `SIMD` | 65387.04 | 2678.15 | 48.11 GB/s |
| `BM_VectorAdd_SIMD` (4194304) | `SIMD` | 7464444.40 | 814786.65 | 6.74 GB/s |
| `BM_VectorAdd_Scalar` (1024) | `Scalar` | 220.81 | 8.31 | 55.83 GB/s |
| `BM_VectorAdd_Scalar` (16384) | `Scalar` | 3480.50 | 92.46 | 56.50 GB/s |
| `BM_VectorAdd_Scalar` (262144) | `Scalar` | 87462.43 | 23157.03 | 35.97 GB/s |
| `BM_VectorAdd_Scalar` (4194304) | `Scalar` | 6758129.17 | 803180.33 | 7.45 GB/s |
