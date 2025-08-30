"""
TensorOps: SIMD-accelerated tensor operations using Numba
"""

import numpy as np
import numba
from numba import jit, prange
from typing import Optional, Union, Tuple
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


class TensorOps:
    """
    High-performance tensor operations with SIMD acceleration
    
    Uses Numba JIT compilation for optimal CPU performance
    """
    
    def __init__(self):
        self.num_cores = mp.cpu_count()
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def add_simd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """SIMD-accelerated element-wise addition"""
        if a.shape != b.shape:
            # Handle broadcasting manually for Numba
            if a.size == 1:
                result = np.empty_like(b)
                scalar_val = a.flat[0]
                for i in prange(b.size):
                    result.flat[i] = scalar_val + b.flat[i]
                return result
            elif b.size == 1:
                result = np.empty_like(a)
                scalar_val = b.flat[0]
                for i in prange(a.size):
                    result.flat[i] = a.flat[i] + scalar_val
                return result
        
        result = np.empty_like(a)
        for i in prange(a.size):
            result.flat[i] = a.flat[i] + b.flat[i]
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def sub_simd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """SIMD-accelerated element-wise subtraction"""
        if a.shape != b.shape:
            if a.size == 1:
                result = np.empty_like(b)
                scalar_val = a.flat[0]
                for i in prange(b.size):
                    result.flat[i] = scalar_val - b.flat[i]
                return result
            elif b.size == 1:
                result = np.empty_like(a)
                scalar_val = b.flat[0]
                for i in prange(a.size):
                    result.flat[i] = a.flat[i] - scalar_val
                return result
        
        result = np.empty_like(a)
        for i in prange(a.size):
            result.flat[i] = a.flat[i] - b.flat[i]
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def mul_simd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """SIMD-accelerated element-wise multiplication"""
        if a.shape != b.shape:
            if a.size == 1:
                result = np.empty_like(b)
                scalar_val = a.flat[0]
                for i in prange(b.size):
                    result.flat[i] = scalar_val * b.flat[i]
                return result
            elif b.size == 1:
                result = np.empty_like(a)
                scalar_val = b.flat[0]
                for i in prange(a.size):
                    result.flat[i] = a.flat[i] * scalar_val
                return result
        
        result = np.empty_like(a)
        for i in prange(a.size):
            result.flat[i] = a.flat[i] * b.flat[i]
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def relu_simd(x: np.ndarray) -> np.ndarray:
        """SIMD-accelerated ReLU activation"""
        result = np.empty_like(x)
        for i in prange(x.size):
            result.flat[i] = max(0.0, x.flat[i])
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def sigmoid_simd(x: np.ndarray) -> np.ndarray:
        """SIMD-accelerated sigmoid activation"""
        result = np.empty_like(x)
        for i in prange(x.size):
            val = x.flat[i]
            # Clamp to prevent overflow
            if val > 500:
                result.flat[i] = 1.0
            elif val < -500:
                result.flat[i] = 0.0
            else:
                result.flat[i] = 1.0 / (1.0 + np.exp(-val))
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def matmul_block(a: np.ndarray, b: np.ndarray, 
                     start_row: int, end_row: int) -> np.ndarray:
        """Block matrix multiplication for parallel processing"""
        m, k = a.shape
        k2, n = b.shape
        
        if k != k2:
            raise ValueError("Matrix dimensions don't match for multiplication")
        
        result = np.zeros((end_row - start_row, n), dtype=a.dtype)
        
        for i in prange(end_row - start_row):
            for j in range(n):
                for l in range(k):
                    result[i, j] += a[start_row + i, l] * b[l, j]
        
        return result
    
    def matmul_parallel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Parallel matrix multiplication"""
        m, k = a.shape
        k2, n = b.shape
        
        if k != k2:
            raise ValueError("Matrix dimensions don't match for multiplication")
        
        # For small matrices, use standard multiplication
        if m * n < 10000:
            return np.matmul(a, b)
        
        # Split work across cores
        rows_per_core = max(1, m // self.num_cores)
        futures = []
        
        with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
            for i in range(0, m, rows_per_core):
                start_row = i
                end_row = min(i + rows_per_core, m)
                future = executor.submit(self.matmul_block, a, b, start_row, end_row)
                futures.append((start_row, future))
        
        # Combine results
        result = np.zeros((m, n), dtype=a.dtype)
        for start_row, future in futures:
            block_result = future.result()
            end_row = start_row + block_result.shape[0]
            result[start_row:end_row] = block_result
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def sum_parallel_1d(x: np.ndarray) -> float:
        """Parallel sum for 1D arrays"""
        total = 0.0
        for i in prange(x.size):
            total += x.flat[i]
        return total
    
    def sum_parallel(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                    keepdims: bool = False) -> np.ndarray:
        """Parallel sum reduction"""
        if axis is None:
            # Sum all elements
            if x.size > 10000:
                return np.array(self.sum_parallel_1d(x.flatten()))
            else:
                return np.sum(x)
        else:
            # Use NumPy for axis-specific reductions (more complex to parallelize)
            return np.sum(x, axis=axis, keepdims=keepdims)
    
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def conv2d_kernel(input_data: np.ndarray, kernel: np.ndarray, 
                     output: np.ndarray, stride: int = 1, padding: int = 0):
        """SIMD-accelerated 2D convolution kernel"""
        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, in_channels_k, kernel_height, kernel_width = kernel.shape
        _, _, out_height, out_width = output.shape
        
        for b in prange(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        value = 0.0
                        for ic in range(in_channels):
                            for kh in range(kernel_height):
                                for kw in range(kernel_width):
                                    ih = oh * stride - padding + kh
                                    iw = ow * stride - padding + kw
                                    
                                    if 0 <= ih < in_height and 0 <= iw < in_width:
                                        value += (input_data[b, ic, ih, iw] * 
                                                kernel[oc, ic, kh, kw])
                        
                        output[b, oc, oh, ow] = value