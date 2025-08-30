"""
Performance benchmarks for ParallelCore library
"""

import numpy as np
import time
import multiprocessing as mp
from typing import List, Callable
import psutil

from parallelcore import VirtualCoreManager, Tensor
from parallelcore.core.scheduler import Scheduler


class BenchmarkSuite:
    """Comprehensive benchmark suite for performance testing"""
    
    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> dict:
        """Get system information for benchmarks"""
        return {
            "cpu_count": mp.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
        }
    
    def benchmark_parallel_map(self, data_sizes: List[int], 
                              complexity_levels: List[int]):
        """Benchmark parallel map operations"""
        print("Benchmarking parallel map operations...")
        
        results = {}
        
        for data_size in data_sizes:
            for complexity in complexity_levels:
                print(f"  Testing data_size={data_size}, complexity={complexity}")
                
                # Generate test data
                data = list(range(data_size))
                
                def test_function(x):
                    # Simulate computational complexity
                    result = x
                    for _ in range(complexity):
                        result = result * 1.001 + 0.001
                    return result
                
                # Sequential baseline
                start_time = time.time()
                sequential_result = [test_function(x) for x in data]
                sequential_time = time.time() - start_time
                
                # Parallel execution
                with VirtualCoreManager() as manager:
                    start_time = time.time()
                    parallel_result = manager.map_parallel(test_function, data)
                    parallel_time = time.time() - start_time
                
                speedup = sequential_time / parallel_time if parallel_time > 0 else 0
                
                results[(data_size, complexity)] = {
                    "sequential_time": sequential_time,
                    "parallel_time": parallel_time,
                    "speedup": speedup,
                    "efficiency": speedup / self.system_info["cpu_count"]
                }
        
        self.results["parallel_map"] = results
        return results
    
    def benchmark_tensor_operations(self, matrix_sizes: List[int]):
        """Benchmark tensor operations with SIMD"""
        print("Benchmarking tensor operations...")
        
        results = {}
        
        for size in matrix_sizes:
            print(f"  Testing matrix size: {size}x{size}")
            
            # Create test tensors
            a = Tensor.randn(size, size)
            b = Tensor.randn(size, size)
            
            # NumPy baseline
            a_np = a.numpy()
            b_np = b.numpy()
            
            operations = {
                "add": lambda x, y: x + y,
                "multiply": lambda x, y: x * y,
                "matmul": lambda x, y: np.matmul(x, y) if isinstance(x, np.ndarray) else x.matmul(y)
            }
            
            size_results = {}
            
            for op_name, op_func in operations.items():
                # NumPy timing
                start_time = time.time()
                numpy_result = op_func(a_np, b_np)
                numpy_time = time.time() - start_time
                
                # Tensor timing
                start_time = time.time()
                tensor_result = op_func(a, b)
                tensor_time = time.time() - start_time
                
                speedup = numpy_time / tensor_time if tensor_time > 0 else 0
                
                size_results[op_name] = {
                    "numpy_time": numpy_time,
                    "tensor_time": tensor_time,
                    "speedup": speedup
                }
            
            results[size] = size_results
        
        self.results["tensor_operations"] = results
        return results
    
    def benchmark_memory_efficiency(self, array_sizes: List[int]):
        """Benchmark memory efficiency"""
        print("Benchmarking memory efficiency...")
        
        results = {}
        
        for size in array_sizes:
            print(f"  Testing array size: {size:,} elements")
            
            # Memory before
            memory_before = psutil.virtual_memory().used
            
            # Create large tensor
            start_time = time.time()
            tensor = Tensor.randn(size)
            creation_time = time.time() - start_time
            
            # Memory after creation
            memory_after_creation = psutil.virtual_memory().used
            
            # Perform operations
            start_time = time.time()
            result = tensor + tensor
            result = result * 2.0
            result = result.sum()
            operation_time = time.time() - start_time
            
            # Memory after operations
            memory_after_ops = psutil.virtual_memory().used
            
            # Clean up
            del tensor, result
            
            memory_used = memory_after_creation - memory_before
            memory_overhead = memory_after_ops - memory_after_creation
            
            results[size] = {
                "creation_time": creation_time,
                "operation_time": operation_time,
                "memory_used_mb": memory_used / (1024**2),
                "memory_overhead_mb": memory_overhead / (1024**2),
                "elements_per_second": size / (creation_time + operation_time)
            }
        
        self.results["memory_efficiency"] = results
        return results
    
    def benchmark_scheduler_optimization(self):
        """Benchmark scheduler optimization"""
        print("Benchmarking scheduler optimization...")
        
        scheduler = Scheduler()
        
        # Test different workload types
        workload_types = ["cpu_bound", "io_bound", "mixed"]
        results = {}
        
        for workload_type in workload_types:
            optimal_cores = scheduler.get_optimal_core_count(workload_type)
            
            # Test chunk size calculation
            data_sizes = [100, 1000, 10000, 100000]
            chunk_sizes = []
            
            for data_size in data_sizes:
                chunk_size = scheduler.calculate_chunk_size(data_size, optimal_cores)
                chunk_sizes.append(chunk_size)
            
            results[workload_type] = {
                "optimal_cores": optimal_cores,
                "chunk_sizes": dict(zip(data_sizes, chunk_sizes))
            }
        
        self.results["scheduler"] = results
        return results
    
    def run_comprehensive_benchmark(self):
        """Run all benchmarks"""
        print("Running comprehensive benchmark suite...")
        print(f"System: {self.system_info['cpu_count']} cores, "
              f"{self.system_info['memory_gb']:.1f}GB RAM")
        print("=" * 60)
        
        # Parallel map benchmarks
        self.benchmark_parallel_map(
            data_sizes=[1000, 5000, 10000],
            complexity_levels=[1, 10, 100]
        )
        
        # Tensor operation benchmarks
        self.benchmark_tensor_operations([100, 500, 1000, 2000])
        
        # Memory efficiency benchmarks
        self.benchmark_memory_efficiency([10000, 100000, 1000000])
        
        # Scheduler optimization benchmarks
        self.benchmark_scheduler_optimization()
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        if "parallel_map" in self.results:
            print("\nParallel Map Performance:")
            for (data_size, complexity), result in self.results["parallel_map"].items():
                print(f"  {data_size:,} items, complexity {complexity}: "
                      f"{result['speedup']:.2f}x speedup, "
                      f"{result['efficiency']:.2f} efficiency")
        
        if "tensor_operations" in self.results:
            print("\nTensor Operations Performance:")
            for size, ops in self.results["tensor_operations"].items():
                print(f"  {size}x{size} matrices:")
                for op_name, result in ops.items():
                    print(f"    {op_name}: {result['speedup']:.2f}x speedup")
        
        if "memory_efficiency" in self.results:
            print("\nMemory Efficiency:")
            for size, result in self.results["memory_efficiency"].items():
                print(f"  {size:,} elements: {result['memory_used_mb']:.1f}MB, "
                      f"{result['elements_per_second']:,.0f} elements/sec")
        
        if "scheduler" in self.results:
            print("\nScheduler Optimization:")
            for workload_type, result in self.results["scheduler"].items():
                print(f"  {workload_type}: {result['optimal_cores']} cores")


if __name__ == "__main__":
    benchmark = BenchmarkSuite()
    benchmark.run_comprehensive_benchmark()
    benchmark.print_summary()