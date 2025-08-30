"""
Scheduler: Intelligent workload scheduling and resource management
"""

import psutil
import time
import math
from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class PerformanceMetrics:
    """Performance metrics for scheduling decisions"""
    throughput: float
    execution_time: float
    memory_usage: float
    cpu_utilization: float
    timestamp: float


class Scheduler:
    """
    Intelligent scheduler for optimal resource utilization
    
    Features:
    - Auto-detection of optimal core count
    - Dynamic workload balancing
    - Performance-based optimization
    - Memory-aware scheduling
    """
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.performance_history = deque(maxlen=max_history)
        self.physical_cores = psutil.cpu_count(logical=False)
        self.logical_cores = psutil.cpu_count(logical=True)
        self.total_memory = psutil.virtual_memory().total
        self._last_optimization = time.time()
        self._optimization_interval = 30.0  # seconds
    
    def get_optimal_core_count(self, workload_type: str = "cpu_bound") -> int:
        """
        Determine optimal number of cores based on system and workload
        
        Args:
            workload_type: Type of workload ("cpu_bound", "io_bound", "mixed")
            
        Returns:
            Optimal number of cores to use
        """
        if workload_type == "cpu_bound":
            # For CPU-bound tasks, use physical cores to avoid hyperthreading overhead
            return max(1, self.physical_cores - 1)  # Leave one core for system
        elif workload_type == "io_bound":
            # For I/O-bound tasks, can use more threads
            return min(self.logical_cores * 2, 32)  # Cap at reasonable limit
        else:  # mixed
            return self.logical_cores
    
    def calculate_chunk_size(self, data_size: int, num_cores: int, 
                           min_chunk_size: int = 1) -> int:
        """
        Calculate optimal chunk size for data parallelism
        
        Args:
            data_size: Total size of data to process
            num_cores: Number of cores available
            min_chunk_size: Minimum chunk size
            
        Returns:
            Optimal chunk size
        """
        if data_size <= num_cores:
            return max(min_chunk_size, 1)
        
        # Base chunk size
        base_chunk_size = math.ceil(data_size / num_cores)
        
        # Adjust based on system memory
        memory_factor = self._get_memory_factor()
        adjusted_chunk_size = int(base_chunk_size * memory_factor)
        
        # Ensure minimum chunk size
        return max(min_chunk_size, adjusted_chunk_size)
    
    def should_use_simd(self, array_size: int, operation_complexity: float = 1.0) -> bool:
        """
        Determine if SIMD operations would be beneficial
        
        Args:
            array_size: Size of arrays involved
            operation_complexity: Relative complexity of operation (1.0 = simple add)
            
        Returns:
            True if SIMD should be used
        """
        # SIMD is beneficial for larger arrays and complex operations
        simd_threshold = 1000 / operation_complexity
        return array_size >= simd_threshold
    
    def get_memory_strategy(self, required_memory: int) -> str:
        """
        Determine memory allocation strategy
        
        Args:
            required_memory: Required memory in bytes
            
        Returns:
            Memory strategy ("in_memory", "chunked", "streaming")
        """
        available_memory = psutil.virtual_memory().available
        memory_ratio = required_memory / available_memory
        
        if memory_ratio < 0.3:
            return "in_memory"
        elif memory_ratio < 0.7:
            return "chunked"
        else:
            return "streaming"
    
    def optimize_for_workload(self, workload_characteristics: dict) -> dict:
        """
        Optimize scheduling parameters based on workload characteristics
        
        Args:
            workload_characteristics: Dict with workload info
            
        Returns:
            Optimized parameters
        """
        current_time = time.time()
        
        # Only optimize if enough time has passed
        if current_time - self._last_optimization < self._optimization_interval:
            return self._get_default_params()
        
        self._last_optimization = current_time
        
        # Analyze recent performance
        if len(self.performance_history) < 10:
            return self._get_default_params()
        
        recent_metrics = list(self.performance_history)[-10:]
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_cpu_util = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
        
        # Adjust parameters based on performance
        params = self._get_default_params()
        
        if avg_cpu_util < 70:  # Underutilized
            params["num_cores"] = min(self.logical_cores, params["num_cores"] + 1)
        elif avg_cpu_util > 95:  # Overutilized
            params["num_cores"] = max(1, params["num_cores"] - 1)
        
        return params
    
    def update_performance_history(self, throughput: float, execution_time: float):
        """Update performance history for optimization"""
        metrics = PerformanceMetrics(
            throughput=throughput,
            execution_time=execution_time,
            memory_usage=psutil.virtual_memory().percent,
            cpu_utilization=psutil.cpu_percent(),
            timestamp=time.time()
        )
        self.performance_history.append(metrics)
    
    def _get_memory_factor(self) -> float:
        """Get memory adjustment factor for chunk sizing"""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent < 50:
            return 1.2  # Can use larger chunks
        elif memory_percent < 80:
            return 1.0  # Normal chunk size
        else:
            return 0.7  # Use smaller chunks to avoid memory pressure
    
    def _get_default_params(self) -> dict:
        """Get default scheduling parameters"""
        return {
            "num_cores": self.get_optimal_core_count(),
            "chunk_size": 1000,
            "use_simd": True,
            "memory_strategy": "in_memory"
        }
    
    def get_system_info(self) -> dict:
        """Get comprehensive system information"""
        return {
            "cpu": {
                "physical_cores": self.physical_cores,
                "logical_cores": self.logical_cores,
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "usage_percent": psutil.cpu_percent(interval=1, percpu=True)
            },
            "memory": {
                "total": self.total_memory,
                "available": psutil.virtual_memory().available,
                "percent_used": psutil.virtual_memory().percent
            },
            "scheduler": {
                "optimal_cores": self.get_optimal_core_count(),
                "performance_samples": len(self.performance_history)
            }
        }