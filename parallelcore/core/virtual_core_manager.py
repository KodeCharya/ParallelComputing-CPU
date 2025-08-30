"""
VirtualCoreManager: Manages parallel task execution across CPU cores
"""

import multiprocessing as mp
import threading
import queue
import time
from typing import Callable, List, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import psutil
import numpy as np

from .scheduler import Scheduler


class VirtualCore:
    """Represents a virtual CPU core for task execution"""
    
    def __init__(self, core_id: int, affinity: Optional[List[int]] = None):
        self.core_id = core_id
        self.affinity = affinity
        self.is_busy = False
        self.current_task = None
        self.completed_tasks = 0
        self.total_execution_time = 0.0
    
    def get_utilization(self) -> float:
        """Get core utilization percentage"""
        if self.completed_tasks == 0:
            return 0.0
        return min(100.0, (self.total_execution_time / time.time()) * 100)


class TaskResult:
    """Container for task execution results"""
    
    def __init__(self, task_id: int, result: Any, execution_time: float, core_id: int):
        self.task_id = task_id
        self.result = result
        self.execution_time = execution_time
        self.core_id = core_id
        self.timestamp = time.time()


class VirtualCoreManager:
    """
    High-level manager for parallel task execution across CPU cores
    
    Features:
    - Automatic core detection and optimization
    - Intelligent workload balancing
    - Task scheduling and result aggregation
    - Performance monitoring and statistics
    """
    
    def __init__(self, num_cores: Optional[int] = None, use_threads: bool = False):
        self.scheduler = Scheduler()
        self.num_cores = num_cores or self.scheduler.get_optimal_core_count()
        self.use_threads = use_threads
        self.virtual_cores = [VirtualCore(i) for i in range(self.num_cores)]
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        self.executor = None
        self._setup_executor()
    
    def _setup_executor(self):
        """Initialize the appropriate executor based on configuration"""
        if self.use_threads:
            self.executor = ThreadPoolExecutor(max_workers=self.num_cores)
        else:
            self.executor = ProcessPoolExecutor(max_workers=self.num_cores)
    
    def map_parallel(self, func: Callable, data: List[Any], 
                    chunk_size: Optional[int] = None) -> List[Any]:
        """
        Apply function to data in parallel across virtual cores
        
        Args:
            func: Function to apply to each data element
            data: List of data to process
            chunk_size: Size of chunks to process (auto-calculated if None)
            
        Returns:
            List of results in original order
        """
        if not data:
            return []
        
        # Calculate optimal chunk size
        if chunk_size is None:
            chunk_size = self.scheduler.calculate_chunk_size(len(data), self.num_cores)
        
        # Split data into chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Process chunks in parallel
        futures = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            future = self.executor.submit(self._process_chunk, func, chunk, i)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            chunk_result = future.result()
            results.extend(chunk_result.result)
        
        execution_time = time.time() - start_time
        self._update_performance_stats(execution_time, len(data))
        
        return results
    
    def reduce_parallel(self, func: Callable, data: List[Any], 
                       initial_value: Any = None) -> Any:
        """
        Parallel reduction operation
        
        Args:
            func: Binary function for reduction (e.g., lambda x, y: x + y)
            data: Data to reduce
            initial_value: Starting value for reduction
            
        Returns:
            Single reduced value
        """
        if not data:
            return initial_value
        
        # For small datasets, use sequential reduction
        if len(data) < self.num_cores * 2:
            result = initial_value if initial_value is not None else data[0]
            start_idx = 0 if initial_value is not None else 1
            for item in data[start_idx:]:
                result = func(result, item)
            return result
        
        # Parallel tree reduction
        current_data = list(data)
        if initial_value is not None:
            current_data.insert(0, initial_value)
        
        while len(current_data) > 1:
            chunk_size = max(2, len(current_data) // self.num_cores)
            chunks = [current_data[i:i + chunk_size] 
                     for i in range(0, len(current_data), chunk_size)]
            
            futures = []
            for chunk in chunks:
                future = self.executor.submit(self._reduce_chunk, func, chunk)
                futures.append(future)
            
            current_data = [future.result() for future in as_completed(futures)]
        
        return current_data[0]
    
    def execute_batch(self, tasks: List[tuple]) -> List[TaskResult]:
        """
        Execute a batch of different tasks in parallel
        
        Args:
            tasks: List of (function, args, kwargs) tuples
            
        Returns:
            List of TaskResult objects
        """
        futures = []
        start_time = time.time()
        
        for i, (func, args, kwargs) in enumerate(tasks):
            future = self.executor.submit(self._execute_task, func, args, kwargs, i)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            results.append(future.result())
        
        # Sort by task_id to maintain order
        results.sort(key=lambda x: x.task_id)
        return results
    
    def _process_chunk(self, func: Callable, chunk: List[Any], chunk_id: int) -> TaskResult:
        """Process a single chunk of data"""
        start_time = time.time()
        
        try:
            # Apply function to each item in chunk
            results = [func(item) for item in chunk]
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=chunk_id,
                result=results,
                execution_time=execution_time,
                core_id=mp.current_process().pid % self.num_cores
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=chunk_id,
                result=e,
                execution_time=execution_time,
                core_id=mp.current_process().pid % self.num_cores
            )
    
    def _reduce_chunk(self, func: Callable, chunk: List[Any]) -> Any:
        """Reduce a single chunk"""
        if len(chunk) == 1:
            return chunk[0]
        
        result = chunk[0]
        for item in chunk[1:]:
            result = func(result, item)
        return result
    
    def _execute_task(self, func: Callable, args: tuple, kwargs: dict, task_id: int) -> TaskResult:
        """Execute a single task"""
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_id,
                result=result,
                execution_time=execution_time,
                core_id=mp.current_process().pid % self.num_cores
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task_id,
                result=e,
                execution_time=execution_time,
                core_id=mp.current_process().pid % self.num_cores
            )
    
    def _update_performance_stats(self, execution_time: float, data_size: int):
        """Update performance statistics"""
        throughput = data_size / execution_time if execution_time > 0 else 0
        self.scheduler.update_performance_history(throughput, execution_time)
    
    def get_core_stats(self) -> dict:
        """Get statistics for all virtual cores"""
        return {
            "total_cores": self.num_cores,
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "core_utilization": [core.get_utilization() for core in self.virtual_cores]
        }
    
    def shutdown(self):
        """Gracefully shutdown the core manager"""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.is_running = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()