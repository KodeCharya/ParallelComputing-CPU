"""
Tests for VirtualCoreManager
"""

import unittest
import numpy as np
import time
from parallelcore.core.virtual_core_manager import VirtualCoreManager


class TestVirtualCoreManager(unittest.TestCase):
    
    def setUp(self):
        self.manager = VirtualCoreManager(num_cores=2)
    
    def tearDown(self):
        self.manager.shutdown()
    
    def test_map_parallel_basic(self):
        """Test basic parallel map functionality"""
        data = list(range(100))
        
        def square(x):
            return x * x
        
        result = self.manager.map_parallel(square, data)
        expected = [x * x for x in data]
        
        self.assertEqual(result, expected)
    
    def test_map_parallel_empty(self):
        """Test parallel map with empty data"""
        result = self.manager.map_parallel(lambda x: x, [])
        self.assertEqual(result, [])
    
    def test_reduce_parallel_sum(self):
        """Test parallel reduction with sum"""
        data = list(range(1000))
        
        result = self.manager.reduce_parallel(lambda x, y: x + y, data)
        expected = sum(data)
        
        self.assertEqual(result, expected)
    
    def test_reduce_parallel_with_initial(self):
        """Test parallel reduction with initial value"""
        data = [1, 2, 3, 4, 5]
        
        result = self.manager.reduce_parallel(lambda x, y: x + y, data, initial_value=10)
        expected = 25  # 10 + 1 + 2 + 3 + 4 + 5
        
        self.assertEqual(result, expected)
    
    def test_execute_batch(self):
        """Test batch execution of different tasks"""
        def add(a, b):
            return a + b
        
        def multiply(a, b):
            return a * b
        
        tasks = [
            (add, (1, 2), {}),
            (multiply, (3, 4), {}),
            (add, (5, 6), {})
        ]
        
        results = self.manager.execute_batch(tasks)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].result, 3)
        self.assertEqual(results[1].result, 12)
        self.assertEqual(results[2].result, 11)
    
    def test_core_stats(self):
        """Test core statistics retrieval"""
        stats = self.manager.get_core_stats()
        
        self.assertIn("total_cores", stats)
        self.assertIn("physical_cores", stats)
        self.assertIn("logical_cores", stats)
        self.assertIn("cpu_percent", stats)
        self.assertIn("memory_percent", stats)
        self.assertIn("core_utilization", stats)
        
        self.assertEqual(stats["total_cores"], 2)
        self.assertIsInstance(stats["cpu_percent"], (int, float))
        self.assertIsInstance(stats["memory_percent"], (int, float))


if __name__ == "__main__":
    unittest.main()