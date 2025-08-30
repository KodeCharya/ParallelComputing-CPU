"""
Basic usage examples for ParallelCore library
"""

import numpy as np
import time
from parallelcore import VirtualCoreManager, Tensor, Variable, AutogradEngine
from parallelcore.nn import Linear, ReLU, MSELoss, Adam


def example_parallel_processing():
    """Demonstrate parallel task processing"""
    print("=== Parallel Processing Example ===")
    
    # Create some sample data
    data = list(range(10000))
    
    def expensive_computation(x):
        """Simulate expensive computation"""
        return sum(i * i for i in range(x % 100))
    
    # Sequential processing
    start_time = time.time()
    sequential_results = [expensive_computation(x) for x in data]
    sequential_time = time.time() - start_time
    
    # Parallel processing
    with VirtualCoreManager() as manager:
        start_time = time.time()
        parallel_results = manager.map_parallel(expensive_computation, data)
        parallel_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.3f}s")
    print(f"Parallel time: {parallel_time:.3f}s")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")
    print(f"Results match: {sequential_results == parallel_results}")
    print()


def example_tensor_operations():
    """Demonstrate SIMD-accelerated tensor operations"""
    print("=== Tensor Operations Example ===")
    
    # Create large tensors
    a = Tensor.randn(1000, 1000)
    b = Tensor.randn(1000, 1000)
    
    # Basic operations
    start_time = time.time()
    c = a + b
    d = a * b
    e = a.matmul(b)
    tensor_time = time.time() - start_time
    
    print(f"Tensor operations time: {tensor_time:.3f}s")
    print(f"Result shapes: add={c.shape}, mul={d.shape}, matmul={e.shape}")
    
    # Activation functions
    x = Tensor.randn(1000, 500)
    relu_result = x.relu()
    sigmoid_result = x.sigmoid()
    
    print(f"Activation shapes: relu={relu_result.shape}, sigmoid={sigmoid_result.shape}")
    print()


def example_neural_network():
    """Demonstrate neural network with autograd"""
    print("=== Neural Network Example ===")
    
    # Create autograd engine
    engine = AutogradEngine()
    
    # Generate sample data
    np.random.seed(42)
    X_data = np.random.randn(100, 10)
    y_data = np.random.randn(100, 1)
    
    X = engine.create_variable(X_data)
    y = engine.create_variable(y_data, requires_grad=False)
    
    # Create simple neural network
    layer1 = Linear(10, 20)
    activation = ReLU()
    layer2 = Linear(20, 1)
    
    # Loss and optimizer
    criterion = MSELoss()
    optimizer = Adam(layer1.parameters() + layer2.parameters(), lr=0.01)
    
    # Training loop
    print("Training neural network...")
    for epoch in range(10):
        # Forward pass
        hidden = activation(layer1(X))
        output = layer2(hidden)
        
        # Compute loss
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    print("Training completed!")
    print()


def example_system_monitoring():
    """Demonstrate system monitoring and optimization"""
    print("=== System Monitoring Example ===")
    
    with VirtualCoreManager() as manager:
        # Get system information
        system_info = manager.get_core_stats()
        
        print("System Information:")
        print(f"  Physical cores: {system_info['physical_cores']}")
        print(f"  Logical cores: {system_info['logical_cores']}")
        print(f"  CPU usage: {system_info['cpu_percent']:.1f}%")
        print(f"  Memory usage: {system_info['memory_percent']:.1f}%")
        print(f"  Virtual cores: {system_info['total_cores']}")
        
        # Demonstrate workload balancing
        def cpu_intensive_task(n):
            return sum(i * i for i in range(n))
        
        tasks = [(cpu_intensive_task, (1000,), {}) for _ in range(20)]
        results = manager.execute_batch(tasks)
        
        print(f"\nExecuted {len(tasks)} tasks")
        print(f"Average execution time: {np.mean([r.execution_time for r in results]):.4f}s")
        print(f"Total execution time: {sum(r.execution_time for r in results):.4f}s")
    
    print()


def example_advanced_tensor_ops():
    """Demonstrate advanced tensor operations"""
    print("=== Advanced Tensor Operations Example ===")
    
    # Create tensors with gradient tracking
    x = Tensor.randn(64, 128, requires_grad=True)
    w = Tensor.randn(128, 64, requires_grad=True)
    
    # Complex computation
    y = x.matmul(w)
    z = y.relu()
    loss = z.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {w.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Input gradient shape: {x.grad.shape if x.grad else 'None'}")
    print(f"Weight gradient shape: {w.grad.shape if w.grad else 'None'}")
    print()


if __name__ == "__main__":
    print("ParallelCore Library Examples")
    print("=" * 40)
    
    example_parallel_processing()
    example_tensor_operations()
    example_neural_network()
    example_system_monitoring()
    example_advanced_tensor_ops()
    
    print("All examples completed successfully!")