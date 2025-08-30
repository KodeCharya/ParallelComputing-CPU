#!/usr/bin/env python3
"""
Run all ParallelCore examples and demonstrations
"""

import sys
import os
import time
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_example(module_name: str, description: str):
    """Run a single example module"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Module: {module_name}")
    print('='*60)
    
    try:
        start_time = time.time()
        
        # Import and run the example
        if module_name == "examples.basic_usage":
            from examples import basic_usage
            # The module runs automatically when imported
        elif module_name == "examples.neural_network_demo":
            from examples.neural_network_demo import train_classification_model
            train_classification_model()
        elif module_name == "examples.performance_benchmark":
            from examples.performance_benchmark import BenchmarkSuite
            benchmark = BenchmarkSuite()
            benchmark.run_comprehensive_benchmark()
            benchmark.print_summary()
        
        execution_time = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {execution_time:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Error running {description}:")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all examples"""
    print("ParallelCore Library - Example Runner")
    print("Demonstrating parallel computing, SIMD tensors, and neural networks")
    
    examples = [
        ("examples.basic_usage", "Basic Usage Examples"),
        ("examples.neural_network_demo", "Neural Network Training Demo"),
        ("examples.performance_benchmark", "Performance Benchmarks"),
    ]
    
    successful = 0
    total = len(examples)
    
    overall_start = time.time()
    
    for module_name, description in examples:
        if run_example(module_name, description):
            successful += 1
        
        # Small delay between examples
        time.sleep(1)
    
    overall_time = time.time() - overall_start
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Examples run: {successful}/{total}")
    print(f"Success rate: {(successful/total)*100:.1f}%")
    print(f"Total execution time: {overall_time:.2f}s")
    
    if successful == total:
        print("🎉 All examples completed successfully!")
        return 0
    else:
        print("⚠️  Some examples failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())