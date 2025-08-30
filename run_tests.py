#!/usr/bin/env python3
"""
Run all tests for ParallelCore library
"""

import sys
import os
import unittest
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_test_suite():
    """Run the complete test suite"""
    print("ParallelCore Library - Test Suite")
    print("=" * 50)
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    
    start_time = time.time()
    result = runner.run(suite)
    execution_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Execution time: {execution_time:.2f}s")
    
    if result.wasSuccessful():
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
        
        return 1


def run_specific_test(test_module: str):
    """Run a specific test module"""
    print(f"Running tests from: {test_module}")
    
    try:
        # Import the test module
        module = __import__(f"tests.{test_module}", fromlist=[test_module])
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
        
    except ImportError as e:
        print(f"Error importing test module: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1]
        if test_module.startswith("test_"):
            test_module = test_module[5:]  # Remove "test_" prefix
        sys.exit(run_specific_test(test_module))
    else:
        # Run all tests
        sys.exit(run_test_suite())