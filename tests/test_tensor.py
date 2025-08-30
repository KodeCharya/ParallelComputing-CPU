"""
Tests for Tensor operations
"""

import unittest
import numpy as np
from parallelcore.tensor.tensor import Tensor


class TestTensor(unittest.TestCase):
    
    def test_tensor_creation(self):
        """Test tensor creation from various inputs"""
        # From list
        t1 = Tensor([1, 2, 3, 4])
        self.assertEqual(t1.shape, (4,))
        
        # From numpy array
        arr = np.array([[1, 2], [3, 4]])
        t2 = Tensor(arr)
        self.assertEqual(t2.shape, (2, 2))
        
        # From scalar
        t3 = Tensor(5.0)
        self.assertEqual(t3.shape, ())
    
    def test_tensor_arithmetic(self):
        """Test basic arithmetic operations"""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        
        # Addition
        c = a + b
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(c.data, expected)
        
        # Subtraction
        d = b - a
        expected = np.array([[4, 4], [4, 4]])
        np.testing.assert_array_equal(d.data, expected)
        
        # Multiplication
        e = a * b
        expected = np.array([[5, 12], [21, 32]])
        np.testing.assert_array_equal(e.data, expected)
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication"""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        
        c = a.matmul(b)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(c.data, expected)
        
        # Test @ operator
        d = a @ b
        np.testing.assert_array_equal(d.data, expected)
    
    def test_activation_functions(self):
        """Test activation functions"""
        x = Tensor([-2, -1, 0, 1, 2])
        
        # ReLU
        relu_result = x.relu()
        expected_relu = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(relu_result.data, expected_relu)
        
        # Sigmoid
        sigmoid_result = x.sigmoid()
        expected_sigmoid = 1 / (1 + np.exp(-x.data))
        np.testing.assert_array_almost_equal(sigmoid_result.data, expected_sigmoid)
    
    def test_tensor_reductions(self):
        """Test reduction operations"""
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        
        # Sum all elements
        total_sum = x.sum()
        self.assertEqual(total_sum.item(), 21)
        
        # Sum along axis
        axis_sum = x.sum(axis=0)
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(axis_sum.data, expected)
        
        # Mean
        mean_result = x.mean()
        self.assertEqual(mean_result.item(), 3.5)
    
    def test_tensor_reshaping(self):
        """Test tensor reshaping operations"""
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        
        # Reshape
        reshaped = x.reshape(3, 2)
        self.assertEqual(reshaped.shape, (3, 2))
        
        # Transpose
        transposed = x.transpose()
        self.assertEqual(transposed.shape, (3, 2))
        
        # T property
        t_prop = x.T
        self.assertEqual(t_prop.shape, (3, 2))
    
    def test_tensor_indexing(self):
        """Test tensor indexing and slicing"""
        x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Single element
        element = x[1, 1]
        self.assertEqual(element.item(), 5)
        
        # Row slice
        row = x[1]
        expected = np.array([4, 5, 6])
        np.testing.assert_array_equal(row.data, expected)
        
        # Column slice
        col = x[:, 1]
        expected = np.array([2, 5, 8])
        np.testing.assert_array_equal(col.data, expected)
    
    def test_gradient_tracking(self):
        """Test gradient tracking functionality"""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = Tensor([4, 5, 6], requires_grad=True)
        
        z = x + y
        self.assertTrue(z.requires_grad)
        
        loss = z.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
    
    def test_factory_methods(self):
        """Test tensor factory methods"""
        # Zeros
        zeros = Tensor.zeros(2, 3)
        self.assertEqual(zeros.shape, (2, 3))
        np.testing.assert_array_equal(zeros.data, np.zeros((2, 3)))
        
        # Ones
        ones = Tensor.ones(3, 2)
        self.assertEqual(ones.shape, (3, 2))
        np.testing.assert_array_equal(ones.data, np.ones((3, 2)))
        
        # Random
        randn = Tensor.randn(2, 2)
        self.assertEqual(randn.shape, (2, 2))
        
        # Arange
        arange = Tensor.arange(0, 10, 2)
        expected = np.array([0, 2, 4, 6, 8])
        np.testing.assert_array_equal(arange.data, expected)


if __name__ == "__main__":
    unittest.main()