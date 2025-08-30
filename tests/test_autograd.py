"""
Tests for automatic differentiation engine
"""

import unittest
import numpy as np
from parallelcore.autograd.engine import AutogradEngine
from parallelcore.autograd.variable import Variable
from parallelcore.tensor.tensor import Tensor


class TestAutograd(unittest.TestCase):
    
    def setUp(self):
        self.engine = AutogradEngine()
    
    def test_variable_creation(self):
        """Test variable creation"""
        x = self.engine.create_variable([1, 2, 3])
        self.assertTrue(x.requires_grad)
        self.assertEqual(x.shape, (3,))
    
    def test_simple_addition_gradient(self):
        """Test gradient computation for addition"""
        x = self.engine.create_variable([2.0])
        y = self.engine.create_variable([3.0])
        
        z = x + y
        z.backward()
        
        # Gradients should be 1 for both inputs
        np.testing.assert_array_equal(x.grad.data, [1.0])
        np.testing.assert_array_equal(y.grad.data, [1.0])
    
    def test_multiplication_gradient(self):
        """Test gradient computation for multiplication"""
        x = self.engine.create_variable([2.0])
        y = self.engine.create_variable([3.0])
        
        z = x * y
        z.backward()
        
        # dz/dx = y, dz/dy = x
        np.testing.assert_array_equal(x.grad.data, [3.0])
        np.testing.assert_array_equal(y.grad.data, [2.0])
    
    def test_chain_rule(self):
        """Test chain rule with complex computation"""
        x = self.engine.create_variable([2.0])
        
        # z = (x + 1) * (x + 2) = x^2 + 3x + 2
        # dz/dx = 2x + 3 = 2*2 + 3 = 7
        y = x + Variable(Tensor([1.0]), requires_grad=False)
        z = x + Variable(Tensor([2.0]), requires_grad=False)
        result = y * z
        
        result.backward()
        
        # At x=2, gradient should be 7
        np.testing.assert_array_almost_equal(x.grad.data, [7.0])
    
    def test_matrix_multiplication_gradient(self):
        """Test gradient computation for matrix multiplication"""
        x = self.engine.create_variable([[1, 2], [3, 4]])
        w = self.engine.create_variable([[0.5, 0.5], [0.5, 0.5]])
        
        y = x @ w
        loss = y.sum()
        loss.backward()
        
        # Check that gradients have correct shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(w.grad.shape, w.shape)
    
    def test_relu_gradient(self):
        """Test ReLU gradient computation"""
        x = self.engine.create_variable([-1, 0, 1, 2])
        
        y = x.relu()
        loss = y.sum()
        loss.backward()
        
        # ReLU gradient: 0 for x <= 0, 1 for x > 0
        expected_grad = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(x.grad.data, expected_grad)
    
    def test_zero_grad(self):
        """Test gradient zeroing"""
        x = self.engine.create_variable([1, 2, 3])
        y = x.sum()
        y.backward()
        
        # Check gradient exists
        self.assertIsNotNone(x.grad)
        
        # Zero gradient
        x.zero_grad()
        self.assertIsNone(x.grad)
    
    def test_detach(self):
        """Test tensor detachment from computation graph"""
        x = self.engine.create_variable([1, 2, 3])
        y = x.detach()
        
        self.assertFalse(y.requires_grad)
        
        z = y.sum()
        z.backward()  # Should not affect x
        
        self.assertIsNone(x.grad)


if __name__ == "__main__":
    unittest.main()