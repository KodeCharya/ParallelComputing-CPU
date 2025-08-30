"""
Tests for neural network components
"""

import unittest
import numpy as np
from parallelcore.autograd.engine import AutogradEngine
from parallelcore.nn.layers import Linear, ReLU, Sigmoid, Dropout, BatchNorm1d
from parallelcore.nn.loss import MSELoss, CrossEntropyLoss
from parallelcore.nn.optimizer import SGD, Adam


class TestNeuralNetworks(unittest.TestCase):
    
    def setUp(self):
        self.engine = AutogradEngine()
    
    def test_linear_layer(self):
        """Test linear layer functionality"""
        layer = Linear(10, 5)
        
        # Check parameter shapes
        self.assertEqual(layer.weight.shape, (5, 10))
        self.assertEqual(layer.bias.shape, (5,))
        
        # Test forward pass
        x = self.engine.create_variable(np.random.randn(3, 10))
        output = layer(x)
        
        self.assertEqual(output.shape, (3, 5))
    
    def test_activation_functions(self):
        """Test activation function layers"""
        x = self.engine.create_variable([[-1, 0, 1, 2]])
        
        # ReLU
        relu = ReLU()
        relu_output = relu(x)
        expected_relu = np.array([[0, 0, 1, 2]])
        np.testing.assert_array_equal(relu_output.data, expected_relu)
        
        # Sigmoid
        sigmoid = Sigmoid()
        sigmoid_output = sigmoid(x)
        expected_sigmoid = 1 / (1 + np.exp(-x.data))
        np.testing.assert_array_almost_equal(sigmoid_output.data, expected_sigmoid)
    
    def test_dropout_layer(self):
        """Test dropout layer"""
        dropout = Dropout(p=0.5)
        x = self.engine.create_variable(np.ones((10, 20)))
        
        # Training mode - should apply dropout
        dropout.train()
        output_train = dropout(x)
        
        # Some values should be zeroed out (with high probability)
        self.assertLess(np.sum(output_train.data), np.sum(x.data))
        
        # Evaluation mode - should not apply dropout
        dropout.eval()
        output_eval = dropout(x)
        np.testing.assert_array_equal(output_eval.data, x.data)
    
    def test_batch_norm(self):
        """Test batch normalization layer"""
        bn = BatchNorm1d(5)
        x = self.engine.create_variable(np.random.randn(10, 5))
        
        # Training mode
        bn.train()
        output = bn(x)
        
        # Output should have approximately zero mean and unit variance
        self.assertAlmostEqual(np.mean(output.data), 0, places=5)
        self.assertAlmostEqual(np.std(output.data), 1, places=1)
    
    def test_mse_loss(self):
        """Test MSE loss function"""
        criterion = MSELoss()
        
        pred = self.engine.create_variable([[1, 2], [3, 4]])
        target = self.engine.create_variable([[1.1, 1.9], [3.1, 3.9]], requires_grad=False)
        
        loss = criterion(pred, target)
        
        # Should be a scalar
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0)
    
    def test_cross_entropy_loss(self):
        """Test cross-entropy loss function"""
        criterion = CrossEntropyLoss()
        
        # Logits for 3 classes, 2 samples
        pred = self.engine.create_variable([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])
        target = self.engine.create_variable([0, 1], requires_grad=False)  # Class indices
        
        loss = criterion(pred, target)
        
        # Should be a scalar
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0)
    
    def test_sgd_optimizer(self):
        """Test SGD optimizer"""
        # Create a simple parameter
        param = self.engine.create_variable([[1.0, 2.0]])
        
        # Create optimizer
        optimizer = SGD([param], lr=0.1)
        
        # Simulate gradient
        param.grad = self.engine.create_variable([[0.1, 0.2]], requires_grad=False)
        
        # Store original values
        original_data = param.data.copy()
        
        # Optimization step
        optimizer.step()
        
        # Parameters should have changed
        expected = original_data - 0.1 * np.array([[0.1, 0.2]])
        np.testing.assert_array_almost_equal(param.data, expected)
    
    def test_adam_optimizer(self):
        """Test Adam optimizer"""
        # Create parameters
        param = self.engine.create_variable([[1.0, 2.0]])
        
        # Create optimizer
        optimizer = Adam([param], lr=0.01)
        
        # Simulate multiple optimization steps
        for i in range(5):
            param.grad = self.engine.create_variable([[0.1, 0.2]], requires_grad=False)
            optimizer.step()
        
        # Parameters should have changed
        self.assertNotEqual(param.data[0, 0], 1.0)
        self.assertNotEqual(param.data[0, 1], 2.0)
    
    def test_parameter_collection(self):
        """Test parameter collection from modules"""
        layer1 = Linear(10, 5)
        layer2 = Linear(5, 1)
        
        params1 = layer1.parameters()
        params2 = layer2.parameters()
        
        # Each linear layer should have weight and bias
        self.assertEqual(len(params1), 2)
        self.assertEqual(len(params2), 2)
        
        # Check parameter shapes
        self.assertEqual(params1[0].shape, (5, 10))  # weight
        self.assertEqual(params1[1].shape, (5,))     # bias


if __name__ == "__main__":
    unittest.main()