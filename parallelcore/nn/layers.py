"""
Neural network layers with automatic differentiation
"""

import numpy as np
from typing import Optional, Callable
import math

from ..autograd.variable import Variable
from ..tensor.tensor import Tensor


class Module:
    """Base class for neural network modules"""
    
    def __init__(self):
        self.training = True
        self._parameters = {}
        self._modules = {}
    
    def forward(self, x: Variable) -> Variable:
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError
    
    def __call__(self, x: Variable) -> Variable:
        """Make module callable"""
        return self.forward(x)
    
    def parameters(self) -> list:
        """Get all parameters"""
        params = []
        for param in self._parameters.values():
            params.append(param)
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def train(self, mode: bool = True):
        """Set training mode"""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
    
    def eval(self):
        """Set evaluation mode"""
        self.train(False)
    
    def zero_grad(self):
        """Zero gradients for all parameters"""
        for param in self.parameters():
            param.zero_grad()


class Linear(Module):
    """
    Linear (fully connected) layer
    
    Performs: y = xW^T + b
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weights with Xavier/Glorot initialization
        weight_data = np.random.randn(out_features, in_features) * math.sqrt(2.0 / in_features)
        self.weight = Variable(Tensor(weight_data, requires_grad=True))
        self._parameters['weight'] = self.weight
        
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Variable(Tensor(bias_data, requires_grad=True))
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, x: Variable) -> Variable:
        """Forward pass"""
        # x @ W^T + b
        output = x @ self.weight.tensor.T
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"


class ReLU(Module):
    """ReLU activation function"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Variable) -> Variable:
        """Forward pass"""
        return x.relu()
    
    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation function"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Variable) -> Variable:
        """Forward pass"""
        return x.sigmoid()
    
    def __repr__(self) -> str:
        return "Sigmoid()"


class Softmax(Module):
    """Softmax activation function"""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Variable) -> Variable:
        """Forward pass"""
        # Numerically stable softmax
        x_max = Variable(Tensor(np.max(x.data, axis=self.dim, keepdims=True)))
        x_shifted = x - x_max
        exp_x = Variable(Tensor(np.exp(x_shifted.data)))
        sum_exp = exp_x.sum(axis=self.dim, keepdims=True)
        return exp_x / sum_exp
    
    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class Dropout(Module):
    """Dropout regularization layer"""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: Variable) -> Variable:
        """Forward pass"""
        if not self.training:
            return x
        
        # Generate dropout mask
        mask = np.random.binomial(1, 1 - self.p, x.shape) / (1 - self.p)
        mask_tensor = Tensor(mask.astype(x.dtype))
        
        return x * Variable(mask_tensor, requires_grad=False)
    
    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class BatchNorm1d(Module):
    """1D Batch normalization layer"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = Variable(Tensor(np.ones(num_features), requires_grad=True))
        self.bias = Variable(Tensor(np.zeros(num_features), requires_grad=True))
        self._parameters['weight'] = self.weight
        self._parameters['bias'] = self.bias
        
        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: Variable) -> Variable:
        """Forward pass"""
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(axis=0)
            batch_var = ((x - batch_mean) ** 2).mean(axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
            
            # Normalize
            x_norm = (x - batch_mean) / Variable(Tensor(np.sqrt(batch_var.data + self.eps)))
        else:
            # Use running statistics
            x_norm = (x - Variable(Tensor(self.running_mean))) / Variable(Tensor(np.sqrt(self.running_var + self.eps)))
        
        # Scale and shift
        return self.weight * x_norm + self.bias
    
    def __repr__(self) -> str:
        return f"BatchNorm1d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"