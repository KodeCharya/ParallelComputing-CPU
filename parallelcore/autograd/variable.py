"""
Variable: Wrapper for tensors with automatic differentiation
"""

import numpy as np
from typing import Optional, Union, List
import weakref

from ..tensor.tensor import Tensor


class GradientFunction:
    """Represents a function in the computation graph"""
    
    def __init__(self, function_name: str, inputs: List['Variable']):
        self.function_name = function_name
        self.inputs = inputs
        self.saved_values = {}
    
    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        """Compute gradients for inputs"""
        # This will be implemented by specific function classes
        raise NotImplementedError


class Variable:
    """
    Variable wrapper for tensors with automatic differentiation
    
    Similar to PyTorch's Variable (now integrated into Tensor)
    """
    
    def __init__(self, tensor: Tensor, requires_grad: bool = True, engine=None):
        self.tensor = tensor
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self.engine = engine
        
        # Ensure tensor also tracks gradients
        self.tensor.requires_grad = requires_grad
    
    @property
    def data(self) -> np.ndarray:
        """Get underlying data"""
        return self.tensor._data
    
    @property
    def shape(self) -> tuple:
        """Get tensor shape"""
        return self.tensor.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get tensor dtype"""
        return self.tensor.dtype
    
    def __repr__(self) -> str:
        return f"Variable({self.tensor}, requires_grad={self.requires_grad})"
    
    def __add__(self, other: Union['Variable', Tensor, float, int]) -> 'Variable':
        """Addition with gradient tracking"""
        if isinstance(other, Variable):
            result_tensor = self.tensor + other.tensor
            result = Variable(result_tensor, 
                            requires_grad=self.requires_grad or other.requires_grad,
                            engine=self.engine)
            
            if result.requires_grad:
                result._grad_fn = GradientFunction('add', [self, other])
            
            return result
        else:
            result_tensor = self.tensor + other
            result = Variable(result_tensor, requires_grad=self.requires_grad, engine=self.engine)
            
            if result.requires_grad:
                result._grad_fn = GradientFunction('add', [self])
            
            return result
    
    def __mul__(self, other: Union['Variable', Tensor, float, int]) -> 'Variable':
        """Multiplication with gradient tracking"""
        if isinstance(other, Variable):
            result_tensor = self.tensor * other.tensor
            result = Variable(result_tensor,
                            requires_grad=self.requires_grad or other.requires_grad,
                            engine=self.engine)
            
            if result.requires_grad:
                result._grad_fn = GradientFunction('mul', [self, other])
            
            return result
        else:
            result_tensor = self.tensor * other
            result = Variable(result_tensor, requires_grad=self.requires_grad, engine=self.engine)
            
            if result.requires_grad:
                result._grad_fn = GradientFunction('mul', [self])
            
            return result
    
    def __matmul__(self, other: 'Variable') -> 'Variable':
        """Matrix multiplication with gradient tracking"""
        result_tensor = self.tensor.matmul(other.tensor)
        result = Variable(result_tensor,
                        requires_grad=self.requires_grad or other.requires_grad,
                        engine=self.engine)
        
        if result.requires_grad:
            result._grad_fn = GradientFunction('matmul', [self, other])
        
        return result
    
    def relu(self) -> 'Variable':
        """ReLU activation with gradient tracking"""
        result_tensor = self.tensor.relu()
        result = Variable(result_tensor, requires_grad=self.requires_grad, engine=self.engine)
        
        if result.requires_grad:
            result._grad_fn = GradientFunction('relu', [self])
        
        return result
    
    def sigmoid(self) -> 'Variable':
        """Sigmoid activation with gradient tracking"""
        result_tensor = self.tensor.sigmoid()
        result = Variable(result_tensor, requires_grad=self.requires_grad, engine=self.engine)
        
        if result.requires_grad:
            result._grad_fn = GradientFunction('sigmoid', [self])
        
        return result
    
    def sum(self, axis: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> 'Variable':
        """Sum with gradient tracking"""
        result_tensor = self.tensor.sum(axis=axis, keepdims=keepdims)
        result = Variable(result_tensor, requires_grad=self.requires_grad, engine=self.engine)
        
        if result.requires_grad:
            result._grad_fn = GradientFunction('sum', [self])
        
        return result
    
    def mean(self, axis: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> 'Variable':
        """Mean with gradient tracking"""
        result_tensor = self.tensor.mean(axis=axis, keepdims=keepdims)
        result = Variable(result_tensor, requires_grad=self.requires_grad, engine=self.engine)
        
        if result.requires_grad:
            result._grad_fn = GradientFunction('mean', [self])
        
        return result
    
    def backward(self, gradient: Optional[Tensor] = None):
        """Compute gradients via backpropagation"""
        if self.engine:
            self.engine.backward(self, gradient)
        else:
            # Fallback to tensor's backward method
            self.tensor.backward(gradient)
    
    def zero_grad(self):
        """Zero out gradients"""
        self.grad = None
        self.tensor.zero_grad()
    
    def detach(self) -> 'Variable':
        """Detach from computation graph"""
        return Variable(self.tensor.detach(), requires_grad=False, engine=self.engine)
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return self.tensor.numpy()
    
    def item(self) -> Union[float, int]:
        """Get scalar value"""
        return self.tensor.item()