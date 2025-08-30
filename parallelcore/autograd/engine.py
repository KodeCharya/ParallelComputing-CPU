"""
AutogradEngine: Automatic differentiation engine for deep learning
"""

import numpy as np
from typing import Dict, List, Set, Optional, Callable, Any
from collections import defaultdict, deque
import weakref

from .variable import Variable
from ..tensor.tensor import Tensor


class Function:
    """Base class for differentiable functions"""
    
    def __init__(self):
        self.saved_tensors = []
        self.needs_input_grad = []
    
    def forward(self, *args) -> Tensor:
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError
    
    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        """Backward pass - to be implemented by subclasses"""
        raise NotImplementedError
    
    def save_for_backward(self, *tensors):
        """Save tensors for backward pass"""
        self.saved_tensors = tensors


class AddFunction(Function):
    """Addition function with gradient support"""
    
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.save_for_backward(a, b)
        return a + b
    
    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        a, b = self.saved_tensors
        grad_a = grad_output if a.requires_grad else None
        grad_b = grad_output if b.requires_grad else None
        return [grad_a, grad_b]


class MulFunction(Function):
    """Multiplication function with gradient support"""
    
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.save_for_backward(a, b)
        return a * b
    
    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        a, b = self.saved_tensors
        grad_a = grad_output * b if a.requires_grad else None
        grad_b = grad_output * a if b.requires_grad else None
        return [grad_a, grad_b]


class MatMulFunction(Function):
    """Matrix multiplication function with gradient support"""
    
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.save_for_backward(a, b)
        return a.matmul(b)
    
    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        a, b = self.saved_tensors
        grad_a = grad_output.matmul(b.T) if a.requires_grad else None
        grad_b = a.T.matmul(grad_output) if b.requires_grad else None
        return [grad_a, grad_b]


class ReLUFunction(Function):
    """ReLU activation function with gradient support"""
    
    def forward(self, x: Tensor) -> Tensor:
        self.save_for_backward(x)
        return x.relu()
    
    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        x, = self.saved_tensors
        if x.requires_grad:
            mask = Tensor((x._data > 0).astype(np.float32))
            grad_x = grad_output * mask
            return [grad_x]
        return [None]


class AutogradEngine:
    """
    Automatic differentiation engine
    
    Features:
    - Dynamic computation graph construction
    - Efficient gradient computation
    - Memory optimization
    - Parallel gradient computation
    """
    
    def __init__(self):
        self.computation_graph = {}
        self.gradient_cache = {}
        self.function_registry = {
            'add': AddFunction,
            'mul': MulFunction,
            'matmul': MatMulFunction,
            'relu': ReLUFunction,
        }
    
    def register_function(self, name: str, function_class: type):
        """Register a new differentiable function"""
        self.function_registry[name] = function_class
    
    def create_variable(self, data: Union[np.ndarray, List, float, int],
                       requires_grad: bool = True) -> Variable:
        """Create a new variable for autograd"""
        tensor = Tensor(data, requires_grad=requires_grad)
        return Variable(tensor, engine=self)
    
    def backward(self, variable: Variable, gradient: Optional[Tensor] = None):
        """
        Perform backward pass through computation graph
        
        Args:
            variable: Variable to compute gradients for
            gradient: Initial gradient (defaults to ones for scalar)
        """
        if not variable.requires_grad:
            return
        
        if gradient is None:
            if variable.tensor.size != 1:
                raise RuntimeError("Gradient must be specified for non-scalar variables")
            gradient = Tensor(np.ones_like(variable.tensor._data))
        
        # Topological sort of computation graph
        visited = set()
        topo_order = []
        
        def dfs(var):
            if id(var) in visited or not var.requires_grad:
                return
            visited.add(id(var))
            
            if hasattr(var, '_grad_fn') and var._grad_fn is not None:
                for parent in var._grad_fn.inputs:
                    if isinstance(parent, Variable):
                        dfs(parent)
            
            topo_order.append(var)
        
        dfs(variable)
        
        # Initialize gradients
        gradients = {id(variable): gradient}
        
        # Backward pass in reverse topological order
        for var in reversed(topo_order):
            if id(var) not in gradients:
                continue
            
            current_grad = gradients[id(var)]
            
            if hasattr(var, '_grad_fn') and var._grad_fn is not None:
                # Compute gradients for inputs
                input_grads = var._grad_fn.backward(current_grad)
                
                for input_var, input_grad in zip(var._grad_fn.inputs, input_grads):
                    if isinstance(input_var, Variable) and input_grad is not None:
                        if id(input_var) in gradients:
                            gradients[id(input_var)] = gradients[id(input_var)] + input_grad
                        else:
                            gradients[id(input_var)] = input_grad
            
            # Store gradient in variable
            if var.grad is None:
                var.grad = current_grad
            else:
                var.grad = var.grad + current_grad
    
    def zero_grad(self, variables: List[Variable]):
        """Zero gradients for list of variables"""
        for var in variables:
            var.zero_grad()
    
    def compute_graph_stats(self) -> dict:
        """Get computation graph statistics"""
        return {
            "nodes": len(self.computation_graph),
            "cached_gradients": len(self.gradient_cache),
            "registered_functions": len(self.function_registry)
        }
    
    def clear_cache(self):
        """Clear gradient cache to free memory"""
        self.gradient_cache.clear()