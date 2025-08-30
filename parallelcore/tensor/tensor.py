"""
Tensor: High-performance CPU tensor with SIMD acceleration
"""

import numpy as np
from typing import Union, Tuple, List, Optional, Any
import warnings

from .ops import TensorOps
from ..core.scheduler import Scheduler


class Tensor:
    """
    High-performance CPU tensor with SIMD acceleration
    
    Features:
    - NumPy-compatible interface
    - SIMD-accelerated operations via Numba
    - Automatic memory layout optimization
    - Broadcasting support
    - Gradient tracking for autograd
    """
    
    def __init__(self, data: Union[np.ndarray, List, float, int], 
                 dtype: Optional[np.dtype] = None,
                 requires_grad: bool = False,
                 device: str = "cpu"):
        
        # Convert input to numpy array
        if isinstance(data, Tensor):
            self._data = data._data.copy()
        elif isinstance(data, np.ndarray):
            self._data = data.copy()
        else:
            self._data = np.array(data, dtype=dtype)
        
        # Ensure contiguous memory layout for SIMD
        if not self._data.flags.c_contiguous:
            self._data = np.ascontiguousarray(self._data)
        
        self.requires_grad = requires_grad
        self.device = device
        self.grad = None
        self._grad_fn = None
        
        # Initialize operations handler
        self.ops = TensorOps()
        self.scheduler = Scheduler()
    
    @property
    def data(self) -> np.ndarray:
        """Get underlying numpy array"""
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape"""
        return self._data.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get tensor data type"""
        return self._data.dtype
    
    @property
    def size(self) -> int:
        """Get total number of elements"""
        return self._data.size
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions"""
        return self._data.ndim
    
    def __repr__(self) -> str:
        return f"Tensor({self._data}, requires_grad={self.requires_grad})"
    
    def __str__(self) -> str:
        return str(self._data)
    
    # Arithmetic operations with SIMD acceleration
    def __add__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise addition with SIMD acceleration"""
        other_data = other._data if isinstance(other, Tensor) else other
        
        if self.scheduler.should_use_simd(self.size):
            result_data = self.ops.add_simd(self._data, other_data)
        else:
            result_data = self._data + other_data
        
        result = Tensor(result_data, requires_grad=self.requires_grad or 
                       (isinstance(other, Tensor) and other.requires_grad))
        
        if result.requires_grad:
            result._grad_fn = ('add', self, other)
        
        return result
    
    def __sub__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise subtraction with SIMD acceleration"""
        other_data = other._data if isinstance(other, Tensor) else other
        
        if self.scheduler.should_use_simd(self.size):
            result_data = self.ops.sub_simd(self._data, other_data)
        else:
            result_data = self._data - other_data
        
        result = Tensor(result_data, requires_grad=self.requires_grad or 
                       (isinstance(other, Tensor) and other.requires_grad))
        
        if result.requires_grad:
            result._grad_fn = ('sub', self, other)
        
        return result
    
    def __mul__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise multiplication with SIMD acceleration"""
        other_data = other._data if isinstance(other, Tensor) else other
        
        if self.scheduler.should_use_simd(self.size):
            result_data = self.ops.mul_simd(self._data, other_data)
        else:
            result_data = self._data * other_data
        
        result = Tensor(result_data, requires_grad=self.requires_grad or 
                       (isinstance(other, Tensor) and other.requires_grad))
        
        if result.requires_grad:
            result._grad_fn = ('mul', self, other)
        
        return result
    
    def __truediv__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise division"""
        other_data = other._data if isinstance(other, Tensor) else other
        result_data = self._data / other_data
        
        result = Tensor(result_data, requires_grad=self.requires_grad or 
                       (isinstance(other, Tensor) and other.requires_grad))
        
        if result.requires_grad:
            result._grad_fn = ('div', self, other)
        
        return result
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication with parallel acceleration"""
        return self.matmul(other)
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication with parallel processing
        
        Args:
            other: Tensor to multiply with
            
        Returns:
            Result tensor
        """
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication requires another Tensor")
        
        # Use parallel matrix multiplication for large matrices
        if self.size > 10000 or other.size > 10000:
            result_data = self.ops.matmul_parallel(self._data, other._data)
        else:
            result_data = np.matmul(self._data, other._data)
        
        result = Tensor(result_data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = ('matmul', self, other)
        
        return result
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, 
            keepdims: bool = False) -> 'Tensor':
        """Sum reduction with parallel processing"""
        if self.size > 10000:
            result_data = self.ops.sum_parallel(self._data, axis, keepdims)
        else:
            result_data = np.sum(self._data, axis=axis, keepdims=keepdims)
        
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = ('sum', self, axis, keepdims)
        
        return result
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False) -> 'Tensor':
        """Mean reduction"""
        result_data = np.mean(self._data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = ('mean', self, axis, keepdims)
        
        return result
    
    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape tensor"""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        
        result_data = self._data.reshape(shape)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = ('reshape', self, self.shape)
        
        return result
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """Transpose tensor"""
        result_data = np.transpose(self._data, axes)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = ('transpose', self, axes)
        
        return result
    
    @property
    def T(self) -> 'Tensor':
        """Transpose property"""
        return self.transpose()
    
    def relu(self) -> 'Tensor':
        """ReLU activation function"""
        if self.scheduler.should_use_simd(self.size):
            result_data = self.ops.relu_simd(self._data)
        else:
            result_data = np.maximum(0, self._data)
        
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = ('relu', self)
        
        return result
    
    def sigmoid(self) -> 'Tensor':
        """Sigmoid activation function"""
        if self.scheduler.should_use_simd(self.size):
            result_data = self.ops.sigmoid_simd(self._data)
        else:
            result_data = 1 / (1 + np.exp(-self._data))
        
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = ('sigmoid', self)
        
        return result
    
    def tanh(self) -> 'Tensor':
        """Tanh activation function"""
        result_data = np.tanh(self._data)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = ('tanh', self)
        
        return result
    
    def exp(self) -> 'Tensor':
        """Exponential function"""
        result_data = np.exp(self._data)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = ('exp', self)
        
        return result
    
    def log(self) -> 'Tensor':
        """Natural logarithm"""
        result_data = np.log(self._data)
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = ('log', self)
        
        return result
    
    def backward(self, gradient: Optional['Tensor'] = None):
        """Backward pass for gradient computation"""
        if not self.requires_grad:
            return
        
        if gradient is None:
            if self.size != 1:
                raise RuntimeError("Gradient must be specified for non-scalar tensors")
            gradient = Tensor(np.ones_like(self._data))
        
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self._data))
        
        self.grad = self.grad + gradient
        
        # Propagate gradients through computation graph
        if self._grad_fn is not None:
            self._backward_through_graph(gradient)
    
    def _backward_through_graph(self, gradient: 'Tensor'):
        """Propagate gradients through computation graph"""
        op, *args = self._grad_fn
        
        if op == 'add':
            self_tensor, other = args
            if isinstance(self_tensor, Tensor) and self_tensor.requires_grad:
                self_tensor.backward(gradient)
            if isinstance(other, Tensor) and other.requires_grad:
                self_tensor.backward(gradient)
        
        elif op == 'mul':
            self_tensor, other = args
            if isinstance(self_tensor, Tensor) and self_tensor.requires_grad:
                other_data = other._data if isinstance(other, Tensor) else other
                grad = gradient * Tensor(other_data)
                self_tensor.backward(grad)
            if isinstance(other, Tensor) and other.requires_grad:
                grad = gradient * self_tensor
                other.backward(grad)
        
        elif op == 'matmul':
            self_tensor, other = args
            if self_tensor.requires_grad:
                grad = gradient.matmul(other.transpose())
                self_tensor.backward(grad)
            if other.requires_grad:
                grad = self_tensor.transpose().matmul(gradient)
                other.backward(grad)
        
        elif op == 'relu':
            self_tensor = args[0]
            if self_tensor.requires_grad:
                mask = Tensor((self_tensor._data > 0).astype(np.float32))
                grad = gradient * mask
                self_tensor.backward(grad)
        
        elif op == 'sigmoid':
            self_tensor = args[0]
            if self_tensor.requires_grad:
                sigmoid_val = self
                grad = gradient * sigmoid_val * (Tensor(np.ones_like(sigmoid_val._data)) - sigmoid_val)
                self_tensor.backward(grad)
    
    def zero_grad(self):
        """Zero out gradients"""
        if self.grad is not None:
            self.grad = None
    
    def detach(self) -> 'Tensor':
        """Detach tensor from computation graph"""
        result = Tensor(self._data, requires_grad=False)
        return result
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return self._data.copy()
    
    def item(self) -> Union[float, int]:
        """Get scalar value"""
        if self.size != 1:
            raise ValueError("Only single-element tensors can be converted to scalars")
        return self._data.item()
    
    # Factory methods
    @staticmethod
    def zeros(*shape: int, dtype: np.dtype = np.float32, requires_grad: bool = False) -> 'Tensor':
        """Create tensor filled with zeros"""
        data = np.zeros(shape, dtype=dtype)
        return Tensor(data, requires_grad=requires_grad)
    
    @staticmethod
    def ones(*shape: int, dtype: np.dtype = np.float32, requires_grad: bool = False) -> 'Tensor':
        """Create tensor filled with ones"""
        data = np.ones(shape, dtype=dtype)
        return Tensor(data, requires_grad=requires_grad)
    
    @staticmethod
    def randn(*shape: int, dtype: np.dtype = np.float32, requires_grad: bool = False) -> 'Tensor':
        """Create tensor with random normal distribution"""
        data = np.random.randn(*shape).astype(dtype)
        return Tensor(data, requires_grad=requires_grad)
    
    @staticmethod
    def rand(*shape: int, dtype: np.dtype = np.float32, requires_grad: bool = False) -> 'Tensor':
        """Create tensor with random uniform distribution"""
        data = np.random.rand(*shape).astype(dtype)
        return Tensor(data, requires_grad=requires_grad)
    
    @staticmethod
    def arange(start: float, stop: Optional[float] = None, step: float = 1,
               dtype: np.dtype = np.float32, requires_grad: bool = False) -> 'Tensor':
        """Create tensor with evenly spaced values"""
        if stop is None:
            stop = start
            start = 0
        data = np.arange(start, stop, step, dtype=dtype)
        return Tensor(data, requires_grad=requires_grad)
    
    @staticmethod
    def linspace(start: float, stop: float, num: int = 50,
                 dtype: np.dtype = np.float32, requires_grad: bool = False) -> 'Tensor':
        """Create tensor with linearly spaced values"""
        data = np.linspace(start, stop, num, dtype=dtype)
        return Tensor(data, requires_grad=requires_grad)
    
    # Indexing and slicing
    def __getitem__(self, key) -> 'Tensor':
        """Get tensor slice"""
        result_data = self._data[key]
        result = Tensor(result_data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._grad_fn = ('getitem', self, key)
        
        return result
    
    def __setitem__(self, key, value):
        """Set tensor slice"""
        if isinstance(value, Tensor):
            self._data[key] = value._data
        else:
            self._data[key] = value