"""
ParallelCore: High-performance parallel computing library for CPU

A comprehensive library providing:
- VirtualCoreManager for task parallelism
- SIMD-accelerated Tensor operations
- Intelligent workload scheduling
- Autograd engine for deep learning
- Neural network primitives
"""

from .core.virtual_core_manager import VirtualCoreManager
from .core.scheduler import Scheduler
from .tensor.tensor import Tensor
from .tensor.ops import TensorOps
from .autograd.engine import AutogradEngine
from .autograd.variable import Variable
from .nn.layers import Linear, ReLU, Sigmoid, Softmax
from .nn.loss import MSELoss, CrossEntropyLoss
from .nn.optimizer import SGD, Adam

__version__ = "0.1.0"
__all__ = [
    "VirtualCoreManager",
    "Scheduler", 
    "Tensor",
    "TensorOps",
    "AutogradEngine",
    "Variable",
    "Linear",
    "ReLU", 
    "Sigmoid",
    "Softmax",
    "MSELoss",
    "CrossEntropyLoss",
    "SGD",
    "Adam"
]