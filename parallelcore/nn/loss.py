"""
Loss functions for neural networks
"""

import numpy as np
from typing import Optional

from ..autograd.variable import Variable
from ..tensor.tensor import Tensor


class Loss:
    """Base class for loss functions"""
    
    def __init__(self, reduction: str = 'mean'):
        self.reduction = reduction
    
    def __call__(self, input: Variable, target: Variable) -> Variable:
        """Compute loss"""
        return self.forward(input, target)
    
    def forward(self, input: Variable, target: Variable) -> Variable:
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error loss"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)
    
    def forward(self, input: Variable, target: Variable) -> Variable:
        """Compute MSE loss"""
        diff = input - target
        squared_diff = diff * diff
        
        if self.reduction == 'mean':
            return squared_diff.mean()
        elif self.reduction == 'sum':
            return squared_diff.sum()
        else:  # 'none'
            return squared_diff


class CrossEntropyLoss(Loss):
    """Cross-entropy loss for classification"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)
    
    def forward(self, input: Variable, target: Variable) -> Variable:
        """Compute cross-entropy loss"""
        # Apply log-softmax for numerical stability
        log_softmax = self._log_softmax(input)
        
        # Compute negative log-likelihood
        if target.data.ndim == 1:  # Class indices
            # Convert to one-hot if needed
            num_classes = input.shape[-1]
            target_one_hot = np.zeros((target.shape[0], num_classes))
            target_one_hot[np.arange(target.shape[0]), target.data.astype(int)] = 1
            target = Variable(Tensor(target_one_hot), requires_grad=False)
        
        loss = -(target * log_softmax).sum(axis=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
    
    def _log_softmax(self, x: Variable) -> Variable:
        """Numerically stable log-softmax"""
        x_max = Variable(Tensor(np.max(x.data, axis=-1, keepdims=True)))
        x_shifted = x - x_max
        exp_x = Variable(Tensor(np.exp(x_shifted.data)))
        sum_exp = exp_x.sum(axis=-1, keepdims=True)
        log_sum_exp = Variable(Tensor(np.log(sum_exp.data)))
        return x_shifted - log_sum_exp


class L1Loss(Loss):
    """L1 (Mean Absolute Error) loss"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)
    
    def forward(self, input: Variable, target: Variable) -> Variable:
        """Compute L1 loss"""
        diff = input - target
        abs_diff = Variable(Tensor(np.abs(diff.data)), requires_grad=diff.requires_grad)
        
        if self.reduction == 'mean':
            return abs_diff.mean()
        elif self.reduction == 'sum':
            return abs_diff.sum()
        else:  # 'none'
            return abs_diff


class HuberLoss(Loss):
    """Huber loss (smooth L1 loss)"""
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.delta = delta
    
    def forward(self, input: Variable, target: Variable) -> Variable:
        """Compute Huber loss"""
        diff = input - target
        abs_diff = Variable(Tensor(np.abs(diff.data)))
        
        # Huber loss: 0.5 * x^2 if |x| <= delta, delta * (|x| - 0.5 * delta) otherwise
        quadratic = Variable(Tensor(0.5)) * diff * diff
        linear = Variable(Tensor(self.delta)) * (abs_diff - Variable(Tensor(0.5 * self.delta)))
        
        # Use quadratic for small errors, linear for large errors
        mask = abs_diff.data <= self.delta
        loss_data = np.where(mask, quadratic.data, linear.data)
        loss = Variable(Tensor(loss_data), requires_grad=input.requires_grad or target.requires_grad)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss