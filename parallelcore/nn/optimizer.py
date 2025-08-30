"""
Optimizers for neural network training
"""

import numpy as np
from typing import List, Optional, Dict, Any
import math

from ..autograd.variable import Variable


class Optimizer:
    """Base class for optimizers"""
    
    def __init__(self, parameters: List[Variable], lr: float):
        self.parameters = parameters
        self.lr = lr
        self.state = {}
    
    def step(self):
        """Perform optimization step - to be implemented by subclasses"""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero gradients for all parameters"""
        for param in self.parameters:
            param.zero_grad()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer
    
    Features:
    - Momentum support
    - Weight decay (L2 regularization)
    - Nesterov momentum
    """
    
    def __init__(self, parameters: List[Variable], lr: float = 0.01,
                 momentum: float = 0, weight_decay: float = 0,
                 nesterov: bool = False):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Initialize momentum buffers
        for i, param in enumerate(self.parameters):
            self.state[i] = {
                'momentum_buffer': np.zeros_like(param.data)
            }
    
    def step(self):
        """Perform SGD optimization step"""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Apply momentum
            if self.momentum != 0:
                buf = self.state[i]['momentum_buffer']
                buf = self.momentum * buf + grad
                
                if self.nesterov:
                    grad = grad + self.momentum * buf
                else:
                    grad = buf
                
                self.state[i]['momentum_buffer'] = buf
            
            # Update parameters
            param.tensor._data -= self.lr * grad


class Adam(Optimizer):
    """
    Adam optimizer
    
    Features:
    - Adaptive learning rates
    - Bias correction
    - Weight decay support
    """
    
    def __init__(self, parameters: List[Variable], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize state
        for i, param in enumerate(self.parameters):
            self.state[i] = {
                'step': 0,
                'exp_avg': np.zeros_like(param.data),
                'exp_avg_sq': np.zeros_like(param.data)
            }
    
    def step(self):
        """Perform Adam optimization step"""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.data
            state = self.state[i]
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = self.betas
            
            state['step'] += 1
            
            # Exponential moving average of gradient values
            exp_avg *= beta1
            exp_avg += (1 - beta1) * grad
            
            # Exponential moving average of squared gradient values
            exp_avg_sq *= beta2
            exp_avg_sq += (1 - beta2) * grad * grad
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            
            # Compute step size
            step_size = self.lr / bias_correction1
            
            # Update parameters
            denom = (np.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + self.eps
            param.tensor._data -= step_size * exp_avg / denom


class AdamW(Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay)
    """
    
    def __init__(self, parameters: List[Variable], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize state
        for i, param in enumerate(self.parameters):
            self.state[i] = {
                'step': 0,
                'exp_avg': np.zeros_like(param.data),
                'exp_avg_sq': np.zeros_like(param.data)
            }
    
    def step(self):
        """Perform AdamW optimization step"""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.data
            state = self.state[i]
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = self.betas
            
            state['step'] += 1
            
            # Exponential moving average of gradient values
            exp_avg *= beta1
            exp_avg += (1 - beta1) * grad
            
            # Exponential moving average of squared gradient values
            exp_avg_sq *= beta2
            exp_avg_sq += (1 - beta2) * grad * grad
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            
            # Compute step size
            step_size = self.lr / bias_correction1
            
            # AdamW weight decay (applied directly to parameters)
            if self.weight_decay != 0:
                param.tensor._data -= self.lr * self.weight_decay * param.data
            
            # Update parameters
            denom = (np.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + self.eps
            param.tensor._data -= step_size * exp_avg / denom


class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, parameters: List[Variable], lr: float = 0.01,
                 alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0, momentum: float = 0):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        # Initialize state
        for i, param in enumerate(self.parameters):
            self.state[i] = {
                'square_avg': np.zeros_like(param.data),
                'momentum_buffer': np.zeros_like(param.data) if momentum > 0 else None
            }
    
    def step(self):
        """Perform RMSprop optimization step"""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.data
            state = self.state[i]
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            square_avg = state['square_avg']
            
            # Update squared gradient average
            square_avg *= self.alpha
            square_avg += (1 - self.alpha) * grad * grad
            
            # Compute update
            avg = np.sqrt(square_avg) + self.eps
            
            if self.momentum > 0:
                buf = state['momentum_buffer']
                buf = self.momentum * buf + grad / avg
                param.tensor._data -= self.lr * buf
                state['momentum_buffer'] = buf
            else:
                param.tensor._data -= self.lr * grad / avg