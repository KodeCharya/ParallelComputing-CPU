"""
Variable: Wrapper for tensors with automatic differentiation
"""

import numpy as np
from typing import Optional, Union, List, Callable

from ..tensor.tensor import Tensor


def _as_tensor(x) -> Tensor:
    """Convert Variable/Tensor/ndarray/scalar to Tensor."""
    if isinstance(x, Variable):
        return x.tensor
    if isinstance(x, Tensor):
        return x
    return Tensor(np.asarray(x))


def _unbroadcast(grad_data: np.ndarray, input_shape: tuple) -> np.ndarray:
    """Sum broadcast gradient back to the input shape."""
    # Handle scalar inputs
    if input_shape == ():
        return np.array(grad_data.sum())
    # Sum over leading extra dims
    while grad_data.ndim > len(input_shape):
        grad_data = grad_data.sum(axis=0)
    # Sum over dims where input was size 1 (broadcast)
    for i, dim in enumerate(input_shape):
        if dim == 1 and grad_data.shape[i] != 1:
            grad_data = grad_data.sum(axis=i, keepdims=True)
    if grad_data.shape != input_shape:
        grad_data = np.reshape(grad_data, input_shape)
    return grad_data


class GradientFunction:
    """Represents a function in the computation graph"""

    def __init__(self, function_name: str, inputs: List['Variable'],
                 backward_fn: Optional[Callable] = None):
        self.function_name = function_name
        self.inputs = inputs
        self.backward_fn = backward_fn
        self.saved_values = {}

    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        """Compute gradients for inputs"""
        if self.backward_fn is not None:
            return self.backward_fn(grad_output)
        raise NotImplementedError(
            f"No backward defined for '{self.function_name}'")


class Variable:
    """
    Variable wrapper for tensors with automatic differentiation

    Similar to PyTorch's Variable (now integrated into Tensor)
    """

    def __init__(self, tensor: Union[Tensor, np.ndarray, list, float, int],
                 requires_grad: bool = True, engine=None):
        if isinstance(tensor, Variable):
            tensor = tensor.tensor
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor, requires_grad=requires_grad)
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

    @property
    def T(self) -> 'Variable':
        return self.transpose()

    def __repr__(self) -> str:
        return f"Variable({self.tensor}, requires_grad={self.requires_grad})"

    # -- internal helpers -------------------------------------------------
    def _wrap_result(self, result_tensor: Tensor, requires_grad: bool,
                     op_name: str, inputs: List['Variable'],
                     backward_fn: Optional[Callable] = None) -> 'Variable':
        result = Variable(result_tensor, requires_grad=requires_grad,
                          engine=self.engine)
        # Propagate engine from any input that has one
        if result.engine is None:
            for inp in inputs:
                if isinstance(inp, Variable) and inp.engine is not None:
                    result.engine = inp.engine
                    break
        if requires_grad:
            result._grad_fn = GradientFunction(op_name, inputs, backward_fn)
        return result

    @staticmethod
    def _coerce_other(other) -> tuple:
        """Return (tensor, variable_or_None, requires_grad)."""
        if isinstance(other, Variable):
            return other.tensor, other, other.requires_grad
        if isinstance(other, Tensor):
            return other, None, other.requires_grad
        return Tensor(np.asarray(other)), None, False

    # -- arithmetic --------------------------------------------------------
    def __add__(self, other) -> 'Variable':
        other_t, other_v, other_rg = self._coerce_other(other)
        result_tensor = self.tensor + other_t
        rg = self.requires_grad or other_rg
        inputs = [self] if other_v is None else [self, other_v]
        self_shape = self.shape
        other_shape = other_t.shape

        def _bw(grad: Tensor):
            g = grad._data
            grads = []
            if self.requires_grad:
                grads.append(Tensor(_unbroadcast(g, self_shape)))
            elif len(inputs) == 1:
                grads.append(None)
            if other_v is not None:
                if other_v.requires_grad:
                    grads.append(Tensor(_unbroadcast(g, other_shape)))
                else:
                    grads.append(None)
            return grads

        return self._wrap_result(result_tensor, rg, 'add', inputs, _bw)

    __radd__ = __add__

    def __sub__(self, other) -> 'Variable':
        other_t, other_v, other_rg = self._coerce_other(other)
        result_tensor = Tensor(self.tensor._data - other_t._data)
        rg = self.requires_grad or other_rg
        inputs = [self] if other_v is None else [self, other_v]
        self_shape = self.shape
        other_shape = other_t.shape

        def _bw(grad: Tensor):
            g = grad._data
            grads = []
            if self.requires_grad:
                grads.append(Tensor(_unbroadcast(g, self_shape)))
            elif len(inputs) == 1:
                grads.append(None)
            if other_v is not None:
                if other_v.requires_grad:
                    grads.append(Tensor(_unbroadcast(-g, other_shape)))
                else:
                    grads.append(None)
            return grads

        return self._wrap_result(result_tensor, rg, 'sub', inputs, _bw)

    def __rsub__(self, other) -> 'Variable':
        other_t, _, _ = self._coerce_other(other)
        result_tensor = Tensor(other_t._data - self.tensor._data)
        rg = self.requires_grad
        self_shape = self.shape

        def _bw(grad: Tensor):
            if self.requires_grad:
                return [Tensor(_unbroadcast(-grad._data, self_shape))]
            return [None]

        return self._wrap_result(result_tensor, rg, 'sub', [self], _bw)

    def __mul__(self, other) -> 'Variable':
        other_t, other_v, other_rg = self._coerce_other(other)
        result_tensor = self.tensor * other_t
        rg = self.requires_grad or other_rg
        inputs = [self] if other_v is None else [self, other_v]
        self_data = self.tensor._data.copy()
        other_data = other_t._data.copy()
        self_shape = self.shape
        other_shape = other_t.shape

        def _bw(grad: Tensor):
            g = grad._data
            grads = []
            if self.requires_grad:
                grads.append(Tensor(_unbroadcast(g * other_data, self_shape)))
            elif len(inputs) == 1:
                grads.append(None)
            if other_v is not None:
                if other_v.requires_grad:
                    grads.append(Tensor(_unbroadcast(g * self_data, other_shape)))
                else:
                    grads.append(None)
            return grads

        return self._wrap_result(result_tensor, rg, 'mul', inputs, _bw)

    __rmul__ = __mul__

    def __truediv__(self, other) -> 'Variable':
        other_t, other_v, other_rg = self._coerce_other(other)
        result_tensor = Tensor(self.tensor._data / other_t._data)
        rg = self.requires_grad or other_rg
        inputs = [self] if other_v is None else [self, other_v]
        self_data = self.tensor._data.copy()
        other_data = other_t._data.copy()
        self_shape = self.shape
        other_shape = other_t.shape

        def _bw(grad: Tensor):
            g = grad._data
            grads = []
            if self.requires_grad:
                grads.append(Tensor(_unbroadcast(g / other_data, self_shape)))
            elif len(inputs) == 1:
                grads.append(None)
            if other_v is not None:
                if other_v.requires_grad:
                    grads.append(Tensor(_unbroadcast(
                        -g * self_data / (other_data ** 2), other_shape)))
                else:
                    grads.append(None)
            return grads

        return self._wrap_result(result_tensor, rg, 'div', inputs, _bw)

    def __pow__(self, power) -> 'Variable':
        p = power.data if isinstance(power, Variable) else (
            power._data if isinstance(power, Tensor) else power)
        result_tensor = Tensor(self.tensor._data ** p)
        rg = self.requires_grad
        self_data = self.tensor._data.copy()
        self_shape = self.shape

        def _bw(grad: Tensor):
            if self.requires_grad:
                return [Tensor(_unbroadcast(
                    grad._data * p * (self_data ** (p - 1)), self_shape))]
            return [None]

        return self._wrap_result(result_tensor, rg, 'pow', [self], _bw)

    def __neg__(self) -> 'Variable':
        result_tensor = Tensor(-self.tensor._data)
        rg = self.requires_grad

        def _bw(grad: Tensor):
            if self.requires_grad:
                return [Tensor(-grad._data)]
            return [None]

        return self._wrap_result(result_tensor, rg, 'neg', [self], _bw)

    def __matmul__(self, other) -> 'Variable':
        if isinstance(other, Variable):
            other_t, other_v = other.tensor, other
        elif isinstance(other, Tensor):
            other_t, other_v = other, None
        else:
            other_t, other_v = Tensor(np.asarray(other)), None
        result_tensor = self.tensor.matmul(other_t)
        rg = self.requires_grad or (other_v.requires_grad if other_v else other_t.requires_grad)
        inputs = [self] if other_v is None else [self, other_v]
        self_data = self.tensor._data.copy()
        other_data = other_t._data.copy()

        def _bw(grad: Tensor):
            g = grad._data
            grads = []
            if self.requires_grad:
                if other_data.ndim == 1:
                    gs = g * other_data if g.ndim == 0 else np.matmul(g, other_data) if g.ndim >= 1 else g * other_data
                    # 1-D weight case: grad is outer product
                    if isinstance(gs, np.ndarray) and gs.shape != self_data.shape:
                        gs = np.outer(g, other_data) if g.ndim == 1 else gs
                else:
                    gs = np.matmul(g, other_data.T)
                grads.append(Tensor(_unbroadcast(np.asarray(gs), self_data.shape)))
            elif len(inputs) == 1:
                grads.append(None)
            if other_v is not None:
                if other_v.requires_grad:
                    go = np.matmul(self_data.T, g)
                    grads.append(Tensor(_unbroadcast(np.asarray(go), other_data.shape)))
                else:
                    grads.append(None)
            return grads

        return self._wrap_result(result_tensor, rg, 'matmul', inputs, _bw)

    def transpose(self, axes=None) -> 'Variable':
        result_tensor = self.tensor.transpose(axes)
        rg = self.requires_grad
        inv_axes = None
        if axes is not None:
            inv_axes = np.argsort(axes)

        def _bw(grad: Tensor):
            if self.requires_grad:
                g = grad._data
                if inv_axes is not None:
                    g = np.transpose(g, inv_axes)
                else:
                    g = np.transpose(g)
                return [Tensor(g)]
            return [None]

        return self._wrap_result(result_tensor, rg, 'transpose', [self], _bw)

    # -- activations --------------------------------------------------------
    def relu(self) -> 'Variable':
        result_tensor = self.tensor.relu()
        rg = self.requires_grad
        mask = (self.tensor._data > 0).astype(self.tensor._data.dtype
                                              if self.tensor._data.dtype in (np.float32, np.float64)
                                              else np.float32)

        def _bw(grad: Tensor):
            if self.requires_grad:
                return [Tensor(grad._data * mask)]
            return [None]

        return self._wrap_result(result_tensor, rg, 'relu', [self], _bw)

    def sigmoid(self) -> 'Variable':
        result_tensor = self.tensor.sigmoid()
        rg = self.requires_grad
        sig = result_tensor._data.copy()

        def _bw(grad: Tensor):
            if self.requires_grad:
                return [Tensor(grad._data * sig * (1 - sig))]
            return [None]

        return self._wrap_result(result_tensor, rg, 'sigmoid', [self], _bw)

    def exp(self) -> 'Variable':
        out = np.exp(self.tensor._data)
        result_tensor = Tensor(out)
        rg = self.requires_grad

        def _bw(grad: Tensor):
            if self.requires_grad:
                return [Tensor(grad._data * out)]
            return [None]

        return self._wrap_result(result_tensor, rg, 'exp', [self], _bw)

    def log(self) -> 'Variable':
        out = np.log(self.tensor._data)
        result_tensor = Tensor(out)
        rg = self.requires_grad
        self_data = self.tensor._data.copy()

        def _bw(grad: Tensor):
            if self.requires_grad:
                return [Tensor(grad._data / self_data)]
            return [None]

        return self._wrap_result(result_tensor, rg, 'log', [self], _bw)

    def tanh(self) -> 'Variable':
        out = np.tanh(self.tensor._data)
        result_tensor = Tensor(out)
        rg = self.requires_grad

        def _bw(grad: Tensor):
            if self.requires_grad:
                return [Tensor(grad._data * (1 - out ** 2))]
            return [None]

        return self._wrap_result(result_tensor, rg, 'tanh', [self], _bw)

    # -- reductions ----------------------------------------------------------
    def sum(self, axis: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> 'Variable':
        result_tensor = self.tensor.sum(axis=axis, keepdims=keepdims)
        rg = self.requires_grad
        self_shape = self.shape

        def _bw(grad: Tensor):
            if self.requires_grad:
                g = grad._data
                if axis is None:
                    out = np.ones(self_shape, dtype=g.dtype if hasattr(g, 'dtype') else np.float32) * g
                else:
                    # Broadcast grad back to input shape
                    if not keepdims:
                        shape = list(self_shape)
                        ax = (axis,) if isinstance(axis, int) else tuple(axis)
                        ax = tuple(a if a >= 0 else a + len(self_shape) for a in ax)
                        for a in sorted(ax):
                            g = np.expand_dims(g, axis=a)
                    out = np.broadcast_to(g, self_shape).copy()
                return [Tensor(np.asarray(out))]
            return [None]

        return self._wrap_result(result_tensor, rg, 'sum', [self], _bw)

    def mean(self, axis: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> 'Variable':
        result_tensor = self.tensor.mean(axis=axis, keepdims=keepdims)
        rg = self.requires_grad
        self_shape = self.shape
        # Number of elements averaged
        if axis is None:
            n = int(np.prod(self_shape)) if len(self_shape) > 0 else 1
        else:
            ax = (axis,) if isinstance(axis, int) else tuple(axis)
            n = 1
            for a in ax:
                n *= self_shape[a]

        def _bw(grad: Tensor):
            if self.requires_grad:
                g = grad._data
                if axis is None:
                    out = np.ones(self_shape) * (g / n)
                else:
                    if not keepdims:
                        ax2 = (axis,) if isinstance(axis, int) else tuple(axis)
                        axn = tuple(a if a >= 0 else a + len(self_shape) for a in ax2)
                        for a in sorted(axn):
                            g = np.expand_dims(g, axis=a)
                    out = np.broadcast_to(g / n, self_shape).copy()
                return [Tensor(np.asarray(out, dtype=self.tensor._data.dtype if hasattr(self.tensor._data, 'dtype') else None))]
            return [None]

        return self._wrap_result(result_tensor, rg, 'mean', [self], _bw)

    def backward(self, gradient: Optional[Tensor] = None):
        """Compute gradients via backpropagation"""
        if self.engine is not None:
            self.engine.backward(self, gradient)
        else:
            # Fallback: engine-less reverse-mode using stored graph
            from .engine import AutogradEngine
            AutogradEngine().backward(self, gradient)

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
