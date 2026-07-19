"""Minimal NumPy-backed shim of the torch API surface used by the bandlet
notebook, for offline correctness testing (no GPU/torch available in this
dev sandbox). Not a general torch replacement."""
import numpy as np
from types import SimpleNamespace
import math as _math

float32 = np.float32
long = np.int64
bool = np.bool_

class _CudaNS:
    @staticmethod
    def is_available():
        return False
    @staticmethod
    def synchronize():
        pass
    @staticmethod
    def get_device_name(i):
        return "shim-no-gpu"

cuda = _CudaNS()

def device(name):
    if isinstance(name, str):
        return SimpleNamespace(type=name)
    return name

def _dt(x):
    return None if x is None else x

class Tensor:
    __array_priority__ = 1000

    def __init__(self, data, dev='cpu'):
        self.data = np.asarray(data)
        self.device = device(dev) if isinstance(dev, str) else dev

    # -- basic props --
    @property
    def shape(self):
        return self.data.shape
    @property
    def dtype(self):
        return self.data.dtype
    @property
    def ndim(self):
        return self.data.ndim

    def to(self, dev):
        t = dev.type if hasattr(dev, 'type') else dev
        return Tensor(self.data, t)

    def clone(self):
        return Tensor(self.data.copy(), self.device.type)

    def transpose(self, d0, d1):
        return Tensor(np.swapaxes(self.data, d0, d1), self.device.type)

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(self.data.reshape(shape), self.device.type)
    reshape = view

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        shape = tuple(self.data.shape[i] if s == -1 else s for i, s in enumerate(shape))
        return Tensor(np.broadcast_to(self.data, shape), self.device.type)

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.data, dim), self.device.type)

    def squeeze(self, dim=None):
        return Tensor(np.squeeze(self.data, axis=dim), self.device.type)

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]
        return Tensor(np.transpose(self.data, dims), self.device.type)

    def abs(self):
        return Tensor(np.abs(self.data), self.device.type)

    def sum(self, dim=None):
        return Tensor(self.data.sum(axis=dim), self.device.type)

    def mean(self, dim=None):
        return Tensor(self.data.mean(axis=dim), self.device.type)

    def float(self):
        return Tensor(self.data.astype(np.float32), self.device.type)

    def long(self):
        return Tensor(self.data.astype(np.int64), self.device.type)

    def item(self):
        return self.data.item()

    def max(self):
        return Tensor(self.data.max(), self.device.type)

    def min(self):
        return Tensor(self.data.min(), self.device.type)

    def clamp(self, lo, hi):
        return Tensor(np.clip(self.data, lo, hi), self.device.type)

    def gather(self, dim, index):
        idx = index.data if isinstance(index, Tensor) else index
        return Tensor(np.take_along_axis(self.data, idx, axis=dim), self.device.type)

    def index_select(self, dim, index):
        idx = index.data if isinstance(index, Tensor) else index
        return Tensor(np.take(self.data, idx, axis=dim), self.device.type)

    def scatter_(self, dim, index, src):
        idx = index.data if isinstance(index, Tensor) else index
        val = src.data if isinstance(src, Tensor) else src
        assert dim == 0 and self.data.ndim == 1, "shim scatter_ only supports the 1-D dim=0 case used here"
        self.data[idx] = val
        return self

    def flatten(self):
        return Tensor(self.data.flatten(), self.device.type)

    def cpu(self):
        return Tensor(self.data, 'cpu')

    def numpy(self):
        return self.data

    def tolist(self):
        return self.data.tolist()

    def numel(self):
        return self.data.size

    def __bool__(self):
        return True if self.data.item() else False

    def __getitem__(self, key):
        key = tuple(k.data if isinstance(k, Tensor) else k for k in key) if isinstance(key, tuple) else \
              (key.data if isinstance(key, Tensor) else key)
        return Tensor(self.data[key], self.device.type)

    def __setitem__(self, key, value):
        key = tuple(k.data if isinstance(k, Tensor) else k for k in key) if isinstance(key, tuple) else \
              (key.data if isinstance(key, Tensor) else key)
        v = value.data if isinstance(value, Tensor) else value
        self.data[key] = v

    def _other(self, o):
        return o.data if isinstance(o, Tensor) else o

    def __add__(self, o): return Tensor(self.data + self._other(o), self.device.type)
    __radd__ = __add__
    def __sub__(self, o): return Tensor(self.data - self._other(o), self.device.type)
    def __rsub__(self, o): return Tensor(self._other(o) - self.data, self.device.type)
    def __mul__(self, o): return Tensor(self.data * self._other(o), self.device.type)
    __rmul__ = __mul__
    def __truediv__(self, o): return Tensor(self.data / self._other(o), self.device.type)
    def __rtruediv__(self, o): return Tensor(self._other(o) / self.data, self.device.type)
    def __neg__(self): return Tensor(-self.data, self.device.type)
    def __le__(self, o): return Tensor(self.data <= self._other(o), self.device.type)
    def __ge__(self, o): return Tensor(self.data >= self._other(o), self.device.type)
    def __lt__(self, o): return Tensor(self.data < self._other(o), self.device.type)
    def __gt__(self, o): return Tensor(self.data > self._other(o), self.device.type)
    def __eq__(self, o): return Tensor(self.data == self._other(o), self.device.type)
    def __pow__(self, o): return Tensor(self.data ** self._other(o), self.device.type)
    def __invert__(self): return Tensor(np.logical_not(self.data), self.device.type)
    def __and__(self, o): return Tensor(np.logical_and(self.data, self._other(o)), self.device.type)
    def __or__(self, o): return Tensor(np.logical_or(self.data, self._other(o)), self.device.type)
    def __repr__(self): return f"Tensor({self.data!r})"


def tensor(data, device='cpu', dtype=None):
    arr = np.array(data, dtype=dtype)
    return Tensor(arr, device.type if hasattr(device, 'type') else device)

def from_numpy(arr):
    return Tensor(arr, 'cpu')

def arange(*args, device='cpu'):
    return Tensor(np.arange(*args), device.type if hasattr(device, 'type') else device)

def zeros(*shape, device='cpu', dtype=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.zeros(shape, dtype=dtype), device.type if hasattr(device, 'type') else device)

def zeros_like(x):
    return Tensor(np.zeros_like(x.data), x.device.type)

def ones(*shape, dtype=None, device='cpu'):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.ones(shape, dtype=dtype), device.type if hasattr(device, 'type') else device)

def full(shape, value, dtype=None, device='cpu'):
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    return Tensor(np.full(shape, value, dtype=dtype), device.type if hasattr(device, 'type') else device)

def empty(*shape, dtype=None, device='cpu'):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.empty(shape, dtype=dtype), device.type if hasattr(device, 'type') else device)

def randn(*shape, device='cpu'):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.random.randn(*shape), device.type if hasattr(device, 'type') else device)

def rand(*shape, device='cpu'):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.random.rand(*shape), device.type if hasattr(device, 'type') else device)

def randint(low, high, size, device='cpu'):
    return Tensor(np.random.randint(low, high, size=size), device.type if hasattr(device, 'type') else device)

def cat(tensors, dim=-1):
    return Tensor(np.concatenate([t.data for t in tensors], axis=dim), tensors[0].device.type)

def stack(tensors, dim=0):
    return Tensor(np.stack([t.data for t in tensors], axis=dim), tensors[0].device.type)

def gather(inp, dim, index):
    return Tensor(np.take_along_axis(inp.data, index.data, axis=dim), inp.device.type)

def remainder(a, b):
    ad = a.data if isinstance(a, Tensor) else a
    bd = b.data if isinstance(b, Tensor) else b
    return Tensor(np.mod(ad, bd), a.device.type if isinstance(a, Tensor) else b.device.type)

def round(x):
    return Tensor(np.round(x.data), x.device.type)

def where(cond, a, b):
    ad = a.data if isinstance(a, Tensor) else a
    bd = b.data if isinstance(b, Tensor) else b
    return Tensor(np.where(cond.data, ad, bd), cond.device.type)

def argmin(x, dim=1):
    return Tensor(np.argmin(x.data, axis=dim), x.device.type)

def atan2(y, x):
    return Tensor(np.arctan2(y.data, x.data), y.device.type)

def exp(x):
    return Tensor(np.exp(x.data), x.device.type)

def sin(x):
    return Tensor(np.sin(x.data), x.device.type)

def sqrt(x):
    return Tensor(np.sqrt(x.data), x.device.type)

def manual_seed(n):
    np.random.seed(n)

def set_grad_enabled(b):
    pass

def meshgrid(*tensors, indexing='ij'):
    arrs = np.meshgrid(*[t.data for t in tensors], indexing=indexing)
    return tuple(Tensor(a, tensors[0].device.type) for a in arrs)

class _TopKResult:
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

def topk(x, k, largest=True):
    d = x.data
    if largest:
        idx = np.argpartition(-d, k - 1)[:k]
    else:
        idx = np.argpartition(d, k - 1)[:k]
    vals = d[idx]
    order = np.argsort(-vals) if largest else np.argsort(vals)
    idx = idx[order]
    vals = vals[order]
    return _TopKResult(Tensor(vals, x.device.type), Tensor(idx, x.device.type))

__version__ = "shim-numpy-backed"

from . import nn  # noqa
