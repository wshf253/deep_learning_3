import numpy as np
import cupy as cp

x = cp.arange(6).reshape(2, 3)
print(x)
y = x.sum(axis=1)
print(y)

# numpy -> cupy
n = np.array([1, 2, 3])
c = cp.asarray(n)
assert type(c) == cp.ndarray

# cupy -> numpy
c = cp.array([1, 2, 3])
n = cp.asnumpy(c)
assert type(n) == np.ndarray

x = np.array([1, 2, 3])
xp = cp.get_array_module(x) # get_array_module returns np
assert xp == np

x = cp.array([1, 2, 3])
xp = cp.get_array_module(x) # get_array_module returns cp
assert xp == cp


import numpy as np
gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False
from dezero import Variable


def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data
    
    if not gpu_enable: # if no cupy
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data
    
    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data
    
    if not gpu_enable:
        raise Exception('CuPy cannot be loaded. Install CuPy!')
    return cp.asarray(x)