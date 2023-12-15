import numpy as np
from dezero.core import Function

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    # All variables in forward is ndarray class, use np.sin() function

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)  # call Mul function of Dezero
        return gx
    # All variables in backward is Varaible class, need cos() function of Dezero

def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx
    
def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]() # outputs is weakref
        gx = gy * (1 - y * y) # or (1 - y ** 2)
        return gx
    
def tanh(x):
    return Tanh()(x)