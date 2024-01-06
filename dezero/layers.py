if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero.core import Parameter
from dezero import cuda
import weakref
import numpy as np
import dezero.functions as F
import os
from dezero.utils import pair

class Layer:
    def __init__(self):
        self._params = set()
    
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value) # super() -> can user method from parent class

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer): # get params from obj(Layer)
                yield from obj.params()
            else:
                yield obj
    
    def cleargrads(self):
        for param in self.params():
            param.clear_grad()

    def to_cpu(self):
        for param in self.params():
            # param - Variable
            param.to_cpu()
    
    def to_gpu(self):
        for param in self.params():
            param.to_gpu()
    
    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key) # not self._flatten it will cause recurrsion depth exceeded error, obj - Layer -> obj._f0latten
            else:
                params_dict[key] = obj
    
    def save_weights(self, path):
        self.to_cpu()
        # save as numpy ndarray

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

        
'''
# test
from dezero.core import Variable

layer = Layer()

layer.p1 = Parameter(np.array(1))
layer.p2 = Parameter(np.array(2))
layer.p3 = Variable(np.array(3))
layer.p4 = 'test'

print(layer._params)
print("----------------")

for name in layer._params:
    print(name, layer.__dict__[name]) # layer.__dict__ -> dictonary that has all instance variable, layer.__dict__[name] -> get Parameter instance
'''


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__() # call init of Layer class
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            # when in_size is None, init W at forward, if not None, init W now
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')
    
    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
    
    def forward(self, x):
        # init W when data flows
        if self.W.data is None:
            self.in_size = x.shape[1] # x is 2d matrix
            xp = cuda.get_array_module(x)
            self._init_W(xp)
        
        y = F.linear(x, self.W, self.b)
        return y
    

class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels # C - number of channels
        self.out_channels = out_channels # OC - number of kernels
        self.kernel_size = kernel_size # (KH, KW)
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data
    
    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1] # x.shape - N x C x H x W
            xp = cuda.get_array_module(x)
            self._init_W(xp)
        
        #y = F.conv2d_simple(x, self.W, self.b, self.stride, self.pad)
        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y
    

class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size) # dot(x, Wx)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True) # dot(h(t-1), Wh)
        self.h = None # hidden state

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new
    

class LSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        H, I = hidden_size, in_size
        self.x2f = Linear(H, in_size=I)
        self.x2i = Linear(H, in_size=I)
        self.x2o = Linear(H, in_size=I)
        self.x2u = Linear(H, in_size=I)
        self.h2f = Linear(H, in_size=I, nobias=True)
        self.h2i = Linear(H, in_size=I, nobias=True)
        self.h2o = Linear(H, in_size=I, nobias=True)
        self.h2u = Linear(H, in_size=I, nobias=True)
        self.reset_state()
    
    def reset_state(self):
        self.h = None
        self.c = None
    
    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(self.x2u(x) + self.h2i(self.h))
        
        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)
        
        h_new = o * F.tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new