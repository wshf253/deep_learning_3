if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, as_variable
import dezero.functions as F
from dezero.models import MLP

'''
# test for get_item()
x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.get_item(x, 1)
print(y)
y.backward()
print(x.grad)
indices = np.array([0, 0, 1])
y = F.get_item(x, indices)
print(y)
y = x[1]
print(y)
y = x[:, 2]
print(y)
'''

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

model = MLP((10, 3))

x = np.array([[0.2, -0.4]]) # (1, 2)
y = model(x)
p = softmax1d(y)
print(y)
print(p)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0]) # model - (10, 3) -> label (0, 1, 2)
y = model(x)
# loss = F.softmax_corss_entropy_simple(y, t)
loss = F.softmax_cross_entropy(y, t)
print(loss)