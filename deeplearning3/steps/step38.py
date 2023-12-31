if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(y)
print(y.grad)
print(x.grad)

# using reshape of Variable
x = Variable(np.random.randn(1, 2, 3))
y = x.reshape((2, 3))
print(y)
y = x.reshape(2, 3)
print(y)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
y.backward()
print(x.grad)

x = Variable(np.random.randn(2, 3))
y = x.transpose()
print(y)
y = x.T
print(y)


x = Variable(np.random.randn(1, 2, 3))
y = x.transpose(2, 0, 1)
y.backward()
print(y)
print(x.grad)