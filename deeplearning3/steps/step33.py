if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

# second derivative of y = x^4 - 2x^2 from step29
def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
print(x.grad)

gx = x.grad
x.clear_grad() # reset grad from 1st derivative
gx.backward()
print(x.grad)

# using Newton method
x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.clear_grad()
    y.backward(create_graph=True)

    gx = x.grad
    x.clear_grad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
