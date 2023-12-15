if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

# y = x^2
# z = (dy/dx)^3 + y

x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)
gx = x.grad

z = gx ** 3 + y
x.clear_grad()
z.backward()
print(x.grad)

# z = (2x)^3 + x^2
# z = 8x^3 + x^2
# dz/dx = 24x^2 + 2x