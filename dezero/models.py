from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L


class Model(Layer):
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)
    

# MLP - Multi Layer Perceptron
class MLP(Model):
    def __init__(self, fc_output_size, activation=F.sigmoid):
        # fc_output_size -> list or tuple of out_size for each layer
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_size):
            layer = L.Linear(out_size)
            setattr(self, 'l'+str(i), layer) # add layer to _params
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
    
# ex) model = MLP((10, 1)) -> 2층
# ex) model = MLP((10, 20, 30, 40, 1)) -> 5층