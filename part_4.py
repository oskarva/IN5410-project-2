import numpy as np

"""
For this part, we decided to make a neural network class.
The class will be able to have an adjustable number of layers and nodes in all layers except the output layer.
"""

def part_4():
    train_X = [[0.04, 0.2]] #x1 and x2
    train_Y = [0.5] #y1
    layers = [NN_layer(nodes=2, input_size=2, output_size=2, input_layer=True), 
              NN_layer(nodes=2, input_size=2, output_size=2),   #Hidden layer
              NN_layer(nodes=1, input_size=2, output_size=1, output_layer=True),
              ]
    nn = NN(layers)

class NN:
    def __init__(self, layers) -> None:
        self._layers = layers
        if layers[-1] != 1: #Enforce 1 output node for simplicity
            self._layers[-1] = 1
        self._learning_rate = 0.4

    def fit(self, X, Y):
        return None
    
    def _forward_propagation(self, X):
        return None

    def _back_propagation(self, X, Y):
        return None
    
    def predict(self, test_X):
        predictions = []
        for x in test_X:
            predictions.append(self._forward_propagation(x))
        
        return predictions

class NN_layer:
    def __init__(self, nodes, input_size, output_size, input_layer=False, output_layer=False) -> None:
        self._nodes = nodes
        self._weights = None if input_layer or output_layer \
                        else np.random.rand(nodes, input_size)
        self._input_layer = input_layer
        self._output_layer = output_layer
    
    def _forward_propagation(self, X):
        out = []
        for i in range(self._nodes):
            total = 0
            for j, x in enumerate(X):
                total += self._weights[i][j] * x
            out.append(self.activation(total))
        
        return out


    def _back_propagation(self, X, Y, learning_rate):
        return None
    
    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, test_X, batch_size):
        return None

if __name__ == "__main__":
    part_4()