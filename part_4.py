import numpy as np

"""
For this part, we decided to make a neural network class.
The class will be able to have an adjustable number of layers and nodes in all layers except the output layer.
"""

def part_4():
    train_X = [[0.04, 0.2]] #x1 and x2
    train_Y = [0.5] #y1
    layers = [2, 2, 1]

class NN:
    def __init__(self, layers) -> None:
        self._layers = layers
        if layers[-1] != 1: #Enforce 1 output node for simplicity
            self._layers[-1] = 1
    
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
    def __init__(self):
        self._learning_rate = 0.4
    
    def _forward_propagation(self, X):
        return None

    def _back_propagation(self, X, Y):
        return None
    
    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, test_X, batch_size):
        return None

if __name__ == "__main__":
    part_4()