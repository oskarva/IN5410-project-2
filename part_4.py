import numpy as np

def part_4():
    return None

class NN:
    def __init__(self, layers) -> None:
        self._layers = layers
    
    def fit(self, X, Y):
        return None
    
    def _forward_propagation(self, X):
        return None

    def _back_propagation(self, X, Y):
        return None
    
    def predict(self, test_X):
        return None

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