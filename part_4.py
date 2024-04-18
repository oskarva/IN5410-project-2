import numpy as np

"""
For this part, we decided to make a neural network class.
The class will be able to have an adjustable number of layers and nodes in all layers except the output layer.
"""

def part_4():
    train_X = [
                [0.04, 0.2] #x1 and x2
              ] 
    train_Y = [0.5] #y1
    w1w2w3w4 = np.array([[0.11, 0.22], [0.33, 0.44]])
    w5w6 = np.array([[0.55, 0.66]])

    layers = [NN_layer(nodes=2, input_size=2, output_size=2, input_layer=True), 
              NN_layer(nodes=2, input_size=2, output_size=2, weights=w1w2w3w4),   #Hidden layer
              NN_layer(nodes=1, input_size=2, output_size=1, output_layer=True, weights=w5w6),
              ]
    nn = NN(layers)

    nn.fit(train_X, train_Y)

    predictions = nn.predict(train_X)
    print( 0.5 * (train_Y[0] - predictions[0][0]) ** 2) #MSE


class NN:
    def __init__(self, layers) -> None:
        self._layers = layers
        if layers[-1]._nodes != 1: #Enforce 1 output node for simplicity
            self._layers[-1]._nodes = 1
        self._learning_rate = 0.4

    def fit(self, X, Y):
        error = None 
        threshold = 0.00001
        #max iterations in case of threshold not being reached?
        while True:
            for x, y in zip(X, Y):
                self._forward_propagation(x)
                self._back_propagation(x, y)
            
            new_error = 0.5 * (y - self._layers[-1].output[0]) ** 2 #MSE
            if error != None and error - new_error <= threshold and new_error <= error: 
                return
            error = new_error
    
    def _forward_propagation(self, X):
        for i, layer in enumerate(self._layers):
            X = layer.forward_propagation(X)

        return X

    def _back_propagation(self, X, Y):
        for i in range(len(self._layers) - 1, 0, -1): #all layers except input layer
            if self._layers[i]._output_layer:
                layer_to_send_with = self._layers[1]
            elif not self._layers[i]._input_layer and not self._layers[i]._output_layer:
                layer_to_send_with = self._layers[2]
            self._layers[i].back_propagation(X, Y, self._learning_rate, layer_to_send_with)
    
    def predict(self, test_X):
        predictions = []
        for x in test_X:
            predictions.append(self._forward_propagation(x))
        
        return predictions

class NN_layer:
    def __init__(self, nodes, input_size, output_size, input_layer=False, output_layer=False, weights=None) -> None:
        np.random.seed(42)
        self._nodes = nodes
        self._weights = None if input_layer \
                        else (
                            weights if isinstance(weights, np.ndarray) \
                            else np.random.rand(nodes, input_size)
                            )
        self._input_layer = input_layer
        self._output_layer = output_layer

        self.output = None
    
    def forward_propagation(self, X):
        if self._input_layer:
            self.output = X
            return X
        
        out = []
        for i in range(self._nodes):
            total = 0
            for j, x in enumerate(X):
                total += self._weights[i][j] * x
            out.append(self.activation(total))
        
        self.output = out
        return out

    def back_propagation(self, X, Y, learning_rate, prev_or_next_layer): 
        if self._input_layer:
            raise Exception("Cannot backpropagate on input layer")

        elif self._output_layer:
            error = - Y + self.output[0]
            nodeNumber = 0 #only one node in output layer
            for j in range(len(self._weights[nodeNumber])):
                #TODO: Minus or plus below?
                self._weights[nodeNumber][j] -= learning_rate * (error * self.output[0] * (1 - self.output[0]) * prev_or_next_layer.output[j])
        else:
            error = - Y + prev_or_next_layer.output[0]
            for i in range(self._nodes):
                for j in range(len(self._weights[i])):
                    #TODO: Minus or plus below?
                    self._weights[i][j] -= learning_rate * (error * prev_or_next_layer.output[0] * (1 - prev_or_next_layer.output[0]) * prev_or_next_layer._weights[0][i] * self.output[i] * (1 - self.output[i]) * X[j])
    
    def activation(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    part_4()