import numpy as np
from math import exp

"""
For this part, we decided to make a neural network class.
The class will be able to have an adjustable number of layers and nodes in all layers except the output layer.
"""
# part 4a calculates the equations calculated manually in the maths section of the exercise
def part_4a():
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    # Define initial variables
    variables = {
        'x1': 0.04, 'x2': 0.2, 'y': 0.5,
        'w1': 0.11, 'w2': 0.22, 'w3': 0.33, 'w4': 0.44, 'w5': 0.55, 'w6': 0.66
    }

    # Calculate h1, h2, and yo
    h = [sigmoid(variables['w1']*variables['x1'] + variables['w2']*variables['x2']),
        sigmoid(variables['w3']*variables['x1'] + variables['w4']*variables['x2'])]
    yo = sigmoid(variables['w5']*h[0] + variables['w6']*h[1])

    # Calculate mse
    mse = 0.5 * (variables['y'] - yo)**2

    # Calculate derivatives and new weights
    dedw = [
        (yo - variables['y']) * yo * (1 - yo) * variables['w5'] * h[0] * (1 - h[0]) * variables['x1'],
        (yo - variables['y']) * yo * (1 - yo) * variables['w5'] * h[0] * (1 - h[0]) * variables['x2'],
        (yo - variables['y']) * yo * (1 - yo) * variables['w6'] * h[1] * (1 - h[1]) * variables['x1'],
        (yo - variables['y']) * yo * (1 - yo) * variables['w6'] * h[1] * (1 - h[1]) * variables['x2'],
        (yo - variables['y']) * yo * (1 - yo) * h[0],
        (yo - variables['y']) * yo * (1 - yo) * h[1]
    ]

    alpha = 0.4
    new_weights = [variables['w1'] - alpha * dedw[0], variables['w2'] - alpha * dedw[1],
                variables['w3'] - alpha * dedw[2], variables['w4'] - alpha * dedw[3],
                variables['w5'] - alpha * dedw[4], variables['w6'] - alpha * dedw[5]]

    # Calculate h1n and h2n
    hn = [sigmoid(new_weights[0]*variables['x1'] + new_weights[1]*variables['x2']),
        sigmoid(new_weights[2]*variables['x1'] + new_weights[3]*variables['x2'])]

    # Calculate yon and msen
    yon = sigmoid(new_weights[4]*hn[0] + new_weights[5]*hn[1])
    msen = 0.5 * (variables['y'] - yon)**2
    
    # Print results
    for key, value in variables.items():
        print(f"{key} = {value}")

    print(f"h1 = {h[0]}\nh2 = {h[1]}\nyo = {yo}\nmse = {mse}")
    print(f"dedw1 = {dedw[0]}\ndedw2 = {dedw[1]}\ndedw3 = {dedw[2]}\ndedw4 = {dedw[3]}\ndedw5 = {dedw[4]}\ndedw6 = {dedw[5]}")
    print(f"w1n = {new_weights[0]}\nw2n = {new_weights[1]}\nw3n = {new_weights[2]}\nw4n = {new_weights[3]}\nw5n = {new_weights[4]}\nw6n = {new_weights[5]}")
    print(f"h1n = {hn[0]}\nh2n = {hn[1]}\nyon = {yon}\nmsen = {msen}")
    
    

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
                self._weights[nodeNumber][j] -= learning_rate * (error * self.output[0] * (1 - self.output[0]) * prev_or_next_layer.output[j])
        else:
            error = - Y + prev_or_next_layer.output[0]
            for i in range(self._nodes):
                for j in range(len(self._weights[i])):
                    self._weights[i][j] -= learning_rate * (error * prev_or_next_layer.output[0] * (1 - prev_or_next_layer.output[0]) * prev_or_next_layer._weights[0][i] * self.output[i] * (1 - self.output[i]) * X[j])
    
    def activation(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    part_4a()
    part_4()