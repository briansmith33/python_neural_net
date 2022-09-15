from activation import Activation
from neuron import Neuron
import numpy as np


class Dense:
    def __init__(self, n_inputs, n_neurons, activation="relu"):
        self.weights = np.random.randn(n_neurons, n_inputs) / 10
        self.biases = [0] * n_neurons
        self.activation = Activation(activation)
        self.output = []

    def forward(self, inputs):
        self.output = [[self.activation(Neuron(inputRow, weightRow, self.biases[i]).output)
                        for i, weightRow in enumerate(self.weights)] for inputRow in inputs]
        #print(self.output)
        if self.activation.function == "softmax":
            self.output = self.activation.softmax(inp=self.output)

    def back(self, loss, actual, predicted):
        scoreArray = []
        y = 0
        for oRow, wRow in zip(predicted, self.weights):
            scoreArray.append([])
            total_error = 0
            j = 0
            errors = []

            for x, w in zip(actual, wRow):
                o = oRow[0]
                score = ((x[0] - o) ** 2) / 2
                scoreArray[y].append(score)
                total_error += score
                derivative = o * (1.0 - o)
                error = (x[0] - o) * derivative
                errors.append(error)
                delta = error * derivative
                error += w * delta
                self.weights[y][j] += error * 0.01 * x
                j += 1
            y += 1


class Flatten:
    def __init__(self, tensor):
        self.tensor = tensor


