class Neuron:
    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.output = self.output()

    def output(self):
        output = 0
        for i, w in zip(self.inputs, self.weights):
            output += i * w
        return output + self.bias
