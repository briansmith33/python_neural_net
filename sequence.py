from layer import Dense


class Sequence:
    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers = layers
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss, metrics):
        for layer in self.layers:
            print(isinstance(layer, Dense))

    def fit(self, x, y, epochs, batch_size=64):
        for epoch in range(epochs):
            output = x
            for layer in self.layers:
                layer.forward(output)
                output = layer.output

            for layer in reversed(self.layers):
                layer.back(self.loss, y, output)
                layer.output = []

    def flatten(self, current=None, vector=None):
        if vector is None:
            vector = []
            for i in range(len(self.inputs)):
                self.flatten(self.inputs[i], vector)
        else:
            if not isinstance(current[0], list):
                for val in current:
                    vector.append(val)
                return vector
            for i in range(len(current)):
                self.flatten(current[i], vector)
            return vector
        return vector
