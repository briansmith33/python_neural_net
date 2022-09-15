from constants import math


class Activation:
    def __init__(self, function):
        self.function = function
        self.relu = lambda n: 0 if n <= 0 else n
        self.step = lambda n: 0 if n <= 0 else 1
        self.sigmoid = lambda n: 1 / (1 + math.e ** -n)
        self.sigderiv = lambda n: n * (1 - n)
        self.softmax = lambda inp=None, n=None: math.e ** n if n is not None else [[value / sum(row) for value in row]
                                                                                   for row in inp]

    def __call__(self, n):
        switch = {
            "relu": self.relu(n),
            "step": self.step(n),
            "sigmoid": self.sigmoid(n),
            "softmax": self.softmax(n=n)
        }
        activation = switch.get(self.function, ValueError(f"'{self.function}' is not a valid activation function"))
        if isinstance(activation, ValueError):
            raise activation
        return activation


if __name__ == "__main__":
    print(Activation('sigmoid')(-3.0))
