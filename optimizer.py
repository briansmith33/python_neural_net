import math


def categorical_cross_entropy(actual, predicted):
    sum_score = 0.0
    for xRow, oRow in zip(actual, predicted):
        for x, o in zip(xRow, oRow):
            sum_score += x * math.log(1e-15 + o)
        mean_sum_score = 1.0 / len(xRow) * sum_score
        print(-mean_sum_score)


class SGD:
    def __init__(self, lr, decay, momentum, nesterov=False):
        self.lr = lr
        self.decay = math.log(decay)
        self.momentum = momentum
        self.nesterov = nesterov
