import numpy as np


class Tensor:
    def __init__(self, tensor, shape, dtype):
        self.tensor = tensor
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f"Tensor(\n{np.array(self.tensor)}, shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return len(self.tensor)
