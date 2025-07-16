import torch

import beignet


class SubtractPolynomial:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.subtract_polynomial,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.other = torch.randn(batch_size, dtype=dtype)

    def time_subtract_polynomial(self, batch_size, dtype):
        self.func(self.input, self.other)

    def peak_memory_subtract_polynomial(self, batch_size, dtype):
        self.func(self.input, self.other)
