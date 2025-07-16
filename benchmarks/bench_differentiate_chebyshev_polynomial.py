import random

import torch

import beignet


class BenchDifferentiateChebyshevPolynomial:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.differentiate_chebyshev_polynomial,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.input = torch.randn(batch_size, 3, dtype=dtype)

        self.order = random.randint(1, 10)

        self.scale = torch.randn(batch_size, dtype=dtype)

        self.axis = torch.randn(batch_size, dtype=dtype)

    def time_differentiate_chebyshev_polynomial(self, batch_size, dtype):
        self.func(self.input, self.order, self.scale, self.axis)

    def peak_memory_differentiate_chebyshev_polynomial(self, batch_size, dtype):
        self.func(self.input, self.order, self.scale, self.axis)
