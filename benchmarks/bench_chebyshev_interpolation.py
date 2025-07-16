import random

import torch

import beignet


class BenchChebyshevInterpolation:
    params = [
        [10, 100, 1000, 10000],
        [torch.float32, torch.float64],
    ]

    param_names = ["batch_size", "dtype"]

    def __init__(self):
        self.func = torch.compile(
            beignet.chebyshev_interpolation,
            fullgraph=True,
        )

    def setup(self, batch_size, dtype):
        self.func = torch.randn(batch_size, dtype=dtype)

        self.degree = random.randint(1, 10)

    def time_chebyshev_interpolation(self, batch_size, dtype):
        self.func(self.input, self.func, self.degree)

    def peak_memory_chebyshev_interpolation(self, batch_size, dtype):
        self.func(self.input, self.func, self.degree)
